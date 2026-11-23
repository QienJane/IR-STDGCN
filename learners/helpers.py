import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tempfile
import sys
import os.path as osp
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils

class Teacher_v1(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        mode = self.training
        self.eval()

        with torch.no_grad():

            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        self.train(mode=mode)

        if threshold is not None:
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            ymax, y = torch.max(y_hat, dim=1)

            return y_hat, y

    def generate_scores_pen(self, x):

        mode = self.training
        self.eval()

        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        self.train(mode=mode)

        return y_hat


class Teacher_v2(nn.Module):
    def __init__(self, solver, sample_shape, iters, num_inverted_class, num_known_classes, deep_inv_params, train=True, config=None):
        super().__init__()
        self.solver = solver
        self.solver.eval().requires_grad_(False)
        self.is_ir = config is not None and config.increm.learner.type == 'ir'
        self.sample_shape = sample_shape
        self.iters = iters
        self.config = config

        # hyperparameters
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        if self.is_ir:
            self.inversion_l2_weight = config.increm.learner.get('inversion_l2_weight', 1e-4)
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        self.inv_mean = deep_inv_params[5]
        self.inv_std = deep_inv_params[6]

        # get class keys
        self.num_known_classes = num_known_classes
        self.num_inverted_class = num_inverted_class
        self.num_k =  self.num_inverted_class + self.num_known_classes

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()

    def inversion_loss(self, inputs, logits, targets, bn_hooks):
        if self.is_ir:
            # [B, T, V, C]: sum temporal variances over joints and coordinates.
            variance = inputs.var(dim=1, unbiased=False).sum(dim=(1, 2)).mean()
            input_norm = inputs.flatten(1).norm(dim=1).mean()
            loss = (self.criterion(logits, targets) + self.di_var_scale * variance
                    + self.inversion_l2_weight * input_norm)
        else:
            loss = self.criterion(logits / self.content_temp, targets) * self.content_weight
        if bn_hooks:
            loss = loss + self.r_feature_weight * torch.stack([hook.r_feature for hook in bn_hooks]).mean()
        return loss

    def generate_inverted_samples(self, class_size, inversion_batch_size, dataloader_batch_size, log_dir_task, inverted_sample_dir):
        device = next(self.solver.parameters()).device
        first_class = 0 if self.is_ir else self.num_known_classes
        all_targets = torch.arange(first_class, self.num_k, device=device).repeat(class_size)
        size = all_targets.numel()
        self.solver.eval()
        bn_hooks = [DeepInversionFeatureHook(module, self.is_ir) for module in self.solver.modules()
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]

        pool = {'pts': [], 'target': [], 'output': []}
        log_dir = osp.join(log_dir_task, 'inverted_samples')
        if self.is_ir:
            inverted_sample_dir = log_dir
            # The logger clears its own directory on restart. Keep the replay pool alongside it.
            log_dir = osp.join(log_dir, 'events')
        tb_logger = utils.TensorBoardLogger(log_dir)
        n_correct = 0
        try:
            batches = all_targets.split(inversion_batch_size)
            for batch_id, targets in enumerate(tqdm(batches, desc='Inversion batches')):
                inputs = torch.empty((len(targets), *self.sample_shape[1:]), device=device)
                inputs.normal_(self.inv_mean, self.inv_std).requires_grad_()
                optimizer = optim.Adam([inputs], lr=self.di_lr)
                for iteration in range(self.iters):
                    outputs = self.solver(inputs)
                    logits = outputs[:, :self.num_k] if self.is_ir else outputs
                    loss = self.inversion_loss(inputs, logits, targets, bn_hooks)
                    correct = (logits.argmax(dim=1) == targets).sum().item()
                    if iteration > 0 and 100 * correct / len(targets) > self.config.increm.learner.inverted_acc_thred:
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    # Save predictions of the final optimized inputs, including the last update.
                    with torch.no_grad():
                        outputs = self.solver(inputs)
                        logits = outputs[:, :self.num_k] if self.is_ir else outputs
                        correct = (logits.argmax(dim=1) == targets).sum().item()

                n_correct += correct
                tb_logger.update({'inverted_training_loss': loss.item()}, step=batch_id, prefix='loss')
                tb_logger.update({'inverted_training_acc': 100 * correct / len(targets)}, step=batch_id, prefix='acc')
                pool['pts'].extend(inputs.detach().cpu().numpy())
                pool['target'].extend(targets.cpu().numpy())
                pool['output'].extend(outputs.detach().cpu().numpy())
                # An early stop skips backward on this forward pass. Release its graph before the next batch.
                del outputs, logits, loss
                for hook in bn_hooks:
                    hook.r_feature = None
        finally:
            tb_logger.close()
            for hook in bn_hooks:
                hook.close()
        print(f"Acc inverted samples = {100 * n_correct / size:.2f}")

        os.makedirs(inverted_sample_dir, exist_ok=True)
        fpath = osp.join(inverted_sample_dir, 'saved_inverted_sample.pkl')
        if not self.is_ir and self.num_known_classes:
            saved = utils.load_pickle(fpath)
            for key in pool:
                pool[key].extend(saved[key])
        with tempfile.TemporaryDirectory(dir=inverted_sample_dir) as scratch:
            pending = osp.join(scratch, 'pool.pkl')
            utils.save_pickle(pending, pool)
            os.replace(pending, fpath)

        self._set_replay_pool(pool, dataloader_batch_size)

    def load_inverted_samples(self, path, dataloader_batch_size):
        self._set_replay_pool(utils.load_pickle(path), dataloader_batch_size)

    def _set_replay_pool(self, pool, dataloader_batch_size):
        self.inverted_dataset = InvertedDataset(pool['pts'], pool['target'], pool['output'])
        device = next(self.solver.parameters()).device
        self.inverted_dataloader = DataLoader(self.inverted_dataset, batch_size=dataloader_batch_size, shuffle=True,
                                               num_workers=self.config.workers, pin_memory=device.type == 'cuda')
        self.inverted_iterator = iter(self.inverted_dataloader)

    def sample(self):
        try:
            inputs, targets, outputs = next(self.inverted_iterator)
        except StopIteration:
            self.inverted_iterator = iter(self.inverted_dataloader)
            inputs, targets, outputs = next(self.inverted_iterator)
        return inputs, targets, outputs



    def generate_scores(self, x, allowed_predictions=None, return_label=False):
        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():

             y_hat = self.solver.forward_feature(x)

        return y_hat


class DeepInversionFeatureHook():

    def __init__(self, module, is_ir):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.is_ir = is_ir

    def hook_fn(self, module, inputs, output):
        if not torch.is_grad_enabled():
            return
        x = inputs[0]
        axes = (0, *range(2, x.ndim))
        var, mean = torch.var_mean(x, dim=axes, unbiased=False)
        mean_diff = module.running_mean - mean
        if self.is_ir:
            self.r_feature = mean_diff.norm() + (module.running_var - var).norm()
        else:
            var = var + 1e-8
            running_var = module.running_var + 1e-8
            self.r_feature = 0.5 * (torch.log(var / running_var) + (running_var + mean_diff.square()) / var - 1).mean()

    def close(self):
        self.hook.remove()

class InvertedDataset(Dataset):
    def __init__(self, pts, targets, outputs):
        self.pts = pts
        self.targets = targets
        self.outputs = outputs

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        x_i = self.pts[idx]
        y = np.array(self.targets[idx])
        y_hat = self.outputs[idx]
        x_i = torch.from_numpy(x_i)
        y = torch.from_numpy(y)
        y_hat = torch.from_numpy(y_hat)

        return x_i, y, y_hat
