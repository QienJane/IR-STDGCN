import torch
import torch.nn.functional as F
import torchmetrics
from easydict import EasyDict as edict

import optimizers as optimizer_defs
import utils
from .base import Base


def real_batch(data, class_mapping, device):
    pts = data['pts'].to(device)
    labels = [class_mapping[str(label)] for label in data['label'].tolist()]
    return pts, torch.tensor(labels, dtype=torch.long, device=device)


class IncrementalRegularization(Base):
    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super().__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)
        defaults = {
            'domain_consistency': {'weight': 1.0},
            'category_boundary_separation': {'weight': 0.5, 'class_weight': None, 'ignore': None},
            'smooth_regularization': {'weight': 0.1},
        }
        configured = cfg.get('loss', defaults)
        cfg.loss = edict({
            name: {**params, **configured[name]}
            for name, params in defaults.items() if name in configured
        })
        if not any(term.weight for term in cfg.loss.values()):
            raise ValueError("IR training requires at least one active incremental loss.")

        self.inversion_replay = False
        self.power_iters = self.cfg.increm.learner.power_iters
        self.deep_inv_params = self.cfg.increm.learner.deep_inv_params
        self.gen_inverted_samples = False

    def train_epoch(self, tbar, epoch, train_logger, current_t_index):
        losses = {name: utils.AverageMeter() for name in self.criteria}
        device = next(self.model.parameters()).device
        acc_meter_global = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).to(device)
        self.model.train()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.train_sampler.epoch + 1)
        if tbar is not None:
            tbar.reset(total=len(self.train_loader))
            tbar.refresh()

        step_per_batch = self.cfg.optimizer.get('scheduler', {}).get('step_per_batch', False)
        active = {name: term for name, term in self.criteria.items() if term.weight}
        for bi, data in enumerate(self.train_loader, start=1):
            pts, target = real_batch(data, self.cfg.class_mapping, device)
            if current_t_index == 0:
                output = self.model(pts)[:, :self.valid_out_dim]
                loss = F.cross_entropy(output, target)
                if 'category_boundary_separation' in losses:
                    losses['category_boundary_separation'].update(loss.item(), len(pts))
                combined_targets = target
            else:
                old_pts, old_target = self.sample(self.previous_teacher)[:2]
                combined_pts = torch.cat((pts, old_pts.to(device)))
                combined_targets = torch.cat((target, old_target.to(device)))
                features, frame_feats = self.model.forward_feature(combined_pts, return_frame_features=True)

                # L_CBS and accuracy use all seen classes. Only the linear head receives its gradient.
                output = self.model.final(features.detach())[:, :self.valid_out_dim]
                old_fraction = self.last_valid_out_dim / self.valid_out_dim
                new_fraction = 1 - old_fraction
                values = {}
                if 'domain_consistency' in active:
                    # L_DC supervises only the real sequences and the current new class logits.
                    new_logits = self.model.final(features[:len(pts)])[:, self.last_valid_out_dim:self.valid_out_dim]
                    dc = active['domain_consistency'].func
                    if new_logits.size(1) == 1:
                        values['domain_consistency'] = dc(new_logits, torch.ones_like(new_logits), new_fraction, binary=True)
                    else:
                        values['domain_consistency'] = dc(new_logits, target - self.last_valid_out_dim, new_fraction)
                if 'category_boundary_separation' in active:
                    weights = output.new_full((len(combined_targets),), old_fraction)
                    weights[:len(pts)] = new_fraction
                    values['category_boundary_separation'] = active['category_boundary_separation'].func(
                        output, combined_targets, weights)
                if 'smooth_regularization' in active:
                    values['smooth_regularization'] = active['smooth_regularization'].func(frame_feats)

                loss = sum(active[name].weight * value for name, value in values.items())
                for name, value in values.items():
                    count = len(pts) if name == 'domain_consistency' else len(combined_targets)
                    losses[name].update(value.item(), count)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_acc_global = acc_meter_global(output.detach(), combined_targets) * 100
            if step_per_batch:
                optimizer_defs.step_scheduler(self.scheduler)
            if tbar is not None:
                tbar.update()
                tbar.set_postfix({'it': bi, 'loss': loss.item(), 'train_acc_global': train_acc_global.item()})
                tbar.refresh()

        if train_logger is not None:
            train_logger.update({'learning_rate': self.optimizer.param_groups[0]['lr']},
                                step=epoch, prefix="stepwise")
            train_logger.update({name: meter.avg for name, meter in losses.items()}, step=epoch, prefix="loss")
            train_logger.update({'global': acc_meter_global.compute() * 100}, step=epoch, prefix="acc")
            train_logger.flush()

    @torch.no_grad()
    def validate_epoch(self, vbar, epoch, val_logger):
        device = next(self.model.parameters()).device
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).to(device)
        ce_meter = utils.AverageMeter()
        self.model.eval()
        if vbar is not None:
            vbar.reset(total=len(self.val_loader))
            vbar.refresh()

        for bi, data in enumerate(self.val_loader, start=1):
            pts, target = real_batch(data, self.cfg.class_mapping, device)
            output = self.model(pts)[:, :self.valid_out_dim]
            loss = F.cross_entropy(output, target)
            ce_meter.update(loss.item(), len(pts))
            val_acc = acc_meter(output, target) * 100
            if vbar is not None:
                vbar.update()
                vbar.set_postfix({'it': bi, 'loss': loss.item(), 'val_acc': val_acc.item()})
                vbar.refresh()

        metrics = {name: ce_meter.avg if name == 'category_boundary_separation' else 0. for name in self.criteria}
        accuracy = acc_meter.compute() * 100
        if val_logger is not None:
            val_logger.update(metrics, step=epoch, prefix="loss")
            val_logger.update({'global': accuracy}, step=epoch, prefix="acc")
            val_logger.flush()
        return {**metrics, 'acc': accuracy}

    def sample(self, teacher):
        return teacher.sample()

    def generate_inverted_samples(self, teacher, size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir):
        return teacher.generate_inverted_samples(size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir)
