from easydict import EasyDict as edict

import torch

def get_scheduler(optimizer, cfg):
    steps_per_epoch = cfg['n_steps_per_epoch'] if cfg.get('step_per_batch') else 1
    warmup_steps = cfg.get('warmup_epochs', 0) * steps_per_epoch
    milestones = cfg.get('milestones', [])
    gamma = cfg.get('gamma', 0.1)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        epoch = step // steps_per_epoch + 1
        return gamma ** sum(epoch >= milestone for milestone in milestones)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def step_scheduler(scheduler):
    if scheduler is not None:
        scheduler.step()

def get_optimizer_scheduler(model, cfg) :
    cfg = edict(cfg)

    model_params = []
    if isinstance(model, (list, tuple)) :
        for m in model :
            model_params.append(list(m.parameters()))
    else :
        model_params = model.parameters()

    if cfg.name == 'adam' :
        lr = cfg.lr if 'lr' in cfg else 0.001
        betas = cfg.betas if 'betas' in cfg else (0.9, 0.999)
        optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model_params),
                        lr=lr,
                        betas=betas,
        )

    elif cfg.name == 'sgd' :
        assert 'lr' in cfg
        lr = cfg.lr if 'lr' in cfg else 0.001
        momentum = cfg.momentum if 'momentum' in cfg else 0.9
        weight_decay = cfg.weight_decay if 'weight_decay' in cfg else 0.0001
        optimizer = torch.optim.SGD(
                        model_params,
                        lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        nesterov=cfg.get('nesterov', False),
        )

    else :
        raise NotImplementedError

    scheduler = None
    if 'scheduler' in cfg :
        cfg_sch = {
            'total_epochs': cfg.total_epochs,
            'n_steps_per_epoch': cfg.n_steps_per_epoch,
            **cfg.scheduler
        }

        scheduler = get_scheduler(optimizer, cfg_sch)

    return optimizer, scheduler


def optimizer_to_cuda(optimizer, rank) :
    for state in optimizer.state.values() :
        for k, v in state.items() :
            if torch.is_tensor(v) :
                state[k] = v.cuda(rank)
