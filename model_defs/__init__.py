try :
    from . import stdgcn
    from .helpers import print_n_params
except :
    import model_defs.stdgcn as stdgcn
    from helpers import print_n_params

def get_model(cfg) :
    if cfg.name not in globals():
        raise ValueError(f"Model {cfg.name} not found.")

    if cfg.name == 'stdgcn' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes,
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            d_feat=cfg.d_feat,
            seq_len=cfg.seq_len,
            n_joints=cfg.n_joints,
            dropout=cfg.dropout,
            num_person=cfg.num_person,
            graph=cfg.graph,
            graph_args=cfg.graph_args,
            adaptive=cfg.adaptive,)
    else :
        raise NotImplementedError
