from typing import Optional

from easydict import EasyDict as edict
import math

try :
    from . import ir_distillation
    from . import domain_consistency
    from . import category_boundary_separation
    from . import distillation
    from . import bce
    from . import smooth_regularization
except :
    import category_boundary_separation
    import distillation
    import bce
    import losses.ir_distillation as ir_distillation
    import domain_consistency
    import smooth_regularization
def _dict_to_list(d) :
    max_k = max(d.keys())
    assert len(d) == max_k + 1
    return [d[k] for k in range(max_k+1)]


def get_losses(cfg: dict, n_classes: int, class_freq: Optional[dict] = None) :
    criteria = edict()
    for lname in cfg :
        assert lname in globals(), f"{lname} not found."
        criteria[lname] = edict()
        criteria[lname].weight = cfg[lname].weight
        if lname == 'category_boundary_separation' :
            weights = None
            if cfg[lname].class_weight == 'class_freq' :
                raise NotImplementedError
                
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                n_classes,
                cfg[lname].ignore,
                weights,
            )
        elif lname == 'distillation' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()  

        elif lname == 'ir_distillation' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()  

        elif lname == 'bce' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].reduction
            )    
        elif lname == 'domain_consistency' :
            criteria[lname].func = getattr(globals()[lname], 'Loss')()
        elif lname == 'smooth_regularization' :
            criteria[lname].func = getattr(globals()[lname], 'SmoothLoss')()
        else :
            raise NotImplementedError

    return criteria