"""Optional frozen-f0 replay constraint for retention diagnostics."""
import math

import torch
import torch.nn.functional as F

from .dataset_io import require


def distillation_settings(protocol):
    settings=protocol.get('replay_distillation')
    if settings is None:
        return None
    require(set(settings)=={'weight','temperature','heads','scope','teacher','cache_dtype'},
            'unknown replay-distillation settings')
    require(settings['scope']=='replay_only' and settings['teacher']=='initial_f0'
            and settings['cache_dtype']=='float32' and settings['heads']==['drivable','road_marking'],
            'unsupported replay-distillation scope')
    require(all(isinstance(settings[k],(float,int)) and math.isfinite(settings[k])
                and settings[k]>0 for k in ['weight','temperature']), 'invalid replay-distillation weight/temperature')
    return settings


def masked_replay_distillation(student, teacher, valid, temperature=1.):
    """Mean head KL(teacher||student), excluding padding; teacher is always detached."""
    require(math.isfinite(temperature) and temperature>0,'invalid distillation temperature')
    require(set(student)==set(teacher)=={'drivable','road_marking'},'distillation heads differ')
    require(valid.dtype==torch.bool and valid.ndim==3 and bool(valid.any()),'empty/invalid distillation mask')
    losses=[]
    for head in ['drivable','road_marking']:
        left,right=student[head],teacher[head]
        require(left.shape==right.shape and left.ndim==4 and left.shape[0]==valid.shape[0]
                and left.shape[2:]==valid.shape[1:], 'distillation shapes differ')
        require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()),'non-finite distillation logits')
        kl=F.kl_div(F.log_softmax(left/temperature,dim=1),
                    F.softmax(right.detach()/temperature,dim=1),reduction='none').sum(dim=1)
        losses.append(kl[valid].mean()*temperature**2)
    return torch.stack(losses).mean()
