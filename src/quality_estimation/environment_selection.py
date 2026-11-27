"""Calibrated GT-free review priority with supported environment references.

Environment references alter review priority only. The absolute IoU benchmark
and precision/recall diagnostic labels remain unchanged across environments.
"""
from __future__ import annotations
import numpy as np
from .environment_references import weighted_quantile
from .cluster_review import select_review, mixture_priority


def fit_environment_reference(quality, groups, roads, xy, *, quantile=.25,
                              shrinkage=100., absolute_threshold=.7,
                              eligible=None, min_rows=30, min_roads=2):
    q=np.asarray(quality,float); groups=np.asarray(groups); roads=np.asarray(roads); xy=np.asarray(xy,float)
    if q.ndim!=1 or groups.shape!=q.shape or roads.shape!=q.shape or xy.shape!=(len(q),2) or not np.isfinite(xy).all():
        raise ValueError('invalid environment reference dimensions')
    if not 0<quantile<1 or shrinkage<=0 or not 0<absolute_threshold<1:
        raise ValueError('invalid environment reference parameters')
    use=np.isfinite(q) if eligible is None else np.isfinite(q)&np.asarray(eligible,bool)
    if use.shape!=q.shape or not use.any() or ((q[use]<0)|(q[use]>1)).any():raise ValueError('invalid reference quality')
    def summary(mask):
        _,inverse,count=np.unique(roads[mask],return_inverse=True,return_counts=True)
        return weighted_quantile(q[mask],1/count[inverse],quantile)
    global_quantile=summary(use); result={}
    for group in np.unique(groups):
        mask=use&(groups==group);n=int(mask.sum());nr=len(np.unique(roads[mask]));cells=len(np.unique(np.floor(xy[mask]/5).astype(np.int64),axis=0))
        supported=bool(group>=0 and n>=min_rows and nr>=min_roads and cells>=10)
        value=summary(mask) if supported else None;weight=cells/(cells+shrinkage) if supported else 0.
        threshold=float(np.clip(absolute_threshold+weight*(value-global_quantile),.45,.85)) if supported else absolute_threshold
        result[str(int(group))]=dict(threshold=threshold,rows=n,roads=nr,cells=cells,supported=supported,reference_quantile=value,shrinkage_weight=weight)
    return dict(version='environment-selection-reference-1',absolute_threshold=absolute_threshold,
                global_quantile=global_quantile,quantile=quantile,shrinkage=shrinkage,groups=result,
                interpretation='Distribution-relative review reference; never operational normality or a replacement for fixed absolute quality evaluation.')


def thresholds(reference,groups, *, unsupported=None):
    out=np.array([reference['groups'].get(str(int(g)),{}).get('threshold',reference['absolute_threshold']) for g in np.asarray(groups)])
    if unsupported is not None:out[np.asarray(unsupported,bool)]=reference['absolute_threshold']
    return out


def residual_distribution(quality,mean,scale=None):
    q=np.asarray(quality,float);m=np.asarray(mean,float);s=np.ones_like(m) if scale is None else np.maximum(np.asarray(scale,float),1e-3)
    if q.ndim!=1 or m.shape!=q.shape or s.shape!=q.shape or not np.isfinite(np.column_stack((q,m,s))).all() or (s<=0).any():
        raise ValueError('invalid calibration residuals')
    return np.sort((q-m)/s)


def calibrated_priority(mean,residuals,*,threshold=.7,scale=None,uncertainty_weight=0.):
    m=np.asarray(mean,float);r=np.asarray(residuals,float);s=np.ones_like(m) if scale is None else np.maximum(np.asarray(scale,float),1e-3);t=np.broadcast_to(threshold,m.shape)
    if m.ndim!=1 or s.shape!=m.shape or r.ndim!=1 or not len(r) or not np.isfinite(np.concatenate((m,s,r,t))).all() or np.any(np.diff(r)<0) or (s<=0).any():
        raise ValueError('invalid calibrated priority')
    p=(np.searchsorted(r,(t-m)/s,side='right')+.5)/(len(r)+1)
    return mixture_priority(p,s,uncertainty_weight)


def select_environment_review(ids,scores,groups,budget,*,balanced_floor=0.,audit_fraction=.1,seed=271127):
    """Reserve a balanced floor, then spend the remainder on global risk.

    In contrast to a full per-group quota, partial floors allow environment
    thresholds to affect how the remaining cross-group budget is allocated.
    """
    ids=np.asarray(ids);scores=np.asarray(scores,float);groups=np.asarray(groups)
    if not 0<=balanced_floor<=1 or not 0<=audit_fraction<=1 or type(budget) is not int or not 0<=budget<=len(ids):
        raise ValueError('invalid review mixture')
    # Central validation also covers empty/duplicate IDs and non-finite scores.
    select_review(ids,scores,groups,0)
    audit=int(np.floor(budget*audit_fraction));priority=budget-audit;floor=int(np.floor(priority*balanced_floor))
    chosen=select_review(ids,scores,groups,floor,quota_exponent=0.).tolist()
    remaining=np.setdiff1d(np.arange(len(ids)),chosen)
    order=remaining[np.lexsort((ids[remaining],-scores[remaining]))]
    chosen+=order[:priority-floor].tolist()
    remaining=np.setdiff1d(np.arange(len(ids)),chosen)
    if audit:chosen+=np.random.default_rng(seed).choice(remaining,audit,replace=False).tolist()
    if len(chosen)!=budget or len(set(chosen))!=budget:raise RuntimeError('selection budget changed')
    return np.array(chosen,dtype=int)
