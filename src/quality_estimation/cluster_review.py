"""GT-free fixed-budget review with explicit environment quotas and audit sampling."""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from scipy.stats import rankdata

def mixture_priority(scores,epistemic_sd,uncertainty_weight):
    """Archive-level exploration ranks; equal uncertainties receive equal ranks."""
    scores=np.asarray(scores,float);sd=np.asarray(epistemic_sd,float)
    if scores.ndim!=1 or sd.shape!=scores.shape or not np.isfinite(scores).all() or not np.isfinite(sd).all() or (sd<0).any() or not 0<=uncertainty_weight<=1:
        raise ValueError('invalid review exploration mixture')
    rank=(rankdata(sd,method='average')-1)/max(len(sd)-1,1) if len(sd) else sd
    return (1-uncertainty_weight)*scores+uncertainty_weight*rank


def review_scores(mean,variance,*,threshold=.7,uncertainty_weight=0.):
    mean=np.asarray(mean,float);variance=np.asarray(variance,float)
    if mean.shape!=variance.shape or not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance<0).any():
        raise ValueError('finite aligned posterior moments required')
    if not 0<=uncertainty_weight<=1:raise ValueError('invalid uncertainty mixture')
    sd=np.sqrt(np.maximum(variance,1e-12))
    probability=ndtr((threshold-mean)/sd)
    sigma_rank=np.argsort(np.argsort(sd,kind='stable'),kind='stable')/max(len(sd)-1,1)
    return (1-uncertainty_weight)*probability+uncertainty_weight*sigma_rank


def select_review(ids,scores,clusters,budget,*,quota_exponent=1.,audit_fraction=0.,seed=261126,audit_rounding='floor'):
    """Within-cluster quality priority; exponent 1 preserves population proportions.

    Exponent 0 allocates equally. Small clusters saturate and remaining slots are
    redistributed. Audit rows are drawn without replacement from the remainder.
    This chooses candidates for GT review, never direct training admission.
    """
    ids=np.asarray(ids);scores=np.asarray(scores,float);clusters=np.asarray(clusters)
    n=len(ids)
    if ids.ndim!=1 or len(set(ids))!=n or scores.shape!=(n,) or clusters.shape!=(n,) or not np.isfinite(scores).all():
        raise ValueError('invalid review population')
    if type(budget) is not int or not 0<=budget<=n or not 0<=quota_exponent<=1 or not 0<=audit_fraction<=1:
        raise ValueError('invalid fixed review budget')
    if audit_rounding not in ('floor','ceil'):raise ValueError('invalid audit rounding')
    audit=int((np.ceil if audit_rounding=='ceil' else np.floor)(budget*audit_fraction));priority=budget-audit
    labels,counts=np.unique(clusters,return_counts=True);weights=counts.astype(float)**quota_exponent
    queues={k:np.flatnonzero(clusters==k)[np.lexsort((ids[clusters==k],-scores[clusters==k]))].tolist() for k in labels}
    taken={k:0 for k in labels};chosen=[]
    while len(chosen)<priority:
        available=[k for k in labels if taken[k]<len(queues[k])]
        k=min(available,key=lambda k:((taken[k]+1)/weights[np.flatnonzero(labels==k)[0]],int(k)))
        chosen.append(queues[k][taken[k]]);taken[k]+=1
    remaining=np.setdiff1d(np.arange(n),chosen,assume_unique=False)
    if audit:chosen.extend(np.random.default_rng(seed).choice(remaining,audit,replace=False).tolist())
    result=np.asarray(chosen,int)
    if len(result)!=budget or len(set(result))!=budget:raise RuntimeError('review budget invariant failed')
    return result
