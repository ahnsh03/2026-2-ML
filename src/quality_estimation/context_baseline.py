"""Training-only environment-mean controls for per-image quality regression."""
from __future__ import annotations
import numpy as np


def fit_context_mean(quality, groups, roads, *, prior_rows=50., min_rows=30, min_roads=2):
    q=np.asarray(quality,float);groups=np.asarray(groups);roads=np.asarray(roads)
    if q.ndim!=1 or groups.shape!=q.shape or roads.shape!=q.shape or prior_rows<0 or min_rows<1 or min_roads<1:
        raise ValueError('invalid context-mean training data')
    valid=np.isfinite(q)
    if not valid.any() or ((q[valid]<0)|(q[valid]>1)).any():raise ValueError('defined IoU training targets required')
    q=q[valid];groups=groups[valid];roads=roads[valid];mean=float(q.mean());values={};support={}
    for group in np.unique(groups):
        take=groups==group;n=int(take.sum());r=len(np.unique(roads[take]));key=str(int(group));support[key]=dict(rows=n,roads=r)
        if n>=min_rows and r>=min_roads:values[key]=float((q[take].sum()+prior_rows*mean)/(n+prior_rows))
    return dict(global_mean=mean,means=values,support=support,defined_training_rows=len(q),prior_rows=prior_rows,min_rows=min_rows,min_roads=min_roads)


def predict_context_mean(model, groups):
    groups=np.asarray(groups)
    if groups.ndim!=1:raise ValueError('one-dimensional context groups required')
    return np.array([model['means'].get(str(int(g)),model['global_mean']) for g in groups],float)
