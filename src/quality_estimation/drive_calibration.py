"""Offline calibration primitives for fixed whole-image quality and spatial recall.

ROI recalls are auxiliary offline targets. Online inputs are GP moments/RGB tags.
Four spatial states are mutually exclusive; independent binary calibrators must
not be presented as a joint distribution. Empirical scales give no OOD guarantee.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax, ndtr
from .spatial_risk import calibrate_moments, apply_calibration, gaussian_lower_probability

SPATIAL_CLASSES = ('both_good', 'near_weak_only', 'far_weak_only', 'both_weak')

def spatial_probabilities(mean, covariance, near_threshold=.8, far_threshold=.5):
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    if mean.ndim!=2 or mean.shape[1]!=3 or covariance.shape!=(len(mean),3,3):
        raise ValueError('three-task moment shapes required')
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('finite moments required')
    sd=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),1e-12))
    near_low=ndtr((near_threshold-mean[:,1])/sd[:,1]);far_low=ndtr((far_threshold-mean[:,2])/sd[:,2])
    both=np.array([gaussian_lower_probability([near_threshold,far_threshold],m[1:],c[1:,1:]) for m,c in zip(mean,covariance)])
    p=np.column_stack([1-near_low-far_low+both,near_low-both,far_low-both,both])
    if p.min(initial=0)<-1e-5:raise ValueError('quadrature violated rectangle consistency')
    p=np.clip(p,0,1);return p/p.sum(1,keepdims=True)

def spatial_labels(truth, near_threshold=.8, far_threshold=.5):
    y=np.asarray(truth,float)
    if y.ndim!=2 or y.shape[1]!=3:raise ValueError('three targets required')
    result=np.full(len(y),-1,int);valid=np.isfinite(y[:,1:]).all(1)
    n=y[:,1]<near_threshold;f=y[:,2]<far_threshold
    result[valid & ~n & ~f]=0;result[valid & n & ~f]=1
    result[valid & ~n & f]=2;result[valid & n & f]=3
    return result

def equal_group_weights(groups):
    groups=np.asarray(groups);_,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    return 1/counts[inverse]/len(counts)

def fit_moments(mean,covariance,truth,environments,roads,*,conditional=False,correlation_weight=0.,coverage=.9,min_rows=20,min_roads=2):
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float);truth=np.asarray(truth,float)
    environments=np.asarray(environments);roads=np.asarray(roads)
    global_fit=calibrate_moments(mean,covariance,truth,coverage)
    bias=np.asarray(global_fit['bias']);scale=np.asarray(global_fit['scale'])
    complete=np.isfinite(truth).all(1)
    z=(truth[complete]-mean[complete]-bias)/np.maximum(np.sqrt(np.diagonal(covariance[complete],axis1=1,axis2=2))*scale,1e-8)
    if len(z)>=20 and np.min(z.std(0))>1e-8:
        correlation=np.corrcoef(np.clip(z,-10,10),rowvar=False)
        correlation=.95*correlation+.05*np.eye(3)
    else:correlation=np.eye(3)
    groups={}
    if conditional:
        for key in np.unique(environments):
            ix=environments==key
            if ix.sum()<min_rows or len(set(roads[ix]))<min_roads or np.min(np.isfinite(truth[ix]).sum(0))<min_rows:continue
            local=calibrate_moments(mean[ix],covariance[ix],truth[ix],coverage)
            weight=float(ix.sum()/(ix.sum()+50))
            groups[str(int(key))]=dict(bias=((1-weight)*bias+weight*np.asarray(local['bias'])).tolist(),scale=np.exp((1-weight)*np.log(scale)+weight*np.log(local['scale'])).tolist(),rows=int(ix.sum()),roads=len(set(roads[ix])),shrinkage=weight)
    if not 0<=correlation_weight<=1:raise ValueError('invalid correlation blend')
    return dict(version='drive-moment-calibration-1',global_fit=global_fit,groups=groups,residual_correlation=correlation.tolist(),correlation_weight=correlation_weight,conditional=conditional,coverage=coverage,scope='Empirical held-road calibration; no independent-frame, exchangeability or domain-shift coverage guarantee')

def apply_moments(mean,covariance,environments,calibration):
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float);environments=np.asarray(environments)
    m,c=apply_calibration(mean,covariance,calibration['global_fit']);fallback=np.ones(len(m),bool)
    for key,fit in calibration['groups'].items():
        ix=environments==int(key);fallback[ix]=False
        m[ix],c[ix]=apply_calibration(mean[ix],covariance[ix],fit)
    weight=calibration['correlation_weight']
    if weight:
        sd=np.sqrt(np.maximum(np.diagonal(c,axis1=1,axis2=2),1e-12));corr=c/(sd[:,:,None]*sd[:,None,:])
        corr=(1-weight)*corr+weight*np.asarray(calibration['residual_correlation'])[None]
        c=corr*sd[:,:,None]*sd[:,None,:]
    return m,c,fallback

def fit_probability_map(probability,labels,roads,kind='temperature',regularization=.01):
    p=np.asarray(probability,float);labels=np.asarray(labels);roads=np.asarray(roads)
    if p.ndim!=2 or len(labels)!=len(p) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
        raise ValueError('normalized finite probabilities and matching labels required')
    if kind not in ['identity','temperature','vector']:raise ValueError('unknown probability map')
    valid=(labels>=0)&(labels<p.shape[1]);p=p[valid];labels=labels[valid].astype(int);roads=roads[valid]
    if kind=='identity' or len(set(labels))<2:return dict(kind='identity',temperature=1.,bias=[0.]*p.shape[1],reason='identity requested or fewer than two target classes')
    requested_kind=kind;k=p.shape[1]
    support=np.bincount(labels,minlength=k)
    road_support=[len(set(roads[labels==i])) for i in range(k)]
    if kind=='vector' and (np.min(support)<10 or min(road_support)<2):kind='temperature'
    weights=equal_group_weights(roads);lp=np.log(np.clip(p,1e-8,1))
    def unpack(v):
        t=np.exp(v[0]);bias=np.r_[v[1:],0.] if kind=='vector' else np.zeros(k)
        return t,bias-bias.mean()
    def objective(v):
        t,b=unpack(v);out=softmax(lp/t+b,axis=1)
        return float(-np.sum(weights*np.log(np.clip(out[np.arange(len(labels)),labels],1e-12,1)))+regularization*np.dot(v,v))
    init=np.zeros(k if kind=='vector' else 1)
    result=minimize(objective,init,method='L-BFGS-B',bounds=[(-2.3,2.3)]+[(-4,4)]*(len(init)-1))
    if not result.success:raise RuntimeError('probability optimizer failed: '+str(result.message))
    t,b=unpack(result.x)
    return dict(kind=kind,requested_kind=requested_kind,temperature=float(t),bias=b.tolist(),regularization=regularization,fit_roads=len(set(roads)),fit_rows=len(p),classes_present=np.unique(labels).tolist(),class_rows=support.tolist(),class_roads=road_support,objective=float(result.fun))

def apply_probability_map(probability,calibration):
    p=np.asarray(probability,float)
    return softmax(np.log(np.clip(p,1e-8,1))/calibration['temperature']+np.asarray(calibration['bias']),axis=1)

def fit_monotone_binary(probability,labels,roads):
    p=np.asarray(probability,float);y=np.asarray(labels,int);roads=np.asarray(roads)
    if len(p)!=len(y) or (p<0).any() or (p>1).any() or not np.isfinite(p).all():raise ValueError('invalid binary calibration input')
    if min(np.sum(y==0),np.sum(y==1))<10 or min(len(set(roads[y==v])) for v in [0,1])<2:return dict(kind='identity',slope=1.,intercept=0.,reason='fewer than10 rows or2roads in either class')
    x=logit(np.clip(p,1e-6,1-1e-6));weights=equal_group_weights(roads)
    def objective(v):
        z=np.exp(v[0])*x+v[1]
        return float(np.sum(weights*(np.logaddexp(0,z)-y*z))+.01*np.dot(v,v))
    result=minimize(objective,np.zeros(2),method='L-BFGS-B',bounds=[(-2.3,2.3),(-5,5)])
    if not result.success:raise RuntimeError('binary calibration optimizer failed')
    return dict(kind='monotone_platt',slope=float(np.exp(result.x[0])),intercept=float(result.x[1]),fit_roads=len(set(roads)),fit_rows=len(y))

def apply_monotone_binary(probability,calibration):
    p=np.asarray(probability,float)
    if calibration['kind']=='identity':return p.copy()
    return expit(calibration['slope']*logit(np.clip(p,1e-6,1-1e-6))+calibration['intercept'])
