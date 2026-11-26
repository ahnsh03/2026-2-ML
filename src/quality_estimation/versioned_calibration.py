"""Generic held-road Gaussian calibration for three/four explicit diagnostic targets.

Empirical calibration is not a conformal or domain-shift coverage guarantee.
Scales cannot shrink predictive variance; calibration never changes raw MAE points.
"""
import numpy as np
from scipy.special import ndtri

def _validate(mean,covariance,truth=None):
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    if mean.ndim!=2 or mean.shape[1] not in (3,4) or covariance.shape!=(len(mean),mean.shape[1],mean.shape[1]):raise ValueError('three/four target posterior required')
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.transpose(0,2,1),atol=1e-9):raise ValueError('invalid Gaussian moments')
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:raise ValueError('covariance is not PSD')
    if truth is not None and np.asarray(truth).shape!=mean.shape:raise ValueError('truth columns differ')
    return mean,covariance

def fit_road_calibration(mean,covariance,truth,roads,*,bias_mode='median',correlation_blend=0.,coverage=.9):
    mean,covariance=_validate(mean,covariance,truth);truth=np.asarray(truth,float);roads=np.asarray(roads)
    if roads.shape!=(len(mean),) or bias_mode not in ('zero','median') or not 0<=correlation_blend<=1 or not 0<coverage<1:raise ValueError('invalid calibration recipe')
    sigma=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),1e-12));bias=[];scales=[];support=[];quantiles=[]
    for j in range(mean.shape[1]):
        valid=np.isfinite(truth[:,j]);residual=truth[:,j]-mean[:,j]
        if valid.sum()<20 or len(set(roads[valid]))<2:raise ValueError('insufficient independent calibration roads')
        offset=float(np.median(residual[valid])) if bias_mode=='median' else 0.
        ratios=np.abs(residual-offset)/sigma[:,j];byroad={str(g):float(np.quantile(ratios[(roads==g)&valid],coverage,method='higher')/ndtri((1+coverage)/2)) for g in np.unique(roads[valid]) if ((roads==g)&valid).sum()>=10}
        factor=max([1.]+list(byroad.values()));bias.append(offset);scales.append(factor);quantiles.append(byroad);support.append(int(valid.sum()))
    complete=np.isfinite(truth).all(1);corr=np.eye(mean.shape[1])
    if complete.sum()>=20:
        residual=(truth[complete]-mean[complete]-np.array(bias))/np.maximum(sigma[complete]*np.array(scales),1e-8)
        if np.min(residual.std(0))>1e-9:corr=.9*np.corrcoef(np.clip(residual,-10,10),rowvar=False)+.1*np.eye(mean.shape[1])
    return dict(version='versioned-road-calibration-1',bias=bias,scale=scales,counts=support,coverage=coverage,bias_mode=bias_mode,
                correlation_blend=correlation_blend,residual_correlation=corr.tolist(),road_scale_quantiles=quantiles,variance_floor='scale>=1',
                scope='Worst observed calibration-road standardized residual quantile; no independence, exchangeability, or unseen-domain coverage guarantee.')

def apply_road_calibration(mean,covariance,spec):
    mean,covariance=_validate(mean,covariance);bias=np.asarray(spec['bias']);scale=np.asarray(spec['scale'])
    if bias.shape!=(mean.shape[1],) or scale.shape!=bias.shape or not np.isfinite(bias).all() or not np.isfinite(scale).all() or (scale<1).any():raise ValueError('invalid nonshrinking calibration')
    result=covariance*scale[None,:,None]*scale[None,None,:]
    weight=spec['correlation_blend']
    if weight:
        residual=np.asarray(spec['residual_correlation'],float)
        if residual.shape!=(mean.shape[1],mean.shape[1]) or not np.isfinite(residual).all() or np.linalg.eigvalsh(residual).min()< -1e-9:raise ValueError('invalid residual correlation')
        sd=np.sqrt(np.maximum(np.diagonal(result,axis1=1,axis2=2),1e-12));corr=result/(sd[:,:,None]*sd[:,None,:])
        result=((1-weight)*corr+weight*residual[None])*sd[:,:,None]*sd[:,None,:]
    return mean+bias,result

def gaussian_nll(mean,covariance,truth):
    mean,covariance=_validate(mean,covariance,truth);truth=np.asarray(truth,float);values=np.full(len(mean),np.nan)
    for i in range(len(mean)):
        observed=np.flatnonzero(np.isfinite(truth[i]))
        if not len(observed):continue
        residual=truth[i,observed]-mean[i,observed];cov=covariance[i][np.ix_(observed,observed)]+np.eye(len(observed))*1e-10
        sign,logdet=np.linalg.slogdet(cov)
        if sign<=0:raise ValueError('nonpositive predictive covariance')
        values[i]=.5*(len(observed)*np.log(2*np.pi)+logdet+residual@np.linalg.solve(cov,residual))
    return values

def select_road_calibration(mean,covariance,truth,roads):
    roads=np.asarray(roads);records=[]
    for bias in ['zero','median']:
        for weight in [0.,.5,1.]:
            losses=[];coverage=[]
            for road in np.unique(roads):
                held=roads==road;fit=~held
                spec=fit_road_calibration(mean[fit],covariance[fit],truth[fit],roads[fit],bias_mode=bias,correlation_blend=weight)
                m,c=apply_road_calibration(mean[held],covariance[held],spec);nll=gaussian_nll(m,c,truth[held]);valid=np.isfinite(nll)
                if valid.any():losses.append(float(nll[valid].mean()))
                valid_y=np.isfinite(truth[held]);covered=abs(truth[held]-m)<=1.6448536269514722*np.sqrt(np.maximum(np.diagonal(c,axis1=1,axis2=2),1e-12))
                coverage.append([float(covered[valid_y[:,j],j].mean()) if valid_y[:,j].any() else None for j in range(mean.shape[1])])
            records.append(dict(bias_mode=bias,correlation_blend=weight,macro_road_nll=float(np.mean(losses)),fold_coverage90=coverage))
    best=min(records,key=lambda r:(r['macro_road_nll'],r['correlation_blend'],r['bias_mode']!='zero'))
    spec=fit_road_calibration(mean,covariance,truth,roads,bias_mode=best['bias_mode'],correlation_blend=best['correlation_blend'])
    return dict(spec,selection='Leave-one-road-out Gaussian NLL among6 recipes; raw quality MAE points unchanged.',trials=records)
