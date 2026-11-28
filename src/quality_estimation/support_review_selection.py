"""Budgeted supported-error review with distinct exploitation, information and audit."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .environment_selection import select_environment_review


def conditional_variance_audit(ids,covariance,observed,count,noise_variance):
    ids=np.asarray(ids);cov=np.asarray(covariance,float)
    if (ids.ndim!=1 or len(set(ids.tolist()))!=len(ids) or cov.shape!=(len(ids),len(ids))
            or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-8,rtol=0)
            or (np.diag(cov)<-1e-8).any() or not np.isfinite(noise_variance)or noise_variance<=0):
        raise ValueError('valid latent utility covariance and positive observation noise required')
    observed=np.asarray(observed,int)
    if (observed.ndim!=1 or len(set(observed.tolist()))!=len(observed)or (observed<0).any()
            or (observed>=len(ids)).any()or type(count)is not int or not 0<=count<=len(ids)-len(observed)):
        raise ValueError('invalid information-audit budget or conditioning indices')
    posterior=cov.copy();available=np.ones(len(ids),bool);available[observed]=False
    if len(observed):
        cross=cov[:,observed];factor=cho_factor(cov[np.ix_(observed,observed)]+noise_variance*np.eye(len(observed)),lower=True)
        posterior-=cross@cho_solve(factor,cross.T)
    selected=[]
    for _ in range(count):
        candidates=np.flatnonzero(available);variance=np.maximum(np.diag(posterior)[candidates],0)
        index=int(candidates[np.lexsort((ids[candidates],-variance))[0]])
        column=posterior[:,index].copy();denominator=max(float(posterior[index,index]),0)+noise_variance
        posterior-=np.outer(column,column)/denominator;posterior=(posterior+posterior.T)*.5
        selected.append(index);available[index]=False
    return np.array(selected,int)


def select_supported_review(ids,mean,sd,groups,budget,*,beta=0.,balanced_floor=0.,
                            audit_fraction=.1,information_fraction=0.,information_mode='pointwise',
                            covariance=None,noise_variance=None,seed=281128):
    ids=np.asarray(ids);mean=np.asarray(mean,float);sd=np.asarray(sd,float);groups=np.asarray(groups)
    if (mean.shape!=ids.shape or sd.shape!=ids.shape or groups.shape!=ids.shape or ids.ndim!=1
            or not np.isfinite(mean).all()or not np.isfinite(sd).all()or(sd<0).any()
            or not np.isfinite(beta)or not 0<=audit_fraction<=1 or not 0<=information_fraction<=1
            or audit_fraction+information_fraction>1 or type(budget)is not int or not 0<=budget<=len(ids)
            or information_mode not in ['pointwise','conditional']):raise ValueError('invalid supported-review inputs')
    score=mean+beta*sd;n_random=int(np.floor(budget*audit_fraction));n_information=int(np.floor(budget*information_fraction))
    n_core=budget-n_random-n_information
    core=select_environment_review(ids,score,groups,n_core,balanced_floor=balanced_floor,audit_fraction=0.,seed=seed)
    remaining=np.setdiff1d(np.arange(len(ids)),core)
    if n_information and information_mode=='conditional':
        if covariance is None or noise_variance is None:raise ValueError('conditional information audit needs latent covariance')
        information=conditional_variance_audit(ids,covariance,core,n_information,noise_variance)
    else:information=remaining[np.lexsort((ids[remaining],-sd[remaining]))[:n_information]]
    remaining=np.setdiff1d(remaining,information)
    random=np.random.default_rng(seed).choice(remaining,n_random,replace=False)if n_random else np.array([],int)
    selected=np.r_[core,information,random]
    floor=int(np.floor(n_core*balanced_floor));reasons=['environment_floor']*floor+['utility_priority']*(n_core-floor)
    reasons += ['conditional_information_audit'if information_mode=='conditional'else'uncertainty_audit']*n_information
    reasons += ['random_audit']*n_random
    probability=[1.]*(n_core+n_information)
    if n_random:probability += [n_random/len(remaining)]*n_random
    if len(selected)!=budget or len(set(selected.tolist()))!=budget:raise RuntimeError('supported-review budget changed')
    return selected,score,dict(reasons=reasons,conditional_inclusion_probability=probability,
        priority_slots=n_core,information_slots=n_information,random_audit_slots=n_random,
        scope='Closed full buffer; covariance conditioning anticipates labels without reading their values.')
