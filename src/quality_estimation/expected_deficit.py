"""Expected positive threshold violation under Gaussian predictive moments.

These are severities in target units, not failure probabilities. Unlike a CDF
rank, expected deficit keeps severity information when probabilities saturate.
No GT input is required; operational/annotation-support assumptions still apply.
"""
import numpy as np
from scipy.special import ndtr
from .drive_calibration import spatial_probabilities

def density(z):return np.exp(-.5*z*z)/np.sqrt(2*np.pi)

def expected_deficit(mean,variance,threshold):
    mean=np.asarray(mean,float);variance=np.asarray(variance,float)
    if (variance<0).any() or not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError('finite means and nonnegative variances required')
    sd=np.sqrt(np.maximum(variance,1e-20));difference=threshold-mean;z=difference/sd
    return np.maximum(difference*ndtr(z)+sd*density(z),0.)

def bounded_expected_deficit(mean,variance,threshold):
    """E[(threshold-clip(Q,0,1))+], in [0,threshold]."""
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('unit-interval threshold required')
    return np.clip(expected_deficit(mean,variance,threshold)-expected_deficit(mean,variance,0.),0.,threshold)

def far_deficit_with_near_retained(mean,covariance,near_threshold=.8,far_threshold=.5):
    """E[(far_threshold-F)+ *1(N>=near_threshold)] for a Gaussian pair.

    Uses the moment Gaussian, prior to any temperature/vector probability map.
    The first truncated moment follows integration by parts; independent tasks
    reduce to P(near retained) times the ordinary far expected deficit.
    """
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    probability=spatial_probabilities(mean,covariance,near_threshold,far_threshold)[:,2]
    sd=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),1e-20))
    rho=np.clip(covariance[:,1,2]/(sd[:,1]*sd[:,2]),-.999999,.999999)
    root=np.sqrt(1-rho*rho);zn=(near_threshold-mean[:,1])/sd[:,1];zf=(far_threshold-mean[:,2])/sd[:,2]
    correction=density(zf)*ndtr((rho*zf-zn)/root)-rho*density(zn)*ndtr((zf-rho*zn)/root)
    result=(far_threshold-mean[:,2])*probability+sd[:,2]*correction
    return np.maximum(result,0.)

def bounded_far_deficit_with_near_retained(mean,covariance,near_threshold=.8,far_threshold=.5):
    """E[(far_threshold-clip(F,0,1))+ *1(N>=near_threshold)]."""
    if not 0<=far_threshold<=1:raise ValueError('unit-interval far threshold required')
    positive=far_deficit_with_near_retained(mean,covariance,near_threshold,far_threshold)
    negative=far_deficit_with_near_retained(mean,covariance,near_threshold,0.)
    return np.clip(positive-negative,0.,far_threshold)
