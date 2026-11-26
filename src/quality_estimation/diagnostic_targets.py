"""Offline FP/FN-aware auxiliary targets and posterior diagnostic probabilities.

Near F1 prevents high recall with many false positives from being called normal.
These are image-band diagnostics, not online ROI IoUs or driving-safety labels.
"""
import numpy as np
from scipy.special import ndtr
from .spatial_risk import gaussian_lower_probability

FOUR_TARGET_NAMES=('whole_frame_lane_iou','near_lane_recall','far_lane_recall','near_lane_f1')

def auxiliary_targets(counts,whole_iou):
    counts=np.asarray(counts);whole_iou=np.asarray(whole_iou,float)
    if counts.shape!=(len(whole_iou),9,3) or (counts<0).any():raise ValueError('aligned nine-region TP/FP/FN counts required')
    near=counts[:,6:8].sum(1);far=counts[:,4:6].sum(1)
    def ratio(a,b):return np.divide(a,b,out=np.full(len(a),np.nan),where=b>0)
    near_recall=ratio(near[:,0],near[:,0]+near[:,2]);far_recall=ratio(far[:,0],far[:,0]+far[:,2])
    near_f1=ratio(2*near[:,0],2*near[:,0]+near[:,1]+near[:,2])
    return np.column_stack([whole_iou,near_recall,far_recall,near_f1])

def near_defect_probability(mean,covariance,threshold=.8):
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    if mean.ndim!=2 or mean.shape[1]!=4 or covariance.shape!=(len(mean),4,4):raise ValueError('four-task posterior required')
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():raise ValueError('nonfinite posterior')
    sd=np.sqrt(np.maximum(covariance[:,3,3],1e-12))
    return ndtr((threshold-mean[:,3])/sd)

def quality_or_near_defect_probability(mean,covariance,quality_threshold=.7,near_f1_threshold=.8):
    """Union probability uses learned task covariance; it is not a sum of alarms."""
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    near=near_defect_probability(mean,covariance,near_f1_threshold)
    low=ndtr((quality_threshold-mean[:,0])/np.sqrt(np.maximum(covariance[:,0,0],1e-12)))
    both=np.array([gaussian_lower_probability([quality_threshold,near_f1_threshold],m[[0,3]],c[np.ix_([0,3],[0,3])]) for m,c in zip(mean,covariance)])
    return np.clip(low+near-both,0,1)

def band_diagnostic_probabilities(mean,covariance,near_threshold=.8,far_threshold=.5):
    """Joint retained/weak image-band states; near uses F1 when available.

    Three-task models retain recall-only semantics and cannot certify precision.
    Four-task models use near F1 to account for both omissions and false positives.
    """
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    if mean.ndim!=2 or mean.shape[1] not in (3,4) or covariance.shape!=(len(mean),mean.shape[1],mean.shape[1]):raise ValueError('three/four task moments required')
    near_index=3 if mean.shape[1]==4 else 1;indices=[near_index,2]
    sd=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),1e-12));pn=ndtr((near_threshold-mean[:,near_index])/sd[:,near_index]);pf=ndtr((far_threshold-mean[:,2])/sd[:,2])
    both=np.array([gaussian_lower_probability([near_threshold,far_threshold],m[indices],c[np.ix_(indices,indices)]) for m,c in zip(mean,covariance)])
    p=np.column_stack([1-pn-pf+both,pn-both,pf-both,both])
    if p.min(initial=0)<-1e-5:raise ValueError('joint diagnostic rectangles are inconsistent')
    p=np.clip(p,0,1);return p/p.sum(1,keepdims=True)
