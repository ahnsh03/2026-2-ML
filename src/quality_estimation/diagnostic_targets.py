"""Offline FP/FN-aware auxiliary targets and posterior diagnostic probabilities.

Near F1 prevents high recall with many false positives from being called normal.
These are image-band diagnostics, not online ROI IoUs or driving-safety labels.
"""
import numpy as np
from scipy.special import ndtr
from .spatial_risk import gaussian_lower_probability

FOUR_TARGET_NAMES=('whole_frame_lane_iou','near_lane_recall','far_lane_recall','near_lane_f1')


def gt_review_flags(counts,whole_iou,*,quality_threshold=.7,near_threshold=.8,far_recall_threshold=.5,minimum_band_pixels=50):
    """Post-GT review taxonomy; no operating-failure or visibility certification.

    Recall support uses GT pixels; precision support uses predicted pixels so a
    large hallucinated lane can be diagnosed even when no GT lane is present.
    """
    c=np.asarray(counts)
    if c.shape!=(9,3) or c.dtype.kind not in 'ui' or (c<0).any():raise ValueError('nine-region integer TP/FP/FN counts required')
    if not all(0<v<1 for v in [quality_threshold,near_threshold,far_recall_threshold]) or not isinstance(minimum_band_pixels,int) or minimum_band_pixels<1:raise ValueError('invalid diagnostic thresholds/support')
    near=c[6:8].sum(0);far=c[4:6].sum(0);whole=c.sum(0)
    union=int(whole.sum());exact_q=float(whole[0]/union) if union else None
    defined=whole_iou is not None and np.isfinite(whole_iou)
    if defined and (exact_q is None or abs(float(whole_iou)-exact_q)>1e-6):raise ValueError('whole IoU and TP/FP/FN counts differ')
    ng,npred=int(near[0]+near[2]),int(near[0]+near[1]);fg=int(far[0]+far[2])
    nr=float(near[0]/ng) if ng else None;precision=float(near[0]/npred) if npred else None;fr=float(far[0]/fg) if fg else None
    recall_low=nr<near_threshold if ng>=minimum_band_pixels else None
    precision_low=precision<near_threshold if npred>=minimum_band_pixels else None
    far_low=fr<far_recall_threshold if fg>=minimum_band_pixels else None
    near_supported=recall_low is not None or precision_low is not None
    near_defect=bool(recall_low or precision_low) if near_supported else None
    retained=bool(nr>=near_threshold and precision>=near_threshold) if ng>=minimum_band_pixels and npred>=minimum_band_pixels else None
    supported_whole=int(whole[0]+whole[2])>=2*minimum_band_pixels
    union_supported=int(whole.sum())>=2*minimum_band_pixels
    return dict(valid_for_low_iou_benchmark=bool(supported_whole and whole_iou is not None and np.isfinite(whole_iou)),
                exact_whole_iou=exact_q if defined else None,
                benchmark_low_iou=bool(exact_q<quality_threshold) if supported_whole and defined else None,
                valid_for_union_supported_benchmark=bool(union_supported and whole_iou is not None and np.isfinite(whole_iou)),
                union_supported_low_iou=bool(exact_q<quality_threshold) if union_supported and defined else None,
                near_omission=recall_low,near_false_positive_issue=precision_low,
                near_precision_or_recall_reduced=near_defect,far_omission=far_low,
                near_precision_and_recall_retained=retained,
                near_retained_far_reduced=bool(retained and far_low) if retained is not None and far_low is not None else None,
                quality_threshold=float(quality_threshold),near_precision_recall_threshold=float(near_threshold),far_recall_threshold=float(far_recall_threshold),
                minimum_band_pixels=int(minimum_band_pixels),operating_failure_established=False,
                scope='Offline image-band diagnostic after verified GT. Low IoU or reduced far recall alone does not establish operating failure or visibility-normality.')

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
