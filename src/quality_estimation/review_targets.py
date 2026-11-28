"""Offline, support-aware error targets; no ROI IoU is required online.

Zero means no supported error event under this review definition, not proof of
safe or normal perception. Whole-frame IoU remains a separate physical target.
"""
import numpy as np

ERROR_TARGET_NAMES = ('near_supported_error', 'far_supported_error', 'offroad_supported_fp')
UTILITY_WEIGHTS = np.array([1., .25, .5])


def support_aware_targets(counts, support_pixels=50, correct_fraction=.8):
    counts=np.asarray(counts)
    if counts.ndim!=3 or counts.shape[1:]!=(9,3) or not np.isfinite(counts).all() or (counts<0).any():
        raise ValueError('finite nonnegative ROI9 TP/FP/FN counts required')
    if support_pixels<=0 or not 0<correct_fraction<1:
        raise ValueError('invalid support/error definition')
    whole=counts.sum(1);union=whole.sum(1);gt=whole[:,0]+whole[:,2]
    quality=np.divide(whole[:,0],union,out=np.full(len(counts),np.nan),where=union>0)

    def band(indices):
        value=counts[:,indices].sum(1);tp,fp,fn=value.T;gt_=tp+fn;pred=tp+fp
        missed=(gt_>=support_pixels)&(tp<correct_fraction*gt_)
        false=(pred>=support_pixels)&(tp<correct_fraction*pred)
        return dict(gt_pixels=gt_,predicted_pixels=pred,false_negative=missed,false_positive=false,
                    supported=(gt_>=support_pixels)|(pred>=support_pixels),error=missed|false)

    near=band([6,7]);far=band([4,5]);offroad=band([0,1,2,3,8])
    errors=np.column_stack([near['error'],far['error'],offroad['false_positive']]).astype(float)
    utility=errors@UTILITY_WEIGHTS
    scene=np.where(near['gt_pixels']>=support_pixels,2,np.where(gt>=100,1,0)).astype(int)
    return dict(whole_frame_iou=quality,error_targets=errors,review_utility=utility,
                scene_gt_class=scene,whole_gt_pixels=gt,near=near,far=far,offroad=offroad,
                target_names=ERROR_TARGET_NAMES)


def gaussian_utility(mean, covariance, weights=UTILITY_WEIGHTS):
    """Exact linear posterior propagation for the untruncated GP task scores."""
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float);weights=np.asarray(weights,float)
    if mean.ndim!=2 or weights.shape!=(mean.shape[1],) or covariance.shape!=(len(mean),mean.shape[1],mean.shape[1]):
        raise ValueError('aligned task means/covariances/weights required')
    if not all(np.isfinite(value).all()for value in [mean,covariance,weights]):
        raise ValueError('nonfinite posterior')
    variance=np.einsum('i,nij,j->n',weights,covariance,weights)
    if (variance < -1e-9).any():raise ValueError('negative posterior contrast variance')
    return mean@weights,np.sqrt(np.maximum(variance,0))
