"""Describe frozen uncertainty/error semantics without fitting or policy changes."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import sha256_file
from quality_estimation.review_targets import support_aware_targets


def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path)as z:return {key:z[key]for key in z.files}
def auc(labels,score):
    labels=np.asarray(labels,bool);score=np.asarray(score,float)
    if not np.isfinite(score).all():raise ValueError('Nonfinite frozen diagnostic score')
    return float(roc_auc_score(labels,score))if labels.any()and not labels.all()else None
def correlation(left,right):
    return float(spearmanr(left,right).statistic)if len(left)>2 and np.ptp(left)>0 and np.ptp(right)>0 else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['freeze','review-assessment','output']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();freeze=read(a.freeze);review=read(a.review_assessment)
    if review['confirmation_freeze_sha256']!=sha256_file(a.freeze):raise ValueError('Different freeze')
    specs={c['name']:c for c in freeze['review_candidates']}
    pack=Path(specs['primary']['pack']);receipt=read(pack.with_suffix('.json'))
    if receipt['sha256']!=sha256_file(pack)or receipt['index_sha256']!=review['source_index_sha256']:raise ValueError('Pack/index changed')
    data=load(pack);paths={name:a.review_assessment.parent/(name+'_predictions.npz')for name in ['primary','GP_only']}
    predictions={name:load(p)for name,p in paths.items()}
    for name,pred in predictions.items():
        if(pred['ids'].tolist()!=data['ids'].tolist()or specs[name]['perception_identity']!=specs['primary']['perception_identity']or sha256_file(Path(specs[name]['bundle'])/'bundle_manifest.json')!=specs[name]['bundle_manifest_sha256']):raise ValueError('Prediction/bundle identity mismatch')
    primary=predictions['primary'];gp=predictions['GP_only']
    if not gp['quality_gp_uncertainty_available'].all()or not gp['review_gp_uncertainty_available'].all():raise ValueError('GP uncertainty unavailable')
    if not np.array_equal(primary['estimated_iou'],gp['estimated_iou']):raise ValueError('Compared quality heads differ')
    target=support_aware_targets(data['counts']);near=target['near'];far=target['far']
    retained=(near['gt_pixels']>=50)&(near['predicted_pixels']>=50)&~near['error']
    far_only=retained&far['error'];both=near['error']&far['error'];pair=far_only|both
    scores=dict(quality_predictive_sd=gp['quality_predictive_scale'],gp_utility_latent_sd=gp['utility_latent_sd'],gp_near_minus_far_mean=gp['near_minus_far_error_score'],rf_near_minus_far_mean=primary['near_minus_far_error_score'])
    contrast_sd=gp['near_minus_far_error_latent_sd']
    scores['gp_near_minus_far_z']=np.divide(gp['near_minus_far_error_score'],contrast_sd,out=np.zeros(len(contrast_sd)),where=contrast_sd>0)
    valid=np.isfinite(target['whole_frame_iou']);absolute=np.abs(primary['estimated_iou']-target['whole_frame_iou'])
    groups=dict(near_and_far_error=both,near_error_only=near['error']&~far['error'],near_retained_far_error=far_only,no_supported_error=target['review_utility']==0)
    result=dict(version='frozen-uncertainty-semantics-diagnostic-1',rows=len(data['ids']),perception_identity=specs['primary']['perception_identity'],
        comparisons={name:dict(near_error_auc=auc(near['error'],value),near_and_far_vs_near_retained_far_auc=auc(both[pair],value[pair]),spearman_vs_absolute_q_error=correlation(value[valid],absolute[valid]))for name,value in scores.items()},
        pair_counts=dict(near_and_far_error=int(both.sum()),near_retained_far_error=int(far_only.sum())),
        groups={name:dict(rows=int(mask.sum()),medians={key:float(np.median(value[mask]))if mask.any()else None for key,value in scores.items()})for name,mask in groups.items()},
        sdk_hours={str(int(hour)):dict(rows=int((data['hour']==hour).sum()),near_errors=int(((data['hour']==hour)&near['error']).sum()))for hour in np.unique(data['hour'])},
        inputs=dict(freeze_sha256=sha256_file(a.freeze),review_assessment_sha256=sha256_file(a.review_assessment),pack_sha256=sha256_file(pack),prediction_sha256={name:sha256_file(path)for name,path in paths.items()}),
        fitted_on_confirmation=False,adoption_or_policy_changed=False,source_sha256=sha256_file(Path(__file__)),
        scope='Descriptive post-assessment diagnosis of fixed GP/RF outputs, not a new validated selector. Higher SD is tested as the proposed near-and-far-error indicator; AUC direction is never flipped. Gaussian latent contrast/z is not a calibrated failure probability. Supported error is not physical visibility or safety truth. No thresholds are fitted.')
    a.output.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    selected=[(label,mask)for label,mask in [('Near+far error',both),('Near retained, far error',far_only),('No supported error',groups['no_supported_error'])]if mask.any()]
    for ax,key,title in [(axes[0],'quality_predictive_sd','Q predictive SD'),(axes[1],'gp_utility_latent_sd','GP utility latent SD')]:
        ax.boxplot([scores[key][mask]for _,mask in selected],showfliers=False);ax.set_xticks(range(1,len(selected)+1),[label+'\nn='+str(int(mask.sum()))for label,mask in selected]);ax.tick_params(axis='x',labelsize=8);ax.set_title(title);ax.set_ylabel('Frozen model scale');ax.grid(axis='y',alpha=.15)
    names=['quality_predictive_sd','gp_utility_latent_sd','gp_near_minus_far_mean','rf_near_minus_far_mean','gp_near_minus_far_z']
    values=[result['comparisons'][name]['near_and_far_vs_near_retained_far_auc']for name in names]
    axes[2].bar(range(5),[np.nan if value is None else value for value in values],color=['#9299A2','#7285AD','#357BAC','#B38145','#528459']);axes[2].set_xticks(range(5),['Q SD','Utility SD','GP mean\ncontrast','RF mean\ncontrast','GP contrast\nz'],fontsize=8);axes[2].axhline(.5,color='#555555',linestyle='--');axes[2].set_ylim(0,1);axes[2].set_ylabel('AUC: near+far error vs near retained/far error');axes[2].set_title('Mean error contrast and uncertainty differ')
    fig.suptitle('Frozen new-confirmation uncertainty diagnosis | no refit, new thresholds or policy changes\nPixel-support error groups; same map and one optimizer seed; not physical visibility/safety labels',fontsize=11)
    outputs=[]
    for ext in ['png','svg']:
        path=a.output/('16_gp_uncertainty_semantics.'+ext);fig.savefig(path,dpi=200,bbox_inches='tight');outputs.append(path)
    plt.close(fig);result['figures']={path.name:sha256_file(path)for path in outputs}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['comparisons'],indent=2))


if __name__=='__main__':main()
