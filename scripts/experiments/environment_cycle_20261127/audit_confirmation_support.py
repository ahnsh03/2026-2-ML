"""Explain frozen test failure without changing any model, policy or test membership."""
from pathlib import Path
import argparse,json
import numpy as np
import cv2
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quality_estimation.environment_review import EnvironmentReview
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest
from quality_estimation.region_feature_views import feature_matrix
from study_data import read,dump,sha,labels


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;C=S/'confirmation';assessment=read(S/'reports/fresh_confirmation_assessment.json');opening=read(S/'protocol/fresh_confirmation_opening_frozen.json');index=read(C/'registry/index_frozen.json');rows=index['fresh_confirmation']['rows']
 if opening['review_manifest_sha256']!=sha(S/'g1_quality/bundle/bundle_manifest.json') or opening['confirmation_index_sha256']!=sha(C/'registry/index_frozen.json'):raise ValueError('do not audit a changed test or policy')
 with np.load(C/'packs_f1/fresh_confirmation.npz') as z:data={k:z[k] for k in z.files}
 fields=['estimated_iou','quality_predictive_scale','review_probability','review_group','review_threshold']
 with np.load(C/'f1_quality_predictions.npz') as z:
  if z['ids'].tolist()!=data['ids'].tolist():raise ValueError('prediction ordering differs')
  pred={name:z[name] for name in fields}
 review=EnvironmentReview(S/'g1_quality/bundle');budget=assessment['quality']['f1']['review'][0]['selected_rows'];chosen,_=review.select(data['ids'],pred,budget);selected=np.isin(np.arange(len(rows)),chosen);q,harm,normal=labels(data);low=np.isfinite(q)&(q<.7)
 if int(low[chosen].sum())!=assessment['quality']['f1']['review'][0]['true_low_iou']:raise ValueError('frozen review selection not reproduced')
 c=data['counts'];whole_gt=(c[:,:,0]+c[:,:,2]).sum(1);whole_pred=(c[:,:,0]+c[:,:,1]).sum(1);near_gt=(c[:,6:8,0]+c[:,6:8,2]).sum(1);near_pred=(c[:,6:8,0]+c[:,6:8,1]).sum(1);near_support=(near_gt>=50)|(near_pred>=50);by_road={}
 for site in np.unique(data['site']):
  mask=data['site']==site;by_road[str(site)]=dict(rows=int(mask.sum()),median_whole_gt_pixels=float(np.median(whole_gt[mask])),median_near_gt_pixels=float(np.median(near_gt[mask])),whole_gt_ge50=int((mask&(whole_gt>=50)).sum()),whole_gt_ge100=int((mask&(whole_gt>=100)).sum()),near_gt_ge50=int((mask&(near_gt>=50)).sum()),near_precision_or_recall_supported=int((mask&near_support).sum()),near_deficits=int((mask&harm).sum()),near_retained=int((mask&normal).sum()),selected=int((mask&selected).sum()),selected_near_deficits=int((mask&selected&harm).sum()),selected_near_retained=int((mask&selected&normal).sum()))
 strata={}
 for name,mask in [('all_defined',np.isfinite(q)),('whole_gt_ge100',np.isfinite(q)&(whole_gt>=100)),('union_ge100',np.isfinite(q)&((whole_gt+whole_pred-c[:,:,0].sum(1))>=100)),('near_supported',np.isfinite(q)&near_support)]:
  positives=mask&low;strata[name]=dict(rows=int(mask.sum()),low_iou_positives=int(positives.sum()),selected_rows=int((mask&selected).sum()),selected_low_iou=int((positives&selected).sum()),low_iou_recall=float((positives&selected).sum()/positives.sum()) if positives.any() else None)
 quality_choice=read(S/'g1_quality/quality_selection_frozen.json');schema=read(C/'packs_f1/schema.json');conditional_quality={}
 for kind in ['GP','RF']:
  rec=quality_choice['selected'][kind];model=SpatialQualityGP.load(S/rec['model']) if kind=='GP' else PortableForest.load(S/rec['model']);x=feature_matrix(data,schema,rec['columns']);mean=model.predict(x)[0][:,0] if kind=='GP' else model.predict(x);mean=np.clip(mean,0,1);conditional_quality[kind]={str(site):float(np.mean(abs(mean[data['site']==site]-q[data['site']==site]))) for site in np.unique(data['site'])}
 result=dict(f1_quality_mae_by_road=conditional_quality,rows=len(rows),budget=budget,selected_near_unsupported=int((selected&~near_support).sum()),selected_near_deficits=int((selected&harm).sum()),near_deficit_support_rows=int(harm.sum()),by_road=by_road,strata=strata,frozen_selection_reproduced=True,gt_used_for_selection=False,policy_changed=False,models_retrained=False,original_acceptance_changed=False,source_sha256=sha(__file__),original_assessment_sha256=sha(S/'reports/fresh_confirmation_assessment.json'),interpretation='Two fresh sections have no supported near markings. Accurate mathematical low IoU on sparse far markings does not establish a useful near-perception failure. Preserve original failed test and hold adoption; add scene/target-availability coverage and a new independent confirmation before policy revision.',future_only=['Stratify marked-road, unmarked-road and far-only scenes before freezing the next test; require multiple supported marked-road sections.','Preserve Q, GT/prediction support and near FP/FN reason codes separately. Existing post-GT support flags are diagnostics, not online GT inputs.','Do not suppress every empty prediction: missing all visible markings is a real failure. Expected marking availability must use RGB/scene evidence and false-positive cases need a separate safeguard.','Weather-only probability embeddings do not encode marking availability. Compare a separate scene representation and eligibility/review-value model before more threshold BO.','Add07:00 examples or revise the lighting-label specification on new Dev data; do not tune on this confirmation.'])
 dump(S/'reports/fresh_support_audit.json',result)
 out=S/'figures';out.mkdir(exist_ok=True);sites=list(by_road);fig,axes=plt.subplots(3,2,figsize=(13,10.5))
 for j,site in enumerate(sites):
  eligible=[i for i,r in enumerate(rows) if r['site']==site and r['weather']==0 and r['hour']==13];i=eligible[len(eligible)//2]
  for k,key in enumerate(['image','semantic']):
   image=cv2.imread(str(Path('/mnt')/rows[i][key]),cv2.IMREAD_COLOR)
   if image is None:raise ValueError('support illustration source missing')
   axes[j,k].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));axes[j,k].axis('off');axes[j,k].set_title(f'Road{j+1} / '+('RGB' if k==0 else 'Native Semantic GT')+f' / near GT={int(near_gt[i])}px',fontsize=10)
 fig.suptitle('Post-test support audit: unmarked scenes and sparse distant targets',fontsize=13);fig.tight_layout();fig.savefig(out/'15_sparse_marking_support.png',dpi=140);plt.close(fig)
 x=np.arange(len(sites));fig,axes=plt.subplots(1,2,figsize=(12,5))
 for dx,key,label,color in [(-.18,'near_gt_ge50','Near GT supported','#527ba5'),(.18,'near_deficits','Near precision/recall deficit','#bf6056')]:axes[0].bar(x+dx,[by_road[s][key] for s in sites],.36,label=label,color=color)
 for dx,key,label,color in [(-.18,'selected','Frozen62-frame review','#527ba5'),(.18,'selected_near_deficits','Selected near deficit','#bf6056')]:axes[1].bar(x+dx,[by_road[s][key] for s in sites],.36,label=label,color=color)
 for ax in axes:ax.set(xticks=x,xticklabels=[f'Road{i+1}' for i in x],ylabel='Frames');ax.legend(fontsize=9)
 axes[0].set_title('Target and diagnostic support');axes[1].set_title('Where the fixed review budget went');fig.tight_layout()
 for suffix in ['png','svg']:fig.savefig(out/('16_review_support_failure.'+suffix),dpi=180,bbox_inches='tight')
 plt.close(fig);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
