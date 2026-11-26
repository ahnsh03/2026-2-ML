"""Evaluate frozen review calibration under predeclared additional conditions."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,numpy as np
from quality_estimation.drive_review import DriveReviewPredictor
from feature_study import read,save,sha,load,subset,macro
from calibrate import spatial_labels,support,marginal_metrics,probability_metrics
from review_policy import evaluate

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','extension','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();n=a.study;e=a.extension;a.output.mkdir(parents=True,exist_ok=False);seal=read(n/'test_freeze_receipt.json')
 for name,digest in seal['artifacts'].items():assert sha(n/name)==digest
 assert read(e/'plans/routes_frozen.json')['created_utc']<seal['created_utc']
 index=read(e/'registry/index_full.json');thin=read(e/'registry/index_thinned.json');data,schema=load(e/'packs',list(index));ids={r['sample_id'] for v in thin.values() for r in v['rows']};data=subset(data,np.isin(data['ids'],list(ids)))
 c=read(n/'calibration_v1/calibration_frozen.json');p=DriveReviewPredictor(n/'feature_study',n/'calibration_v1',n/'review_policy_v1/policy_frozen.json',checkpoint_sha256=c['checkpoint_sha256'],regions_sha256=c['regions_sha256'],simulator_release='26.R1')
 pred=p.predict_features({key:data[key] for key in ['rgb','output','relations','uncertainty','morphology']},schema);labels=spatial_labels(data['Y']);labels[~support(data)]=-1
 results={}
 for key,mean,cov in [('raw',pred['raw_mean'],pred['raw_covariance']),('calibrated',pred['mean'],pred['covariance'])]:results[key]=marginal_metrics(mean,cov,data)
 reviewdata=dict(truth=data['Y'],counts=data['counts'],near_gt=data['near_gt'],far_gt=data['far_gt'],roads=data['site'],ids=data['ids'])
 review={name:evaluate(score,reviewdata) for name,score in [('calibrated_global',pred['global_probability']),('frozen_policy',pred['priority'])]}
 np.savez_compressed(a.output/'predictions.npz',**pred,ids=data['ids'],roads=data['site'],truth=data['Y'],labels=labels,weather=data['weather'],hour=data['hour'],counts=data['counts'],near_gt=data['near_gt'],far_gt=data['far_gt'])
 save(a.output/'results.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),plan_sha256=sha(e/'plans/routes_frozen.json'),main_seal_sha256=sha(n/'test_freeze_receipt.json'),rows=len(labels),roads=len(set(data['site'])),marginals=results,spatial=probability_metrics(pred['spatial'],labels,data['site']),ood_environment_rows=int((pred['environment']<0).sum()),conditional_fallback_rows=int(pred['conditional_fallback'].sum()),review=review,scope='Predeclared extra conditions on the same main-test geometry, absent from current calibration conditions. Not a new-road replicate or wholly unseen GP-training environment. No refit/adoption selection.'))
 print('condition-shift evaluation complete',len(labels),results,flush=True)
if __name__=='__main__':main()
