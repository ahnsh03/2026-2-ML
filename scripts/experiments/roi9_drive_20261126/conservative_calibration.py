"""Retire supported followup into calibration; freeze before a new confirmation.

Whole-IoU moments remain raw. Near/far scales cannot shrink; each is the maximum
roadwise90% normalized-residual scale. Two bias choices use road-held-out NLL.
This is empirical group-aware calibration, not a conformal/OOD guarantee.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,shutil
import numpy as np
from quality_estimation.drive_calibration import apply_moments,spatial_probabilities
from quality_estimation.drive_review import DriveReviewPredictor
from feature_study import load,read,save,sha,subset
from calibrate import spatial_labels,marginal_metrics,probability_metrics

def fit(mean,cov,truth,roads,bias_kind):
 groups=np.unique(roads);bias=np.zeros(3);scale=np.ones(3);z90=1.6448536269514722;sd=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))
 for j in [1,2]:
  if bias_kind=='road_median':bias[j]=np.median([np.median(truth[roads==g,j]-mean[roads==g,j]) for g in groups])
  scale[j]=max(1.,max(float(np.quantile(abs(truth[roads==g,j]-mean[roads==g,j]-bias[j])/np.maximum(sd[roads==g,j]*z90,1e-10),.9,method='higher')) for g in groups))
 return dict(version='drive-moment-calibration-2',global_fit=dict(bias=bias.tolist(),scale=scale.tolist(),coverage=.9,counts=np.isfinite(truth).sum(0).tolist(),scope='Q unchanged. NF floor1 +maximum per-roadq90 normalized residual scale; empirical, no OOD coverage guarantee.'),groups={},conditional=False,correlation_weight=0.,residual_correlation=np.eye(3).tolist(),coverage=.9,bias_kind=bias_kind)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False)
 assert not (n/'confirmation/runs').exists()
 sources=[(n/'registry_calibration_final',n/'packs_calibration'),(n/'marked_support/registry',n/'marked_support/packs')];parts=[]
 for registry,packs in sources:
  index=read(registry/'index_full.json');thin=read(registry/'index_thinned.json');data,schema=load(packs,list(index));ids={r['sample_id'] for v in thin.values() for r in v['rows']};whole=data['counts'][:,:,[0,2]].sum((1,2));keep=np.isin(data['ids'],list(ids))&(data['near_gt']>=100)&(data['far_gt']>=100)&(whole>=500)&np.isfinite(data['Y']).all(1);parts.append(subset(data,keep))
 data={key:np.concatenate([part[key] for part in parts]) for key in parts[0]};roads=data['site'];assert len(set(roads))>=4
 parent=read(n/'calibration_v1/calibration_frozen.json');predictor=DriveReviewPredictor(n/'feature_study',n/'calibration_v1',n/'review_policy_v1/policy_frozen.json',checkpoint_sha256=parent['checkpoint_sha256'],regions_sha256=parent['regions_sha256'],simulator_release='26.R1')
 prediction=predictor.predict_features({key:data[key] for key in ['rgb','output','relations','uncertainty','morphology']},schema);mean=prediction['raw_mean'];cov=prediction['raw_covariance'];labels=spatial_labels(data['Y']);records=[]
 for kind in ['zero','road_median']:
  pm=np.zeros_like(mean);pc=np.zeros_like(cov);folds=[]
  for held in np.unique(roads):
   ix=roads==held;fitcal=fit(mean[~ix],cov[~ix],data['Y'][~ix],roads[~ix],kind);pm[ix],pc[ix],_=apply_moments(mean[ix],cov[ix],np.zeros(ix.sum()),fitcal);folds.append(dict(held_road=str(held),calibration=fitcal))
  records.append(dict(bias_kind=kind,spatial=probability_metrics(spatial_probabilities(pm,pc),labels,roads),marginals=marginal_metrics(pm,pc,data),folds=folds))
 eligible=[r for r in records if min(r['marginals'][key]['coverage90'] for key in ['near_recall','far_recall'])>=.8] or records
 winner=min(eligible,key=lambda r:(r['spatial']['nll'],r['spatial']['brier'],r['bias_kind']!='zero'));moments=fit(mean,cov,data['Y'],roads,winner['bias_kind'])
 shutil.copyfile(n/'calibration_v1/environment_profile.json',a.output/'environment_profile.json');save(a.output/'global_moments_reference.json',moments)
 membership=dict(ids=data['ids'].tolist(),roads=roads.tolist(),sources=[dict(index_sha256=sha(registry/'index_thinned.json'),packs={key:sha(packs/(key+'.npz')) for key in read(registry/'index_full.json')}) for registry,packs in sources],scope='Original calibration plus retired supported followup; main prospective and condition-shift test excluded.')
 save(a.output/'membership.json',membership)
 context_keys=['selection_sha256','model_sha256','columns','profile','checkpoint_sha256','regions_sha256','simulator_release','far_auxiliary','event_definition','environment_profile_sha256','environment_thresholds','auxiliary_gt_minimum']
 frozen={key:copy.deepcopy(parent[key]) for key in context_keys};frozen.update(version='drive-review-calibration-2',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),parent_calibration_sha256=sha(n/'calibration_v1/calibration_frozen.json'),calibration_rows=len(mean),calibration_roads=len(set(roads)),registry_sha256=sha(a.output/'membership.json'),registry_kind='explicit_reassigned_calibration_membership',moments=moments,joint_map=dict(kind='identity',temperature=1.,bias=[0.]*4),global_map=dict(kind='identity',slope=1.,intercept=0.),selected_cv=winner,all_candidates=records,global_reference_sha256=sha(a.output/'global_moments_reference.json'),retired_supported_followup=True,Q_moments_unchanged=True,near_far_scale_floor=1.,confirmation_not_collected=True,scope=__doc__)
 save(a.output/'calibration_frozen.json',frozen)
 policy=read(n/'review_policy_v1/policy_frozen.json');policy.pop('oof_sha256',None);policy.update(version='drive-review-policy-2',created_utc=datetime.now(timezone.utc).isoformat(),calibration_sha256=sha(a.output/'calibration_frozen.json'),calibration_membership_sha256=sha(a.output/'membership.json'),source_sha256=sha(Path(__file__)),scope='Common0.7 review, raw Q moments unchanged; spatial probabilities diagnostic pending fresh confirmation.')
 save(a.output/'policy_frozen.json',policy);save(a.output/'membership.json',dict(ids=data['ids'].tolist(),roads=roads.tolist(),sources=[dict(index_sha256=sha(registry/'index_thinned.json'),packs={key:sha(packs/(key+'.npz')) for key in read(registry/'index_full.json')}) for registry,packs in sources],scope='Original calibration plus retired supported followup; main prospective and condition-shift test excluded.'))
 print('Conservative calibration frozen',len(mean),len(set(roads)),winner['bias_kind'],moments['global_fit'],winner['marginals'],flush=True)
if __name__=='__main__':main()
