"""Open prospective moving-test labels once; never fit or select in this script."""
import argparse,json,joblib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from quality_estimation.drive_review import DriveReviewPredictor
from quality_estimation.drive_calibration import apply_moments,spatial_probabilities
from quality_estimation.portable_forest import PortableForest
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from feature_study import load,read,save,sha,subset,macro
from calibrate import support,spatial_labels,probability_metrics,marginal_metrics,global_probability,group_mean,assign_environment
from review_policy import evaluate,priority

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','feature-study','registry','packs','calibration','policy','rf','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--seal',type=Path);p.add_argument('--old',type=Path,help='Optional provenance location; no fitting or old labels are read')
 a=p.parse_args();a.seal=a.seal or a.study/'test_freeze_receipt.json';a.output.mkdir(parents=True,exist_ok=False)
 seal=read(a.seal)
 for name,digest in seal['artifacts'].items():assert sha(a.study/name)==digest,('frozen artifact changed',name)
 registry=read(a.registry/'index_full.json');thin=read(a.registry/'index_thinned.json');assert all(r['cohort']=='test' for v in registry.values() for r in v['rows'])
 runs=[p for p in (a.study/'runs').glob('*/summary.json') if read(p)['role']=='test']
 assert len(runs)==45 and all(read(p.parent/'design.json')['created_utc']>seal['created_utc'] for p in runs)
 data,schema=load(a.packs,list(registry));ids={r['sample_id'] for v in thin.values() for r in v['rows']};data=subset(data,np.isin(data['ids'],list(ids)))
 frozen=read(a.calibration/'calibration_frozen.json');predictor=DriveReviewPredictor(a.feature_study,a.calibration,a.policy/'policy_frozen.json',checkpoint_sha256=frozen['checkpoint_sha256'],regions_sha256=frozen['regions_sha256'],simulator_release='26.R1')
 assert schema['regions_sha256']==frozen['regions_sha256'] and schema['profile']==frozen['profile']
 assert all(read(a.packs/(key+'.json'))['checkpoint_sha256']==frozen['checkpoint_sha256'] for key in registry)
 feature_only={k:data[k] for k in ['rgb','output','relations','uncertainty','morphology']}
 prediction=predictor.predict_features(feature_only,schema);assert len(prediction['mean'])==len(data['ids'])
 raw_mean=prediction['raw_mean'];raw_cov=prediction['raw_covariance'];roads=data['site'];labels=spatial_labels(data['Y']);labels[~support(data)]=-1
 primary_mean,primary_cov=predictor.model.predict(feature_matrix(data,schema,predictor.columns))
 rfrec=read(a.rf/'frozen.json');rf_aux=joblib.load(a.rf/'primary_auxiliary.joblib').predict(feature_matrix(data,schema,rfrec['primary_columns']))
 rf_mean=rf_aux.copy();rf_mean[:,0]=PortableForest.load(a.rf/'primary.npz').predict(feature_matrix(data,schema,rfrec['primary_columns']))
 if rfrec['far_columns']:rf_mean[:,2]=PortableForest.load(a.rf/'far.npz').predict(feature_matrix(data,schema,rfrec['far_columns']))
 controls={'primary_GP28':primary_mean,'composed_GP':raw_mean,'matched_RF':rf_mean,'RF28_auxiliary':rf_aux}
 secondary={};extra_arrays={}
 optimized=a.study/'rf_optimized_control_frozen'
 if optimized.exists():
  record=read(optimized/'frozen.json');assert record['created_utc']<seal['created_utc'] and sha(optimized/'quality.npz')==record['quality_sha256']
  q=PortableForest.load(optimized/'quality.npz').predict(feature_matrix(data,schema,record['chosen']['columns']))
  secondary['development_selected_RF']=dict(view=record['chosen']['view'],dimensions=len(record['chosen']['columns']),macro_iou_mae=group_mean(abs(q-data['Y'][:,0]),roads),byroad={str(g):float(np.mean(abs(q[roads==g]-data['Y'][roads==g,0]))) for g in np.unique(roads)})
  extra_arrays['optimized_rf_q']=q
 context=a.study/'context_control_frozen'
 if context.exists():
  record=read(context/'frozen.json');assert record['created_utc']<seal['created_utc'] and sha(context/'rgb54_GP.npz')==record['model_sha256']
  cx=feature_matrix(data,schema,record['ordered_features']);cm,_=SpatialQualityGP.load(context/'rgb54_GP.npz').predict(cx)
  fine,_,_=assign_environment(data,schema,read(context/'environment_profile.json'))
  cluster_q=np.array([record['cluster_medians'].get(str(int(k)),{'median':record['fallback_median']})['median'] for k in fine])
  metadata_q=np.array([record['metadata_medians'].get(f'{w}:{h}',{'median':record['fallback_median']})['median'] for w,h in zip(data['weather'],data['hour'])])
  secondary['context_controls']={name:dict(macro_iou_mae=group_mean(abs(np.clip(q,0,1)-data['Y'][:,0]),roads)) for name,q in [('RGB54_GP',cm[:,0]),('e_means_cluster_old_median',cluster_q),('weather_time_old_median_ORACLE_METADATA',metadata_q)]}
 regression={}
 for name,mean in controls.items():
  errors=abs(np.clip(mean,0,1)-data['Y']);errors[data['near_gt']<50,1]=np.nan;errors[data['far_gt']<50,2]=np.nan
  regression[name]=dict(macro_mae=macro(errors,roads).tolist(),byroad={str(g):np.nanmean(errors[roads==g],axis=0).tolist() for g in np.unique(roads)})
 gm,gc,_=apply_moments(raw_mean,raw_cov,prediction['environment'],read(a.calibration/'global_moments_reference.json'))
 spatial={};lowtruth=(data['Y'][:,0]<.7).astype(int)
 for name,m,c,prob,low in [
  ('raw_primary',primary_mean,primary_cov,spatial_probabilities(primary_mean,primary_cov),global_probability(primary_mean,primary_cov)),
  ('raw_composed',raw_mean,raw_cov,spatial_probabilities(raw_mean,raw_cov),global_probability(raw_mean,raw_cov)),
  ('global_moments',gm,gc,spatial_probabilities(gm,gc),global_probability(gm,gc)),
  ('selected_calibration',prediction['mean'],prediction['covariance'],prediction['spatial'],prediction['global_probability'])]:
  spatial[name]=dict(marginals=marginal_metrics(m,c,data),spatial=probability_metrics(prob,labels,roads),global_brier=group_mean((low-lowtruth)**2,roads),byroad={str(g):probability_metrics(prob[roads==g],labels[roads==g],roads[roads==g]) for g in np.unique(roads)})
 reviewdata=dict(truth=data['Y'],counts=data['counts'],near_gt=data['near_gt'],far_gt=data['far_gt'],roads=roads,ids=data['ids'])
 scores=dict(low_GP_mean=-raw_mean[:,0],low_RF_mean=-rf_mean[:,0],raw_GP_low_probability=global_probability(raw_mean,raw_cov),calibrated_global_probability=prediction['global_probability'],environment_relative=prediction['environment_relative_probability'],spatial_far_only=prediction['spatial'][:,2],frozen_policy=prediction['priority'])
 if 'optimized_rf_q' in extra_arrays:scores['development_selected_RF_mean']=-extra_arrays['optimized_rf_q']
 review={name:evaluate(score,reviewdata) for name,score in scores.items()}
 random=[evaluate(np.zeros(len(roads)),reviewdata,random_seed=23426+j*10) for j in range(100)]
 review['random100']=dict(mean_utility=float(np.mean([r['utility'] for r in random])),p05_p95_utility=np.quantile([r['utility'] for r in random],[.05,.95]).tolist(),mean_counts={key:float(np.mean([r['counts'][key] for r in random])) for key in random[0]['counts']},seeds=[23426+j*10 for j in range(100)])
 adoption=dict(spatial_brier_improves=spatial['selected_calibration']['spatial']['brier']<spatial['raw_composed']['spatial']['brier'],far_coverage90=spatial['selected_calibration']['marginals']['far_recall']['coverage90'],policy_beats_global=review['frozen_policy']['utility']>review['calibrated_global_probability']['utility'],scope='Prospective three short segments on the same map. Report per-road evidence; this is not final deployment, independent-map or perception-retraining adoption.')
 np.savez_compressed(a.output/'predictions.npz',**prediction,**extra_arrays,ids=data['ids'],roads=roads,truth=data['Y'],labels=labels,weather=data['weather'],hour=data['hour'],counts=data['counts'],near_gt=data['near_gt'],far_gt=data['far_gt'],rf_mean=rf_mean,primary_mean=primary_mean,primary_covariance=primary_cov)
 condition_metrics={}
 for field in ['weather','hour']:
  condition_metrics[field]={}
  for value in np.unique(data[field]):
   ix=data[field]==value;part=subset(data,ix)
   condition_metrics[field][str(value)]=dict(rows=int(ix.sum()),roads=len(set(roads[ix])),class_counts=np.bincount(labels[ix][labels[ix]>=0],minlength=4).tolist(),raw=marginal_metrics(raw_mean[ix],raw_cov[ix],part),calibrated=marginal_metrics(prediction['mean'][ix],prediction['covariance'][ix],part),spatial=probability_metrics(prediction['spatial'][ix],labels[ix],roads[ix]))
 whole_gt=data['counts'][:,:,[0,2]].sum((1,2));whole_pred=data['counts'][:,:,[0,1]].sum((1,2));strata={}
 for minimum in [50,200,500]:
  ix=whole_gt>=minimum
  strata[str(minimum)]=dict(rows=int(ix.sum()),roads=len(set(roads[ix])),GP_iou_mae=group_mean(abs(np.clip(raw_mean[ix,0],0,1)-data['Y'][ix,0]),roads[ix]) if ix.any() else None,RF_iou_mae=group_mean(abs(rf_mean[ix,0]-data['Y'][ix,0]),roads[ix]) if ix.any() else None,near_threshold_rows=int((ix&(abs(data['Y'][:,0]-.7)<=.1)).sum()))
 support_diagnostics=dict(whole_gt_minimum_sensitivity=strata,whole_gt_below50=int((whole_gt<50).sum()),byroad={str(g):dict(rows=int((roads==g).sum()),median_whole_gt=float(np.median(whole_gt[roads==g])),median_predicted_lane_pixels=float(np.median(whole_pred[roads==g])),spatial_supported=int((support(data)&(roads==g)).sum())) for g in np.unique(roads)},scope='Primary whole-IoU benchmark is unchanged. Tiny/no-marking GT can make IoU unstable and does not define operational failure. These are descriptive support strata, not a new model-selected target.')
 save(a.output/'results.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),seal_sha256=sha(a.seal),rows=len(roads),roads=len(set(roads)),supported_spatial_rows=int(support(data).sum()),regression=regression,secondary_controls=secondary,calibration=spatial,condition_metrics=condition_metrics,support_diagnostics=support_diagnostics,review=review,adoption=adoption,predictions_sha256=sha(a.output/'predictions.npz')))
 print('prospective score complete',len(roads),adoption,flush=True)
if __name__=='__main__':main()
