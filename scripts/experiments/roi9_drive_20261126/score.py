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
from calibrate import support,spatial_labels,probability_metrics,marginal_metrics,global_probability,group_mean
from review_policy import evaluate,priority

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','feature-study','registry','packs','calibration','policy','rf','seal','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 seal=read(a.seal)
 for name,digest in seal['artifacts'].items():assert sha(a.study/name)==digest,('frozen artifact changed',name)
 registry=read(a.registry/'index_full.json');thin=read(a.registry/'index_thinned.json');assert all(r['cohort']=='test' for v in registry.values() for r in v['rows'])
 runs=[p for p in (a.study/'runs').glob('*/summary.json') if read(p)['role']=='test']
 assert len(runs)==45 and all(read(p.parent/'design.json')['created_utc']>seal['created_utc'] for p in runs)
 data,schema=load(a.packs,list(registry));ids={r['sample_id'] for v in thin.values() for r in v['rows']};data=subset(data,np.isin(data['ids'],list(ids)))
 frozen=read(a.calibration/'calibration_frozen.json');predictor=DriveReviewPredictor(a.feature_study,a.calibration,a.policy/'policy_frozen.json',checkpoint_sha256=frozen['checkpoint_sha256'],regions_sha256=frozen['regions_sha256'])
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
 review={name:evaluate(score,reviewdata) for name,score in scores.items()}
 random=[evaluate(np.zeros(len(roads)),reviewdata,random_seed=23426+j*10) for j in range(100)]
 review['random100']=dict(mean_utility=float(np.mean([r['utility'] for r in random])),p05_p95_utility=np.quantile([r['utility'] for r in random],[.05,.95]).tolist(),mean_counts={key:float(np.mean([r['counts'][key] for r in random])) for key in random[0]['counts']},seeds=[23426+j*10 for j in range(100)])
 adoption=dict(spatial_brier_improves=spatial['selected_calibration']['spatial']['brier']<spatial['raw_composed']['spatial']['brier'],far_coverage90=spatial['selected_calibration']['marginals']['far_recall']['coverage90'],policy_beats_global=review['frozen_policy']['utility']>review['calibrated_global_probability']['utility'],scope='Prospective three short segments on the same map. Report per-road evidence; this is not final deployment, independent-map or perception-retraining adoption.')
 np.savez_compressed(a.output/'predictions.npz',**prediction,ids=data['ids'],roads=roads,truth=data['Y'],labels=labels,weather=data['weather'],hour=data['hour'],counts=data['counts'],near_gt=data['near_gt'],far_gt=data['far_gt'],rf_mean=rf_mean,primary_mean=primary_mean,primary_covariance=primary_cov)
 save(a.output/'results.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),seal_sha256=sha(a.seal),rows=len(roads),roads=len(set(roads)),supported_spatial_rows=int(support(data).sum()),regression=regression,calibration=spatial,review=review,adoption=adoption,predictions_sha256=sha(a.output/'predictions.npz')))
 print('prospective score complete',len(roads),adoption,flush=True)
if __name__=='__main__':main()
