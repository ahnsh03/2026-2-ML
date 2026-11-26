"""Post-primary GT-support-stratified road validation of unchanged estimators.

Routes are chosen using marking support, never quality predictions/errors.
This is a conditional followup domain, not a replacement for the primary test.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,numpy as np,joblib
from quality_estimation.drive_review import DriveReviewPredictor
from quality_estimation.drive_calibration import spatial_probabilities,apply_moments
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.portable_forest import PortableForest
from feature_study import read,save,sha,load,subset
from calibrate import support,spatial_labels,marginal_metrics,probability_metrics,group_mean,global_probability
from review_policy import evaluate

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','extension','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();n=a.study;e=a.extension;a.output.mkdir(parents=True,exist_ok=False);seal=read(n/'test_freeze_receipt.json');plan=read(e/'plans/frozen.json')
 for name,digest in seal['artifacts'].items():assert sha(n/name)==digest
 assert all(read(path.parent/'design.json')['created_utc']>plan['created_utc'] for path in (e/'runs').glob('*/summary.json'))
 index=read(e/'registry/index_full.json');thin=read(e/'registry/index_thinned.json');data,schema=load(e/'packs',list(index));ids={r['sample_id'] for v in thin.values() for r in v['rows']};data=subset(data,np.isin(data['ids'],list(ids)));roads=data['site'];c=read(n/'calibration_v1/calibration_frozen.json')
 predictor=DriveReviewPredictor(n/'feature_study',n/'calibration_v1',n/'review_policy_v1/policy_frozen.json',checkpoint_sha256=c['checkpoint_sha256'],regions_sha256=c['regions_sha256'],simulator_release='26.R1')
 pred=predictor.predict_features({key:data[key] for key in ['rgb','output','relations','uncertainty','morphology']},schema);pm,pc=predictor.model.predict(feature_matrix(data,schema,predictor.columns))
 rfrec=read(n/'rf_comparator_frozen/frozen.json');x=feature_matrix(data,schema,rfrec['primary_columns']);rf=joblib.load(n/'rf_comparator_frozen/primary_aux.joblib').predict(x) if (n/'rf_comparator_frozen/primary_aux.joblib').exists() else joblib.load(n/'rf_comparator_frozen/primary_auxiliary.joblib').predict(x)
 rf[:,0]=PortableForest.load(n/'rf_comparator_frozen/primary.npz').predict(x)
 if rfrec['far_columns']:rf[:,2]=PortableForest.load(n/'rf_comparator_frozen/far.npz').predict(feature_matrix(data,schema,rfrec['far_columns']))
 regression={}
 for name,mean in [('GP28_primary',pm),('GP28_plus_far31',pred['raw_mean']),('matched_RF',rf)]:
  error=abs(np.clip(mean,0,1)-data['Y']);error[data['near_gt']<50,1]=np.nan;error[data['far_gt']<50,2]=np.nan
  regression[name]=dict(macro_mae=[group_mean(error[:,j],roads) for j in range(3)],byroad={str(g):[group_mean(error[roads==g,j],roads[roads==g]) for j in range(3)] for g in np.unique(roads)})
 labels=spatial_labels(data['Y']);labels[~support(data)]=-1;gm,gc,_=apply_moments(pred['raw_mean'],pred['raw_covariance'],pred['environment'],read(n/'calibration_v1/global_moments_reference.json'));calibration={}
 for name,mean,cov,prob in [('raw_primary',pm,pc,spatial_probabilities(pm,pc)),('raw_composed',pred['raw_mean'],pred['raw_covariance'],spatial_probabilities(pred['raw_mean'],pred['raw_covariance'])),('global_moments',gm,gc,spatial_probabilities(gm,gc)),('selected_calibration',pred['mean'],pred['covariance'],pred['spatial'])]:
  calibration[name]=dict(marginals=marginal_metrics(mean,cov,data),spatial=probability_metrics(prob,labels,roads),byroad={str(g):probability_metrics(prob[roads==g],labels[roads==g],roads[roads==g]) for g in np.unique(roads)})
 reviewdata=dict(truth=data['Y'],counts=data['counts'],near_gt=data['near_gt'],far_gt=data['far_gt'],roads=roads,ids=data['ids']);scores=dict(low_GP_mean=-pred['raw_mean'][:,0],low_RF_mean=-rf[:,0],raw_GP_low_probability=global_probability(pred['raw_mean'],pred['raw_covariance']),calibrated_global=pred['global_probability'],far_only=pred['spatial'][:,2],frozen_policy=pred['priority']);review={name:evaluate(score,reviewdata) for name,score in scores.items()}
 random=[evaluate(np.zeros(len(roads)),reviewdata,random_seed=23426+j*10) for j in range(100)];review['random100']=dict(mean_counts={key:float(np.mean([r['counts'][key] for r in random])) for key in random[0]['counts']},mean_utility=float(np.mean([r['utility'] for r in random])))
 np.savez_compressed(a.output/'predictions.npz',**pred,ids=data['ids'],roads=roads,truth=data['Y'],labels=labels,weather=data['weather'],hour=data['hour'],near_gt=data['near_gt'],far_gt=data['far_gt'],counts=data['counts'],rf_mean=rf,primary_mean=pm)
 save(a.output/'results.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),plan_sha256=sha(e/'plans/frozen.json'),main_seal_sha256=sha(n/'test_freeze_receipt.json'),rows=len(labels),roads=len(set(roads)),spatial_supported_rows=int(support(data).sum()),regression=regression,calibration=calibration,review=review,scope=__doc__))
 print('GT-supported-road followup scored',len(labels),regression,flush=True)
if __name__=='__main__':main()
