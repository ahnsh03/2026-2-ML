"""Fit road-held-out calibration on new drives, then freeze before test driving."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.special import ndtr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_risk import apply_calibration
from quality_estimation.spatial_composition import independent_far_posterior
from quality_estimation.drive_calibration import (spatial_probabilities,spatial_labels,fit_moments,apply_moments,fit_probability_map,apply_probability_map,fit_monotone_binary,apply_monotone_binary)
sys.path.insert(0,str(Path(__file__).parents[1]/'roi9_gp_20261125'))
from study import load,read,save,sha
from feature_study import old_training,subset

def environment_profile(old,train,schema):
 path=old/'portable_forests/environment.npz'
 with np.load(path,allow_pickle=False) as z:profile={k:z[k].tolist() for k in ['centers','selected','mean','scale','ordered_features']}
 profile.update(source_sha256=sha(path),policy_map=AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(np.asarray(profile['centers'])).tolist())
 _,dist,_=assign_environment(train,schema,profile)
 profile['maximum_reference_distance']=float(np.quantile(dist,.99,method='higher'))
 return profile
def assign_environment(data,schema,profile):
 x=feature_matrix(data,schema,profile['ordered_features'])[:,profile['selected']]
 x=(x-np.asarray(profile['mean']))/np.asarray(profile['scale']);dist=np.linalg.norm(x[:,None]-np.asarray(profile['centers'])[None],axis=2)
 labels=dist.argmin(1);d=dist[np.arange(len(x)),labels];coarse=np.asarray(profile['policy_map'])[labels].copy()
 if 'maximum_reference_distance' in profile:coarse[d>profile['maximum_reference_distance']]=-1
 return labels,d,coarse
def support(data):return (data['near_gt']>=50)&(data['far_gt']>=50)&np.isfinite(data['Y']).all(1)
def global_probability(mean,covariance,threshold=.7):
 return ndtr((np.asarray(threshold)-mean[:,0])/np.sqrt(np.maximum(covariance[:,0,0],1e-12)))
def group_mean(values,roads):return float(np.mean([np.mean(values[roads==g]) for g in np.unique(roads)]))
def probability_metrics(probability,labels,roads):
 valid=labels>=0;p=probability[valid];y=labels[valid];g=roads[valid]
 if not len(y):return dict(rows=0,brier=None,nll=None)
 indicator=np.eye(p.shape[1])[y];brier=((p-indicator)**2).sum(1);nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
 return dict(rows=len(y),roads=len(set(g)),brier=group_mean(brier,g),nll=group_mean(nll,g),class_counts=np.bincount(y,minlength=p.shape[1]).tolist())
def marginal_metrics(mean,covariance,data):
 result={};y=data['Y'];roads=data['site'];sd=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),1e-12))
 for j,name in enumerate(['IoU','near_recall','far_recall']):
  v=np.isfinite(y[:,j])
  if j:v&=data['near_gt' if j==1 else 'far_gt']>=50
  error=abs(y[v,j]-mean[v,j]);covered=error<=1.6448536269514722*sd[v,j]
  result[name]=dict(rows=int(v.sum()),mae=group_mean(abs(np.clip(mean[v,j],0,1)-y[v,j]),roads[v]),coverage90=group_mean(covered,roads[v]),unbounded_width90=group_mean(2*1.6448536269514722*sd[v,j],roads[v]))
 return result
def fit_thresholds(data,env):
 near=data['counts'][:,6:8].sum(1);precision=np.divide(near[:,0],near[:,0]+near[:,1],out=np.full(len(near),np.nan),where=near[:,0]+near[:,1]>0)
 total=data['counts'].sum(1);global_precision=np.divide(total[:,0],total[:,0]+total[:,1],out=np.full(len(total),np.nan),where=total[:,0]+total[:,1]>0)
 anchors=(data['near_gt']>=50)&(data['Y'][:,1]>=.9)&(precision>=.9)&(global_precision>=.9)&np.isfinite(data['Y'][:,0])
 result={}
 for g in np.unique(env):
  ix=(env==g)&anchors;n=int(ix.sum());roads=len(set(data['site'][ix]))
  eligible=g>=0 and n>=30 and roads>=2
  if eligible:
   q=float(np.quantile(data['Y'][ix,0],.1));weight=n/(n+50);threshold=float(np.clip((1-weight)*.7+weight*q,.4,.7))
  else:q=None;weight=0.;threshold=.7
  result[str(int(g))]=dict(threshold=threshold,reference_q10=q,normal_proxy_rows=n,roads=roads,shrinkage=weight,eligible=eligible,scope='Near precision/recall>=.9 and whole-frame precision>=.9 reference; not operational visibility/normality ground truth')
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','old','feature-study','registry','packs','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 selected=read(a.feature_study/'selection_frozen.json');record=selected['chosen'];modelpath=a.feature_study/record['model'];assert sha(modelpath)==record['model_sha256'];model=SpatialQualityGP.load(modelpath)
 assert model.feature_names_.tolist()==record['columns']
 assert not any(json.loads(p.read_text())['role']=='test' for p in (a.study/'runs').glob('*/summary.json'))
 registry=read(a.registry/'index_full.json');thin=read(a.registry/'index_thinned.json');assert all(r['cohort']=='calibration' for pack in registry.values() for r in pack['rows'])
 data,schema=load(a.packs,list(registry));ids={r['sample_id'] for pack in thin.values() for r in pack['rows']};data=subset(data,np.isin(data['ids'],list(ids)));assert len(set(data['site']))==3
 oldtrain,_=old_training(a.old);profile=environment_profile(a.old,oldtrain,schema);fine,distance,env=assign_environment(data,schema,profile)
 mean,cov=model.predict(feature_matrix(data,schema,record['columns']))
 auxiliary=selected.get('far_auxiliary')
 if auxiliary:
  path=a.feature_study/auxiliary['model'];assert sha(path)==auxiliary['model_sha256'];far=SpatialQualityGP.load(path);fm,fc=far.predict(feature_matrix(data,schema,auxiliary['columns']));mean,cov=independent_far_posterior(mean,cov,fm,fc)
 raw_p=spatial_probabilities(mean,cov);labels=spatial_labels(data['Y']);labels[~support(data)]=-1
 roads=data['site'];defined=np.isfinite(data['Y'][:,0]);low=(data['Y'][:,0]<.7).astype(int)
 calibration_truth=data['Y'].copy();calibration_truth[data['near_gt']<50,1]=np.nan;calibration_truth[data['far_gt']<50,2]=np.nan
 oldcal=read(a.old/'frozen_bundle_final/calibration_26r1.json')
 prior_compatible=record['model_sha256']==oldcal['model_sha256'] and auxiliary is None
 raw=dict(marginals=marginal_metrics(mean,cov,data),spatial=probability_metrics(raw_p,labels,roads))
 if prior_compatible:
  pm,pc=apply_calibration(mean,cov,oldcal);raw['old96']=dict(marginals=marginal_metrics(pm,pc,data),spatial=probability_metrics(spatial_probabilities(pm,pc),labels,roads))
 save(a.output/'reference_metrics.json',raw);save(a.output/'environment_profile.json',profile)
 records=[];oof={};splits=list(np.unique(roads))
 for conditional in [False,True]:
  for rho in [0.,.5,1.]:
   moment_name=f'env{int(conditional)}_rho{rho}'
   bases=[]
   for road in splits:
    fit=roads!=road;held=~fit
    calibration=fit_moments(mean[fit],cov[fit],calibration_truth[fit],env[fit],roads[fit],conditional=conditional,correlation_weight=rho)
    tm,tc,_=apply_moments(mean[fit],cov[fit],env[fit],calibration);hm,hc,fallback=apply_moments(mean[held],cov[held],env[held],calibration)
    train_p=spatial_probabilities(tm,tc);held_p=spatial_probabilities(hm,hc)
    train_low=global_probability(tm,tc);held_low=global_probability(hm,hc)
    valid_fit=defined[fit];binary=fit_monotone_binary(train_low[valid_fit],low[fit][valid_fit],roads[fit][valid_fit])
    threshold_fit=fit_thresholds(subset(data,fit),env[fit]);threshold_vector=np.array([threshold_fit.get(str(int(e)),{'threshold':.7})['threshold'] for e in env[held]])
    bases.append(dict(road=str(road),fit=fit,held=held,calibration=calibration,train_p=train_p,held_p=held_p,held_low=held_low,held_binary=apply_monotone_binary(held_low,binary),held_relative=global_probability(hm,hc,threshold_vector),held_threshold=threshold_vector,held_mean=hm,held_cov=hc,held_fallback=fallback,binary=binary,threshold_fit=threshold_fit))
   for mapping in ['identity','temperature','vector']:
    name=moment_name+'_'+mapping;pred=np.zeros((len(mean),4));plow=np.zeros(len(mean));binary=np.zeros(len(mean));relative=np.zeros(len(mean));thresholds=np.full(len(mean),.7);hm=np.zeros_like(mean);hc=np.zeros_like(cov);fallback=np.ones(len(mean),bool);folds=[]
    for b in bases:
     fit=b['fit'];held=b['held'];cal=fit_probability_map(b['train_p'],labels[fit],roads[fit],kind=mapping)
     pred[held]=apply_probability_map(b['held_p'],cal);plow[held]=b['held_low'];binary[held]=b['held_binary'];relative[held]=b['held_relative'];thresholds[held]=b['held_threshold'];hm[held]=b['held_mean'];hc[held]=b['held_cov'];fallback[held]=b['held_fallback']
     folds.append(dict(road=b['road'],moment=b['calibration'],joint_map=cal,binary=b['binary'],environment_threshold_fit=b['threshold_fit']))
    score=probability_metrics(pred,labels,roads);v=defined
    row=dict(name=name,conditional=conditional,correlation_weight=rho,mapping=mapping,spatial=score,global_brier=group_mean((plow[v]-low[v])**2,roads[v]),platt_global_brier=group_mean((binary[v]-low[v])**2,roads[v]),marginals=marginal_metrics(hm,hc,data),conditional_fallback_rows=int(fallback.sum()),folds=folds)
    records.append(row);oof[name]=(pred,plow,binary,hm,hc,relative,thresholds);save(a.output/'progress.json',records)
 # Same calibration roads develop the candidate; only future test is final validation.
 eligible=[r for r in records if r['marginals']['far_recall']['coverage90']>=.8]
 if not eligible:eligible=records
 winner=min(eligible,key=lambda r:(r['spatial']['brier'],r['spatial']['nll'],r['conditional'],r['correlation_weight']))
 cal=fit_moments(mean,cov,calibration_truth,env,roads,conditional=winner['conditional'],correlation_weight=winner['correlation_weight'])
 cm,cc,fallback=apply_moments(mean,cov,env,cal);joint=fit_probability_map(spatial_probabilities(cm,cc),labels,roads,kind=winner['mapping'])
 use_binary=winner['platt_global_brier']<=winner['global_brier']-.001
 binary=fit_monotone_binary(global_probability(cm,cc)[defined],low[defined],roads[defined]) if use_binary else dict(kind='identity',slope=1.,intercept=0.)
 thresholds=fit_thresholds(data,env)
 prediction=oof[winner['name']]
 relative=prediction[5].copy()
 if use_binary:relative[np.isclose(prediction[6],.7)]=prediction[2][np.isclose(prediction[6],.7)]
 np.savez_compressed(a.output/'selected_oof.npz',ids=data['ids'],roads=roads,weather=data['weather'],hour=data['hour'],truth=data['Y'],labels=labels,near_gt=data['near_gt'],far_gt=data['far_gt'],counts=data['counts'],environment=env,fine_environment=fine,environment_distance=distance,spatial=prediction[0],global_probability=prediction[2] if use_binary else prediction[1],environment_relative_probability=relative,environment_threshold=prediction[6],mean=prediction[3],covariance=prediction[4],raw_mean=mean,raw_covariance=cov)
 frozen=dict(version='drive-review-calibration-1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),selection_sha256=sha(a.feature_study/'selection_frozen.json'),model_sha256=record['model_sha256'],columns=record['columns'],profile='roi9-front-v1',checkpoint_sha256=str(model.checkpoint_sha256_),regions_sha256=str(model.regions_sha256_),simulator_release='26.R1',registry_sha256=sha(a.registry/'index_thinned.json'),calibration_rows=len(mean),calibration_roads=len(splits),selected_cv=winner,all_candidates=records,moments=cal,joint_map=joint,global_map=binary,environment_profile_sha256=sha(a.output/'environment_profile.json'),environment_thresholds=thresholds,prior96_compatible=prior_compatible,interval_gate_passed=bool([r for r in records if r['marginals']['far_recall']['coverage90']>=.8]),scope='Developed with calibration-road LOCO. These OOF scores are model/policy development evidence; prospective unobserved test-road drives must validate before adoption. No OOD coverage or operational-failure guarantee.',test_not_collected=True)
 frozen.update(far_auxiliary=auxiliary,auxiliary_gt_minimum=50,source_packs_sha256={key:sha(a.packs/(key+'.npz')) for key in registry})
 save(a.output/'calibration_frozen.json',frozen);print('calibration frozen',winner['name'],winner['spatial'],'globalPlatt',use_binary,flush=True)
if __name__=='__main__':main()
