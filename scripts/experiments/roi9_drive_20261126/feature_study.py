"""Fit preregistered feature candidates on frozen old membership; choose on new drives."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import joblib,numpy as np,shutil
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'roi9_gp_20261125'))
from study import load,read,save,sha

def subset(data,ix):return {k:v[ix] for k,v in data.items()}
def old_training(old):
 membership=read(old/'frozen_bundle_final/training_membership.json')['rows'];keys=sorted({r['source_key'] for r in membership})
 data,schema=load(old/'packs',keys);lookup={sid:i for i,sid in enumerate(data['ids'])};ix=np.array([lookup[r['sample_id']] for r in membership])
 data=subset(data,ix);assert len(data['ids'])==2480 and np.isfinite(data['Y']).all()
 return data,schema
def folds(data):
 xy=data['xy'];locations,inv=np.unique(xy,axis=0,return_inverse=True)
 clusters=KMeans(n_clusters=6,n_init=10,random_state=23411).fit_predict(locations)[inv]
 result=[]
 for j in range(3):
  held=clusters%3==j;dist=cKDTree(xy[held]).query(xy)[0];fit=(~held)&(dist>=25)
  assert fit.sum()>500 and held.sum()>100;result.append((fit,held))
 return result
def bind(model,columns,recipe):
 model.feature_names_=np.array(columns);model.profile_version_=np.array(recipe['profile']);model.regions_sha256_=np.array(recipe['regions_sha256']);model.checkpoint_sha256_=np.array(recipe['checkpoint_sha256']);model.target_names_=np.array(['whole_frame_lane_iou','near_lane_recall','far_lane_recall'])
 return model
def macro(errors,groups):return np.nanmean([np.nanmean(errors[groups==g],axis=0) for g in np.unique(groups)],axis=0)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--old',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['fit','select'],required=True)
 p.add_argument('--packs',type=Path);p.add_argument('--registry',type=Path)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'models').mkdir(exist_ok=True)
 protocol=read(a.protocol);recipe=read(a.old/'frozen_bundle_final/feature_recipe.json');train,s=old_training(a.old);splits=folds(train)
 training_receipt=dict(protocol_sha256=sha(a.protocol),membership_sha256=sha(a.old/'frozen_bundle_final/training_membership.json'),source_sha256=sha(Path(__file__)),rows=len(train['ids']),folds=[dict(train=int(f.sum()),held=int(h.sum())) for f,h in splits])
 if a.phase=='fit':
  save(a.output/'training_context.json',training_receipt);allrecords=[]
  for name,columns in protocol['feature_views'].items():
   done=a.output/(name+'_fit.json')
   if done.exists():
    records=read(done);assert all(r['protocol_sha256']==sha(a.protocol) for r in records);allrecords.extend(records);continue
   x=feature_matrix(train,s,columns);records=[]
   for kind in ['GP','RF']:
    trials=[]
    for params in protocol[kind.lower()+'_settings']:
     start=time.monotonic();losses=[]
     for fit,held in splits:
      if kind=='GP':
       model=SpatialQualityGP(**params).fit(x[fit],train['Y'][fit]);mean,_=model.predict(x[held]);loss=np.mean(abs(np.clip(mean,0,1)-train['Y'][held]),axis=0)
      else:
       model=RandomForestRegressor(**params,random_state=23411,n_jobs=2).fit(x[fit],train['Y'][fit,0]);loss=np.array([np.mean(abs(model.predict(x[held])-train['Y'][held,0]))])
      losses.append(loss.tolist())
     trials.append(dict(parameters=params,cv=np.mean(losses,axis=0).tolist(),folds=losses,seconds=time.monotonic()-start))
    minimum=min(v['cv'][0] for v in trials);eligible=[v for v in trials if v['cv'][0]<=minimum+.002]
    chosen=min(eligible,key=lambda r:r['cv'][0]+(.25*sum(r['cv'][1:]) if kind=='GP' else 0))
    if kind=='GP':
     model=bind(SpatialQualityGP(**chosen['parameters']).fit(x,train['Y']),columns,recipe);path=a.output/'models'/(name+'_GP.npz');model.save(path);aux=None
    else:
     model=RandomForestRegressor(**chosen['parameters'],random_state=23411,n_jobs=2).fit(x,train['Y'][:,0]);path=a.output/'models'/(name+'_RF.joblib');joblib.dump(model,path,compress=2)
     auxiliary=RandomForestRegressor(**chosen['parameters'],random_state=23411,n_jobs=2).fit(x,train['Y']);aux=a.output/'models'/(name+'_RFaux.joblib');joblib.dump(auxiliary,aux,compress=2)
    record=dict(view=name,kind=kind,columns=columns,dimensions=len(columns),chosen=chosen,trials=trials,model=str(path.relative_to(a.output)),model_sha256=sha(path),auxiliary=str(aux.relative_to(a.output)) if aux else None,auxiliary_sha256=sha(aux) if aux else None,protocol_sha256=sha(a.protocol));records.append(record);print('fitted',name,kind,chosen['cv'],flush=True)
   save(done,records);allrecords.extend(records)
  save(a.output/'fit_complete.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),**training_receipt,records=allrecords));return
 if a.packs is None or a.registry is None:p.error('select requires new-development packs and registry')
 registry=read(a.registry/'index_full.json');thin=read(a.registry/'index_thinned.json');assert all(r['cohort']=='development' for v in registry.values() for r in v['rows'])
 dev,_=load(a.packs,list(registry));ids={r['sample_id'] for v in thin.values() for r in v['rows']};dev=subset(dev,np.isin(dev['ids'],list(ids)))
 assert not set(train['ids'])&set(dev['ids']);assert len(set(dev['site']))==3
 fitted=read(a.output/'fit_complete.json');results=[]
 prior_path=a.output/'models/prior_frozen28_GP.npz';prior_source=a.old/'frozen_bundle_final/quality_spatial_gp.npz'
 if not prior_path.exists():shutil.copy2(prior_source,prior_path)
 assert sha(prior_path)==sha(prior_source)
 base_record=next(r for r in fitted['records'] if r['view']=='baseline28' and r['kind']=='GP')
 prior_params=recipe['tasks']['quality_spatial_gp']['parameters']
 prior_trial=next(t for t in base_record['trials'] if t['parameters']==prior_params)
 prior_record=dict(base_record,view='prior_frozen28',model='models/prior_frozen28_GP.npz',model_sha256=sha(prior_path),chosen=prior_trial,source='Unchanged integration baseline, reused as preregistered comparator')
 for r in fitted['records']+[prior_record]:
  x=feature_matrix(dev,s,r['columns']);path=a.output/r['model'];assert sha(path)==r['model_sha256']
  if r['kind']=='GP':mean,cov=SpatialQualityGP.load(path).predict(x)
  else:
   mean=joblib.load(a.output/r['auxiliary']).predict(x);mean[:,0]=joblib.load(path).predict(x);cov=None
  error=abs(np.clip(mean,0,1)-dev['Y']);byroad={str(g):np.nanmean(error[dev['site']==g],axis=0).tolist() for g in np.unique(dev['site'])}
  out=dict(r,development_macro_mae=macro(error,dev['site']).tolist(),development_by_road=byroad)
  pred=a.output/(r['view']+'_'+r['kind']+'_development.npz');arrays=dict(ids=dev['ids'],mean=mean,Y=dev['Y'],site=dev['site'],weather=dev['weather'],hour=dev['hour']);
  if cov is not None:arrays['covariance']=cov
  np.savez_compressed(pred,**arrays);results.append(out)
 base=next(r for r in results if r['view']=='prior_frozen28' and r['kind']=='GP');b=np.array(base['development_macro_mae']);eligible=[]
 for r in results:
  if r['kind']!='GP':continue
  m=np.array(r['development_macro_mae']);road_improvements=sum(r['development_by_road'][g][0]<=base['development_by_road'][g][0] for g in base['development_by_road'])
  improved=(m[0]<=b[0]-.003 and road_improvements>=2 and np.all(m[1:]<=b[1:]+.02) and r['chosen']['cv'][0]<=base['chosen']['cv'][0]+.005)
  smaller=(r['dimensions']<28 and m[0]<=b[0]+.002 and np.all(m[1:]<=b[1:]+.005) and road_improvements>=2 and r['chosen']['cv'][0]<=base['chosen']['cv'][0]+.005)
  if r['view']=='prior_frozen28' or improved or smaller:eligible.append(r)
 chosen=min(eligible,key=lambda r:(r['development_macro_mae'][0]+.25*sum(r['development_macro_mae'][1:]),r['dimensions']))
 prior_far=read(a.old/'far_followup/fit/frozen.json')['selected']['GP'];far_path=a.old/'far_followup/fit'/prior_far['model_file'];assert sha(far_path)==prior_far['model_sha256']
 fm,fc=SpatialQualityGP.load(far_path).predict(feature_matrix(dev,s,prior_far['ordered_features']))
 with np.load(a.output/(chosen['view']+'_GP_development.npz')) as z:primary_mean=z['mean'];assert np.array_equal(z['ids'],dev['ids'])
 valid=(dev['far_gt']>=50)&np.isfinite(dev['Y'][:,2]);far_byroad={}
 for road in np.unique(dev['site']):
  ix=(dev['site']==road)&valid
  if ix.any():far_byroad[str(road)]=dict(rows=int(ix.sum()),primary=float(np.mean(abs(np.clip(primary_mean[ix,2],0,1)-dev['Y'][ix,2]))),auxiliary=float(np.mean(abs(np.clip(fm[ix,0],0,1)-dev['Y'][ix,2]))))
 old_error=float(np.mean([v['primary'] for v in far_byroad.values()]));aux_error=float(np.mean([v['auxiliary'] for v in far_byroad.values()]))
 adopt_far=bool(valid.sum()>=20 and len(far_byroad)==3 and aux_error<=old_error-.005 and sum(v['auxiliary']<v['primary'] for v in far_byroad.values())>=2)
 far_record=None
 if adopt_far:
  target=a.output/'models/far31_auxiliary_GP.npz'
  if not target.exists():shutil.copy2(far_path,target)
  far_record=dict(model='models/far31_auxiliary_GP.npz',model_sha256=sha(target),columns=prior_far['ordered_features'],target='far_lane_recall',composition='PrimaryQ/near unchanged; far process initially independent. Residual correlation calibration is a separate calibration-road comparison.')
 auxiliary_diagnostic=dict(byroad=far_byroad,primary_macro_mae=old_error,auxiliary_macro_mae=aux_error,adopted=adopt_far,source_model_sha256=prior_far['model_sha256'],rule='>=.005far recallMAE reduction,>=2/3development roads,>=20supported rows; no calibration/test labels')
 save(a.output/'selection_frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selection='Preregistered old geographic CV/new-development macro-road gate. New calibration/test labels excluded.',chosen=chosen,far_auxiliary=far_record,auxiliary_diagnostic=auxiliary_diagnostic,required_feature_union=list(dict.fromkeys(chosen['columns']+(far_record['columns'] if far_record else []))),results=results,development_ids=dev['ids'].tolist(),development_rows=len(dev['ids']),registry_sha256=sha(a.registry/'index_thinned.json'),fit_sha256=sha(a.output/'fit_complete.json'),protocol_sha256=sha(a.protocol),source_sha256=sha(Path(__file__)),test_not_collected=True));print('selected',chosen['view'],chosen['development_macro_mae'],'far auxiliary',adopt_far,flush=True)
if __name__=='__main__':main()
