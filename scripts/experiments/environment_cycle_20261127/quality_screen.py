"""Q-only GP/RF feature/HPO screen with opened diagnostics explicitly promoted."""
from pathlib import Path
import argparse,itertools,time,json,joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_identity import bind_model_perception
from quality_estimation.perception_identity import baseline_identity,resolve_quality_perception
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.portable_forest import PortableForest
from quality_estimation.bounded_bo import next_expected_improvement
from study_data import read,dump,sha,load_sources,memberships,labels,folds,quality_views,macro


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--packs',type=Path);p.add_argument('--index',type=Path);p.add_argument('--adaptation-run',type=Path);p.add_argument('--output-name',default='f0_quality');a=p.parse_args();S=a.study;out=S/a.output_name;out.mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
 identity=baseline_identity();weights='ema_state_dict'
 if a.packs:
  if not a.index or not a.adaptation_run:raise ValueError('versioned refit needs index and perception run')
  _,identity,exposure=resolve_quality_perception(a.adaptation_run,None);weights='model_state_dict';parts={};rows=[];membership={};index=read(a.index);offset=0;schema=read(a.packs/'schema.json')
  for key in ['quality_fit','development','calibration']:
   path=a.packs/(key+'.npz');receipt=read(path.with_suffix('.json'));validate_pack_perception(receipt,identity,weights)
   if receipt['sha256']!=sha(path) or receipt['index_sha256']!=sha(a.index):raise ValueError('versioned input provenance changed')
   with np.load(path) as z:
    assert z['ids'].tolist()==[r['sample_id'] for r in index[key]['rows']]
    for name in z.files:parts.setdefault(name,[]).append(z[name])
    n=len(z['ids']);membership[key]=np.arange(offset,offset+n);offset+=n
   rows+=index[key]['rows']
  data={k:np.concatenate(v) for k,v in parts.items()};exposed={r['sample_id'] for r in exposure}
  if exposed.intersection(data['ids']):raise ValueError('perception-exposed quality rows')
 else:data,schema,rows=load_sources(a.parent);membership=memberships(S,data['ids'])
 fit=membership['quality_fit'];dev=membership['development'];cal=membership['calibration'];y,_,_=labels(data);requested_fit=len(fit);fit=fit[np.isfinite(y[fit])]
 if len(fit)<1000:raise ValueError('too few defined Q labels after new perception')
 base=read(Path('config/region_feature_recipe_v3.json'))['tasks']['quality_spatial_gp']['ordered_features'];views=quality_views(data,schema,base);splits=folds(data['xy'][fit]);primary=np.array([rows[i]['source_key'].startswith('drive_development_') for i in dev]);assert primary.sum()==361
 parameters={'GP':[dict(length_multiplier=e,noise_variance=n,task_correlation=0.) for e,n in itertools.product([1.4,2.8,5.6],[.02,.05,.1])],
             'RF':[dict(n_estimators=160,max_depth=d,min_samples_leaf=l,max_features=f) for d,l,f in itertools.product([12,None],[1,3,6],[.5,1.])]}
 records=[];predictions={};search={}
 for view,(columns,all_x) in views.items():
  for kind in ['GP','RF']:
   grid=parameters[kind];coords=np.array([[i/max(len(grid)-1,1)] for i in range(len(grid))]);cache={}
   def make(params):return SpatialQualityGP(**params) if kind=='GP' else RandomForestRegressor(**params,random_state=271127,n_jobs=2)
   def objective(i):
    if i in cache:return cache[i]['loss']
    errors=[];begin=time.monotonic()
    for train,val in splits:
     m=make(grid[i]).fit(all_x[fit[train]],y[fit[train]])
     pr=m.predict(all_x[fit[val]])[0][:,0] if kind=='GP' else m.predict(all_x[fit[val]])
     errors.append(macro(abs(np.clip(pr,0,1)-y[fit[val]]),data['site'][fit[val]]))
    cache[i]=dict(parameters=grid[i],fold_mae=errors,loss=float(np.mean(errors)),seconds=time.monotonic()-begin)
    print('CV',view,kind,i,cache[i]['loss'],flush=True);dump(out/f'{view}_{kind}_cv_{i}.json',cache[i]);return cache[i]['loss']
   budget=min(6,len(grid));initial=[0,len(grid)//2,len(grid)-1];history=[];losses=[]
   for _ in range(budget):
    i=next_expected_improvement(coords,history,losses,initial);history.append(i);losses.append(objective(i))
   random=initial+np.random.default_rng(271127).permutation([i for i in range(len(grid)) if i not in initial]).tolist()[:budget-len(initial)];rloss=[objective(int(i)) for i in random]
   best=min(set(history+random),key=lambda i:(cache[i]['loss'],i));params=grid[best];model=make(params).fit(all_x[fit],y[fit]);mean_parts=[];sd_parts=[]
   for start in range(0,len(all_x),256):
    if kind=='GP':
     m,c=model.predict(all_x[start:start+256]);mean_parts.append(m[:,0]);sd_parts.append(np.sqrt(c[:,0,0]))
    else:mean_parts.append(model.predict(all_x[start:start+256]))
   mean=np.clip(np.concatenate(mean_parts),0,1);sd=np.concatenate(sd_parts) if kind=='GP' else np.full(len(all_x),np.nan)
   path=out/'models'/f'{view}_{kind}.npz'
   context=dict(profile=schema['profile'],regions_sha256=schema['regions_sha256'],checkpoint_sha256=identity['checkpoint_sha256'],perception_identity=identity,perception_weights=weights,target='whole_frame_lane_iou')
   if kind=='GP':
    model.feature_names_=np.array(columns);model.target_names_=np.array(['whole_frame_lane_iou']);model.profile_version_=np.array(schema['profile']);model.regions_sha256_=np.array(schema['regions_sha256']);bind_model_perception(model,identity,weights);model.save(path)
   else:
    portable=PortableForest.from_sklearn(model,columns,context);portable.save(path);assert np.max(abs(portable.predict(all_x[dev])-mean[dev]))<1e-10
   prediction=out/f'{view}_{kind}_predictions.npz';np.savez_compressed(prediction,ids=data['ids'],mean=mean,sd=sd,Y=y)
   record=dict(view=view,kind=kind,columns=columns,parameters=params,cv_mae=cache[best]['loss'],dev_primary_mae=macro(abs(mean[dev[primary]]-y[dev[primary]]),data['site'][dev[primary]]),dev_historical_mae=macro(abs(mean[dev[~primary]]-y[dev[~primary]]),data['site'][dev[~primary]]),
               dev_road_mae={str(site):float(np.nanmean(abs(mean[dev][data['site'][dev]==site]-y[dev][data['site'][dev]==site]))) for site in np.unique(data['site'][dev[primary]])},model=str(path.relative_to(S)),model_sha256=sha(path),prediction=str(prediction.relative_to(S)),calibration_defined_rows=int(np.isfinite(y[cal]).sum()))
   records.append(record);search[view+'_'+kind]=dict(budget_each=budget,bo_losses=losses,random_losses=rloss,trials=cache,selected=params)
   dump(out/'records_partial.json',dict(records=records,search=search));print('FITTED',view,kind,record['dev_primary_mae'],flush=True)
 # Feature view selection uses CV first. Require material development support
 # rather than searching an open confirmation cohort for a favourable view.
 selected={}
 for kind in ['GP','RF']:
  base_rec=next(r for r in records if r['kind']==kind and r['view']=='base28');eligible=[r for r in records if r['kind']==kind and (r['view']=='base28' or (r['cv_mae']<=base_rec['cv_mae'] and r['dev_primary_mae']<=base_rec['dev_primary_mae']-.002 and r['dev_historical_mae']<=base_rec['dev_historical_mae']+.005))]
  selected[kind]=min(eligible,key=lambda r:(r['dev_primary_mae'],len(r['columns'])))
 gp,rf=selected['GP'],selected['RF'];rf_better=rf['dev_primary_mae']<gp['dev_primary_mae']-.003 and rf['cv_mae']<gp['cv_mae'] and sum(rf['dev_road_mae'][k]<gp['dev_road_mae'][k] for k in gp['dev_road_mae'])>=2
 winner='RF' if rf_better else 'GP';dump(out/'quality_selection_frozen.json',dict(records=records,selected=selected,policy_predictor=winner,rf_consistently_better=bool(rf_better),fit_rows=len(fit),requested_fit_rows=requested_fit,undefined_fit_excluded=requested_fit-len(fit),development_rows=937,primary_development_rows=361,search=search,perception_identity=identity,perception_weights=weights,source_sha256=sha(__file__),scope='Q-only regression with common defined-label GP/RF membership; old heldout explicitly promoted. No fresh confirmation opened.'))
 (out/'_SUCCESS').write_text(sha(out/'quality_selection_frozen.json')+'\n');print('QUALITY SCREEN COMPLETE',winner,flush=True)

if __name__=='__main__':main()
