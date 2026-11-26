"""Refit and bind quality GPs/RFs after perception changes; never mix f0/f1 labels."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,itertools,json,sys,time
import joblib,numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import cKDTree
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_identity import bind_model_perception,validate_pack_perception
from quality_estimation.perception_identity import resolve_quality_perception
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.diagnostic_targets import auxiliary_targets,FOUR_TARGET_NAMES
from quality_estimation.versioned_calibration import select_road_calibration,apply_road_calibration
from quality_estimation.bounded_bo import next_expected_improvement
from quality_estimation.region_environment import export_environment
from prepare import dump
sys.path.insert(0,str(Path(__file__).parents[1]/'roi9_gp_20261125'))
from study import load

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def macro(error,groups):
    error=np.asarray(error);return np.nanmean([np.nanmean(error[groups==g],axis=0) for g in np.unique(groups)],axis=0)
def folds(data):
    xy=data['xy'];unique,inv=np.unique(xy,axis=0,return_inverse=True);labels=KMeans(n_clusters=6,n_init=10,random_state=261126).fit_predict(unique)[inv];out=[]
    for j in range(3):
        held=labels%3==j;fit=(~held)&(cKDTree(xy[held]).query(xy)[0]>=25)
        assert fit.sum()>500 and held.sum()>100;out.append((fit,held))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','packs','index','parent-recipe','output']:p.add_argument('--'+name,type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--adaptation-run',type=Path);source.add_argument('--checkpoint',type=Path)
    a=p.parse_args();S=a.study;out=a.output
    if out.exists():raise ValueError('refit output exists')
    out.mkdir(parents=True);(out/'models').mkdir()
    _,identity,exposure=resolve_quality_perception(a.adaptation_run,a.checkpoint);weights='ema_state_dict' if identity['kind']=='baseline' else 'model_state_dict'
    data={};schema=None
    for key in ['quality_fit','development','calibration']:
        receipt=json.loads((a.packs/(key+'.json')).read_text());validate_pack_perception(receipt,identity,weights)
        if receipt['index_sha256']!=sha(a.index):raise ValueError('quality pack source membership changed')
        data[key],schema=load(a.packs,[key])
    exposed_ids={r['sample_id'] for r in exposure};exposed_pixels={r['pixel_sha256'] for r in exposure}
    for key,pack in data.items():
        if exposed_ids.intersection(pack['ids']) or exposed_pixels.intersection(pack['pixel_sha256']):raise ValueError('perception-exposed quality '+key)
    parent=json.loads(a.parent_recipe.read_text());base=parent['tasks']['quality_spatial_gp']['ordered_features'];assert len(base)==28
    views={'base28':base,'fp34':base+['near_left.lane_drivable_disagreement','near_right.lane_drivable_disagreement','global.lane_drivable_disagreement','near.soft_outside','near_left.internal_gap','near_right.internal_gap']}
    targets={key:auxiliary_targets(pack['counts'],pack['y']) for key,pack in data.items()};train=data['quality_fit'];y=targets['quality_fit'];assert len(y)==2480 and np.isfinite(y).all()
    splits=folds(train);x=feature_matrix(train,schema,base)
    parameters=[dict(length_multiplier=ell,noise_variance=noise,task_correlation=rho) for ell,noise,rho in itertools.product([1.4,2.8,5.6],[.02,.05,.1],[0.,.4,.8])]
    coordinates=np.array([[i/2,j/2,k/2] for i,j,k in itertools.product(range(3),repeat=3)]);trials={}
    def objective(i):
        if i in trials:return trials[i]['objective']
        losses=[];t=time.monotonic()
        for fit,held in splits:
            model=SpatialQualityGP(**parameters[i]).fit(x[fit],y[fit,:3]);mean,_=model.predict(x[held]);losses.append(macro(abs(np.clip(mean,0,1)-y[held,:3]),train['site'][held]).tolist())
        m=np.mean(losses,axis=0);value=float(m[0]+.25*(m[1]+m[2]));trials[i]=dict(parameters=parameters[i],fold_mae=losses,mean_mae=m.tolist(),objective=value,seconds=time.monotonic()-t)
        dump(out/f'cv_trial_{i:02d}.json',trials[i]);print('GP CV',i,value,flush=True);return value
    initial=[0,13,20,26];history=[];losses=[]
    for _ in range(8):
        i=next_expected_improvement(coordinates,history,losses,initial);history.append(i);losses.append(objective(i))
    random=initial+np.random.default_rng(261126).permutation([i for i in range(len(parameters)) if i not in initial]).tolist()[:4];random_losses=[objective(int(i)) for i in random]
    # Compare search methods honestly; model candidates are chosen among actually evaluated CV settings.
    best=min(set(history+random),key=lambda i:(trials[i]['objective'],i));params=parameters[best]
    dump(out/'bo_hyperparameters.json',dict(bo_indices=history,bo_losses=losses,random_indices=random,random_losses=random_losses,shared_initial=initial,budget_each=8,
         selected_index=int(best),selected_parameters=params,trials={str(i):r for i,r in trials.items()},scope='Quality-fit geographic CV only; same training rows and25m embargo. BO surrogate is separate from image-quality GP.'))
    dev=data['development'];primary=np.array([str(s).startswith('drive26_') for s in dev['site']]);assert primary.sum()==361
    records=[];models={};predictions={};rf_params=parent['tasks']['quality_rf']['parameters']
    for view,columns in views.items():
        tx=feature_matrix(train,schema,columns);dx=feature_matrix(dev,schema,columns)
        for kind in ['GP3','GP4','RFQ','RF4']:
            t=time.monotonic();n=3 if kind=='GP3' else 4
            if kind.startswith('GP'):
                model=SpatialQualityGP(**params).fit(tx,y[:,:n]);model.feature_names_=np.array(columns);model.target_names_=np.array(FOUR_TARGET_NAMES[:n]);model.profile_version_=np.array(schema['profile']);model.regions_sha256_=np.array(schema['regions_sha256']);bind_model_perception(model,identity,weights)
                path=out/'models'/f'{view}_{kind}.npz';model.save(path);mean,cov=model.predict(dx)
            else:
                model=RandomForestRegressor(**rf_params,random_state=261126,n_jobs=2).fit(tx,y[:,0] if kind=='RFQ' else y)
                path=out/'models'/f'{view}_{kind}.joblib';joblib.dump(model,path,compress=2);mean=model.predict(dx);mean=mean[:,None] if mean.ndim==1 else mean;cov=None;n=mean.shape[1]
            error=abs(np.clip(mean,0,1)-targets['development'][:,:n]);record=dict(view=view,kind=kind,columns=columns,targets=list(FOUR_TARGET_NAMES[:n]),
                development_macro_mae=macro(error[primary],dev['site'][primary]).tolist(),historical_development_mae=macro(error[~primary],dev['site'][~primary]).tolist(),
                model=str(path.relative_to(out)),model_sha256=sha(path),seconds=time.monotonic()-t)
            records.append(record);models[(view,kind)]=model;predictions[(view,kind)]=(mean,cov)
            arrays=dict(ids=dev['ids'],mean=mean,Y=targets['development'][:,:n],site=dev['site'],source='development')
            if cov is not None:arrays['covariance']=cov
            np.savez_compressed(out/f'{view}_{kind}_development.npz',**arrays);print('REFIT',view,kind,record['development_macro_mae'],flush=True)
    def rec(view,kind):return next(r for r in records if r['view']==view and r['kind']==kind)
    base_rec=rec('base28','GP3');extended=rec('fp34','GP3');b=np.array(base_rec['development_macro_mae']);e=np.array(extended['development_macro_mae'])
    use_extra=bool(e[0]<=b[0]-.003 and np.all(e[1:]<=b[1:]+.005));view='fp34' if use_extra else 'base28'
    gp3=rec(view,'GP3');gp4=rec(view,'GP4');rf4=rec(view,'RF4')
    use_four=bool(gp4['development_macro_mae'][0]<=gp3['development_macro_mae'][0]+.002 and gp4['development_macro_mae'][3]<=rf4['development_macro_mae'][3]-.005)
    selected=gp4 if use_four else gp3;chosen=models[(view,selected['kind'])];columns=selected['columns'];n=len(selected['targets']);rfq=rec(view,'RFQ')
    acceptable=bool(selected['development_macro_mae'][0]<=rfq['development_macro_mae'][0]+.005)
    cal=data['calibration'];cx=feature_matrix(cal,schema,columns);cm,cc=chosen.predict(cx);truth=targets['calibration'][:,:n].copy();counts=cal['counts'];ng=(counts[:,6:8,0]+counts[:,6:8,2]).sum(1);fg=(counts[:,4:6,0]+counts[:,4:6,2]).sum(1)
    truth[ng+fg<100,0]=np.nan;truth[ng<50,1]=np.nan;truth[fg<50,2]=np.nan
    if n==4:truth[ng<50,3]=np.nan
    calibration=select_road_calibration(cm,cc,truth,cal['site']);calibration.update(model_sha256=selected['model_sha256'],perception_identity=identity,perception_weights=weights,target_names=selected['targets'],calibration_index_sha256=sha(a.index))
    dump(out/'calibration.json',calibration);np.savez_compressed(out/'calibration_predictions.npz',ids=cal['ids'],mean=cm,covariance=cc,Y=truth,site=cal['site'])
    # RGB environment centers remain independent of perception version.
    f0policy=json.loads((S/'protocol/policy_frozen.json').read_text());cluster=f0policy['cluster'];archive=np.load(S/cluster['path']);master=np.load(S/'source_packs_v2.npz');masterpack={k:master[k] for k in master.files};master_schema=json.loads((S/'schema.json').read_text());pool=masterpack['role']=='candidate_pool'
    environment=export_environment(archive,cluster['columns'],feature_matrix(masterpack,master_schema,cluster['columns'])[pool],source_sha256=cluster['sha256']);dump(out/'environment.json',environment)
    context=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='development_selected_not_heldout_adopted',perception_identity=identity,perception_weights=weights,
        profile=schema['profile'],regions_sha256=schema['regions_sha256'],checkpoint_sha256=identity['checkpoint_sha256'],selected=selected,
        quality_fit_rows=len(train['ids']),development_rows=len(dev['ids']),primary_development_rows=int(primary.sum()),calibration_rows=len(cal['ids']),
        extra_features_adopted=use_extra,near_f1_joint_adopted=use_four,gp_development_acceptable_to_rf=acceptable,rf_comparator=rfq,
        records=records,index_sha256=sha(a.index),source_sha256=sha(Path(__file__)),heldout_accessed=False)
    dump(out/'quality_selection_frozen.json',context);(out/'_SUCCESS').write_text(sha(out/'quality_selection_frozen.json')+'\n')
    print('QUALITY SELECTED',view,selected['kind'],'GP acceptable',acceptable,flush=True)
if __name__=='__main__':main()
