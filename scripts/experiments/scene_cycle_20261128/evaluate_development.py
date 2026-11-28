"""Select version-bound offline review candidates on development, never test data."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score
from common.bridge import sha256_file
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.review_targets import support_aware_targets,gaussian_utility
from quality_estimation.scene_environment import SceneEnvironment,PHOTO_COLUMNS,photo_summary
from quality_estimation.environment_selection import fit_environment_reference,thresholds,calibrated_priority
from quality_estimation.support_review_selection import select_supported_review
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment_cycle_20261127'))
from study_data import lighting,dump,read,macro


def load_pack(path,identity):
    receipt=read(path.with_suffix('.json'))
    validate_pack_perception(receipt,identity,'model_state_dict')
    if receipt['sha256']!=sha256_file(path):raise ValueError('feature pack changed')
    with np.load(path)as z:return {k:z[k]for k in z.files}


def environment_vectors(pack,hist_path,schema,columns):
    with np.load(hist_path)as z:
        if z['ids'].tolist()!=pack['ids'].tolist()or z['columns'].tolist()!=columns:raise ValueError('RGB environment alignment differs')
        hist=z['X']
    return np.c_[hist,pack['rgb'][:,[schema['rgb'].index(n)for n in PHOTO_COLUMNS]]]


def environment_candidates(fit,environment,vectors,pack):
    """Reuse RGB classifiers once; compare centroids with independently known SDK/GT labels."""
    hist=vectors[:,:len(environment.hist_columns)];photo=photo_summary(vectors[:,-8:]);t=environment.transforms
    prob={name:model.predict_proba(hist)for name,model in environment.models.items()}
    reps=dict(rgb_hist_pca16=(((hist-t['hist_mean'])/t['hist_scale']-t['pca_mean'])@t['pca_components'].T)/np.sqrt(t['pca_variance']),
              prob11=np.sqrt(np.maximum(np.c_[prob['weather'],prob['lighting'],prob['presence']],0)),
              photo12=np.c_[np.sqrt(np.maximum(np.c_[prob['weather'],prob['presence']],0)),np.clip((photo-t['photo_mean'])/t['photo_scale'],-3,3)*.25])
    targets=dict(weather=pack['weather'],lighting=lighting(pack['hour']),
                 presence=(support_aware_targets(pack['counts'])['scene_gt_class']==2).astype(int),
                 appearance=np.digitize(photo[:,0],t['illumination_edges']))
    results=[];assignments={}
    for record in fit['records']:
        with np.load(fit['_directory']/record['model'])as z:centers=z['centers'];active=z['active']
        distance=cdist(reps[record['representation']][:,active],centers,'sqeuclidean');groups=distance.argmin(1)
        scores={name:float(balanced_accuracy_score(y,np.asarray(record['semantic_mapping'][name])[groups]))for name,y in targets.items()}
        result={key:record[key]for key in ['representation','method','k','seed','seed_ari']}
        result.update(scores=scores,semantic_score=float(np.mean([scores[n]for n in ['weather','presence','appearance']])))
        results.append(result);assignments[(record['representation'],record['method'],record['k'],record['seed'])]=groups
    first=[r for r in results if r['seed']==281128 and r['seed_ari']>=.7]
    if not first:raise ValueError('no stable environment candidate')
    selected=max(first,key=lambda r:(r['semantic_score'],-r['k'],r['seed_ari']))
    key=tuple(selected[k]for k in ['representation','method','k','seed'])
    record=next(r for r in fit['records']if tuple(r[k]for k in ['representation','method','k','seed'])==key)
    classifier_scores={n:float(balanced_accuracy_score(targets[n],prob[n].argmax(1)))for n in prob}
    return selected,record,results,classifier_scores


def metrics(selected,pack,target):
    selected=np.asarray(selected,int);mask=np.zeros(len(pack['ids']),bool);mask[selected]=True
    def recall(value,where=None):
        where=np.ones(len(mask),bool)if where is None else where
        denominator=float(np.sum(value[where]));return float(np.sum(value[where&mask])/denominator)if denominator else None
    def road_recall(value):
        values=[recall(value,pack['site']==g)for g in np.unique(pack['site'])]
        return float(np.mean([v for v in values if v is not None]))if any(v is not None for v in values)else None
    utility=target['review_utility'];near=target['near']['error'].astype(float)
    return dict(rows=len(mask),selected=len(selected),utility_recall=recall(utility),utility_road_macro=road_recall(utility),
        near_error_recall=recall(near),near_error_road_macro=road_recall(near),
        near_fn_recall=recall(target['near']['false_negative'].astype(float)),near_fp_recall=recall(target['near']['false_positive'].astype(float)),
        dawn_near_error_recall=recall(near,pack['hour']==7),
        no_supported_error_selected_fraction=float(np.mean(utility[selected]==0))if len(selected)else None,
        selected_zero_near_gt_fraction=float(np.mean(target['near']['gt_pixels'][selected]<50))if len(selected)else None,
        near_error_positives=int(near.sum()),near_error_positive_roads=len(np.unique(pack['site'][near>0])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ['study','models','dev-pack','dev-environment','cal-pack','cal-environment','environment-prototype','output']:
        parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();s=args.study;out=args.output;out.mkdir(exist_ok=False)
    frozen=read(s/'protocol/review_development_grid.json');schema=read(s/'schema.json');fit=read(args.models/'fit_complete.json');identity=fit['perception_identity']
    dev=load_pack(args.dev_pack,identity);cal=load_pack(args.cal_pack,identity)
    if set(dev['ids'])&set(cal['ids'])or set(dev['pixel_sha256'])&set(cal['pixel_sha256']):raise ValueError('Dev/Cal overlap')
    envfit=read(s/'environment_fit/fit_complete.json');envfit['_directory']=s/'environment_fit';env=SceneEnvironment(args.environment_prototype)
    dv=environment_vectors(dev,args.dev_environment,schema,env.hist_columns);cv=environment_vectors(cal,args.cal_environment,schema,env.hist_columns)
    selected_env,record,envresults,classifier_scores=environment_candidates(envfit,env,dv,dev)
    # Reuse verified model/transforms; only candidate centers and their OOD limits vary.
    with np.load(envfit['_directory']/record['model'])as z:env.centers=z['centers'];env.active=z['active']
    env.representation=record['representation'];env.spec['cluster_distance_limits']=record['cluster_distance_limits']
    de=env.predict(dv);ce=env.predict(cv);target=support_aware_targets(dev['counts']);ct=support_aware_targets(cal['counts'])
    records={r['tag']:r for r in fit['records']};predictions={};models={};qmetrics={}
    for tag,r in records.items():
        dx=feature_matrix(dev,schema,r['columns']);cx=feature_matrix(cal,schema,r['columns'])
        if r['kind']=='GP':
            model=SpatialQualityGP.load(args.models/r['models'][0]);models[tag]=model
            mean,cov=model.predict(dx,observation=r['task']=='quality');cm,cc=model.predict(cx,observation=r['task']=='quality')
        else:
            model=[PortableForest.load(args.models/p)for p in r['models']];models[tag]=model
            mean=np.column_stack([m.predict(dx)for m in model]);cm=np.column_stack([m.predict(cx)for m in model]);cov=cc=None
        if r['task']=='quality':
            q=np.clip(mean[:,0],0,1);cq=np.clip(cm[:,0],0,1);scale=np.sqrt(np.maximum(cov[:,0,0],1e-6))if cov is not None else np.ones(len(q))
            cs=np.sqrt(np.maximum(cc[:,0,0],1e-6))if cc is not None else np.ones(len(cq));valid=np.isfinite(ct['whole_frame_iou'])
            residuals=np.sort((ct['whole_frame_iou'][valid]-cq[valid])/cs[valid]);err=np.abs(q-target['whole_frame_iou'])
            qmetrics[tag]=dict(mae=macro(err,dev['site']),pooled_mae=float(np.nanmean(err)))
            predictions[tag]=dict(mean=q,scale=scale,residuals=residuals)
        else:
            if cov is None:utility=mean@np.array([1.,.25,.5]);sd=np.zeros(len(mean))
            else:utility,sd=gaussian_utility(mean,cov)
            predictions[tag]=dict(mean=mean,utility=utility,sd=sd,x=dx)
        print('DEVELOPMENT MODEL',tag,qmetrics.get(tag,{}),flush=True)
    qtag=min(qmetrics,key=lambda t:qmetrics[t]['mae']);qpred=predictions[qtag]
    grouping_map={'cluster':'environment_cluster','weather':'predicted_weather','lighting':'predicted_lighting','appearance':'appearance_bin'}
    groups={};references={}
    for name in frozen['grouping']:
        dg=np.zeros(len(dv),int)if name=='global'else de[grouping_map[name]].copy()
        cg=np.zeros(len(cv),int)if name=='global'else ce[grouping_map[name]].copy()
        if name!='global':dg[de['environment_ood']]=-1;cg[ce['environment_ood']]=-1
        groups[name]=dg
        references[name]=fit_environment_reference(ct['whole_frame_iou'],cg,cal['site'],cal['xy'],eligible=ct['near']['gt_pixels']>=50)
    budget=int(np.ceil(len(dv)*frozen['budget_fraction']));candidates=[];prediction_archive={}
    def assess(tag,policy,score,sd,cov=None,noise=None):
        chosen,_,_=select_supported_review(dev['ids'],score,sd,groups[policy['grouping']],budget,
            beta=policy['beta'],balanced_floor=policy['balanced_floor'],audit_fraction=.1,
            information_fraction=policy['information_fraction'],information_mode=policy['information_mode'],
            covariance=cov,noise_variance=noise,seed=281128)
        result=dict(error_tag=tag,quality_tag=qtag,policy=policy,metrics=metrics(chosen,dev,target))
        candidates.append(result)
    for tag,r in records.items():
        if r['task']!='errors':continue
        p=predictions[tag];cov=noise=None
        if r['kind']=='GP':
            cov=models[tag].posterior_covariance(p['x'],contrast=np.array([1.,.25,.5]));noise=max(float(models[tag].noise_variance*np.sum((np.array([1.,.25,.5])*models[tag].y_scale_)**2)),1e-8)
        for grouping in frozen['grouping']:
            for floor in frozen['balanced_floor']:
                for beta in frozen['beta']if r['kind']=='GP'else[0.]:
                    modes=[(0.,'pointwise')]+([(.1,'pointwise'),(.1,'conditional')]if beta==0 and r['kind']=='GP'else[])
                    for info,mode in modes:
                        policy=dict(mode='supported_utility',grouping=grouping,beta=beta,balanced_floor=floor,audit_fraction=.1,
                                    information_fraction=info,information_mode=mode,utility_weights=[1.,.25,.5])
                        assess(tag,policy,p['utility'],p['sd'],cov,noise)
        prediction_archive[tag+'_error_mean']=p['mean'];prediction_archive[tag+'_utility']=p['utility'];prediction_archive[tag+'_sd']=p['sd']
    # Conventional low-IoU controls use the same selected quality head and exact review budget.
    first_error=next(t for t,r in records.items()if r['task']=='errors'and r['kind']=='RF')
    for grouping in frozen['grouping']:
        threshold=thresholds(references[grouping],groups[grouping],unsupported=de['environment_ood'])
        score=calibrated_priority(qpred['mean'],qpred['residuals'],threshold=threshold,scale=qpred['scale'])
        for floor in frozen['balanced_floor']:
            assess(first_error,dict(mode='quality_probability',grouping=grouping,beta=0.,balanced_floor=floor,audit_fraction=.1,
                information_fraction=0.,information_mode='pointwise',utility_weights=[1.,.25,.5],reference=references[grouping]),score,np.zeros(len(score)))
    def key(r):
        m=r['metrics'];p=r['policy'];return (m['utility_road_macro'],m['near_error_road_macro']or 0.,-p['information_fraction'],-abs(p['beta']),-p['balanced_floor'],p['grouping']=='global')
    plain=[r for r in candidates if r['policy']['beta']==0 and r['policy']['information_fraction']==0]
    base=max(plain,key=key);best=max(candidates,key=key)
    gain=best['metrics']['utility_road_macro']-base['metrics']['utility_road_macro']
    if (best['policy']['beta']!=0 or best['policy']['information_fraction']>0)and gain<frozen['uncertainty_minimum_utility_gain']:best=base
    rng=np.random.default_rng(281128);random=[metrics(rng.choice(len(dv),budget,replace=False),dev,target)for _ in range(100)]
    randommean={k:float(np.mean([r[k]for r in random if r[k]is not None]))if any(r[k]is not None for r in random)else None for k in random[0]}
    document=dict(version='support-aware-development-selection-1',perception_identity=identity,quality_scores=qmetrics,
        selected_quality_tag=qtag,selected=best,plain_best=base,uncertainty_gain_vs_plain=gain,random100_mean=randommean,
        environment_selected=selected_env,environment_candidates=envresults,rgb_classifier_development_scores=classifier_scores,
        environment_ood_fraction=float(de['environment_ood'].mean()),candidates=candidates,
        source_sha256=sha256_file(Path(__file__)),frozen_grid_sha256=sha256_file(s/'protocol/review_development_grid.json'),
        input_sha256={str(p):sha256_file(p)for p in [args.dev_pack,args.dev_environment,args.cal_pack,args.cal_environment,args.models/'fit_complete.json']},
        scope='Development selection only. No independent confirmation accessed. Zero supported-error utility is not operational safety.')
    dump(out/'selection.json',document);dump(out/'policy.json',best['policy'])
    dump(out/'environment_key.json',dict(key=','.join(str(selected_env[n])for n in ['representation','method','k','seed'])))
    np.savez_compressed(out/'development_predictions.npz',ids=dev['ids'],quality_mean=qpred['mean'],quality_scale=qpred['scale'],**prediction_archive,**de)
    (out/'_SUCCESS').write_text(sha256_file(out/'selection.json')+'\n')
    print('SELECTED',qtag,best['error_tag'],best['policy'],best['metrics'],flush=True)


if __name__=='__main__':main()
