"""Fit version-bound Q and support-aware error models before confirmation access."""
from pathlib import Path
import argparse
import itertools
import json
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_identity import bind_model_perception
from quality_estimation.portable_forest import PortableForest
from quality_estimation.review_targets import support_aware_targets,ERROR_TARGET_NAMES
from quality_estimation.bounded_bo import next_expected_improvement
from quality_estimation.botorch_search import next_log_expected_improvement

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment_cycle_20261127'))
from study_data import folds,macro,dump,read,sha


def balanced_brier(y,p):
    y=np.asarray(y);p=np.clip(p,0,1);parts=[]
    for target in range(y.shape[1]):
        errors=[float(np.mean((p[y[:,target]==v,target]-v)**2))for v in [0,1]if (y[:,target]==v).any()]
        parts.append(float(np.mean(errors)))
    return float(np.dot(parts,[.6,.25,.15]))


def make_views(schema,base):
    extras=[]
    for region in ['upper_top_left','upper_top_right','upper_bottom_left','upper_bottom_right','hood']:
        extras += [region+'.lane_fraction',region+'.lane_weighted_probability']
    for region in ['far_left','far_right']:
        extras += [region+'.lane_fraction',region+'.lane_weighted_entropy',region+'.lane_weighted_probability']
    extras += ['near_left.lane_drivable_disagreement','near_right.lane_drivable_disagreement',
               'global.lane_drivable_disagreement','near.internal_gap','far.internal_gap','near.soft_outside']
    return dict(env52=base,review_support=list(dict.fromkeys(base+extras)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--fit-pack',type=Path,required=True)
    parser.add_argument('--schema',type=Path,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--base-features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=10)
    parser.add_argument('--search-seeds',type=int,nargs='+',default=[281128,281129,281130])
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    context=dict(fit_pack_sha256=sha(args.fit_pack),schema_sha256=sha(args.schema),identity_sha256=sha(args.identity),
                 base_features_sha256=sha(args.base_features),budget=args.budget,search_seeds=args.search_seeds,
                 source_sha256=sha(__file__))
    import quality_estimation.botorch_search as bo_source
    import quality_estimation.review_targets as target_source
    context['component_sha256']={Path(module.__file__).name:sha(module.__file__)for module in [bo_source,target_source]}
    receipt=out/'run_context.json'
    if receipt.exists():
        if read(receipt)!=context:raise ValueError('resume data/model/search/source context changed')
    else:dump(receipt,context)
    z=np.load(args.fit_pack);data={k:z[k]for k in z.files};schema=read(args.schema)
    identity_doc=read(args.identity);identity=identity_doc.get('perception_identity',identity_doc)
    base=read(args.base_features)['selected']['GP']['columns'];views=make_views(schema,base)
    target=support_aware_targets(data['counts']);targets=dict(quality=target['whole_frame_iou'][:,None],errors=target['error_targets'])
    split=folds(data['xy']);records=[];all_search={}
    if not 1000<=len(data['ids'])<=2500:raise ValueError('common exact-GP/RF fitting population differs')
    for view,columns in views.items():
        x=feature_matrix(data,schema,columns)
        groups=['rgb'if name in schema['rgb']else'relation'if name in schema['relations']else'prediction'for name in columns]
        for task,y in targets.items():
            for kind in ['GP','RF']:
                tag=f'{view}_{task}_{kind}';completed=out/(tag+'_record.json')
                if completed.exists():
                    records.append(read(completed));all_search[tag]=read(out/(tag+'_search.json'));continue
                if kind=='GP':
                    correlations=[0.]if task=='quality'else[0.,.5]
                    grid=[dict(length_multiplier=l,noise_variance=n,task_correlation=c,kernel_mode=k)
                          for l,n,c,k in itertools.product([1.4,2.8,5.6],[.03,.1],correlations,['isotropic','additive'])]
                    coords=np.array([[np.log(v['length_multiplier']/1.4)/np.log(4),np.log(v['noise_variance']/.03)/np.log(10/3),
                                      v['task_correlation']/.5,float(v['kernel_mode']=='additive')]for v in grid])
                else:
                    grid=[dict(n_estimators=160,max_depth=d,min_samples_leaf=l,max_features=f)
                          for d,l,f in itertools.product([12,None],[1,3,6],[.5,1.])]
                    coords=np.array([[float(v['max_depth']is None),(v['min_samples_leaf']-1)/5,(v['max_features']-.5)/.5]for v in grid])
                coords=np.clip(coords,0,1);cache={};trials_path=out/(tag+'_cv.json')
                if trials_path.exists():cache={int(k):v for k,v in read(trials_path).items()}

                def fit(params,xx,yy):
                    if kind=='GP':return SpatialQualityGP(**params,feature_groups=groups).fit(xx,yy)
                    return [RandomForestRegressor(**params,random_state=281128,n_jobs=2).fit(xx,yy[:,j])for j in range(yy.shape[1])]

                def predict(model,xx):
                    if kind=='GP':return model.predict(xx)[0]
                    return np.column_stack([m.predict(xx)for m in model])

                def objective(index):
                    if index in cache:return cache[index]['loss']
                    began=time.monotonic();losses=[]
                    for train,val in split:
                        train=train[np.isfinite(y[train]).all(1)];val=val[np.isfinite(y[val]).all(1)]
                        model=fit(grid[index],x[train],y[train]);pred=predict(model,x[val])
                        value=macro(abs(np.clip(pred[:,0],0,1)-y[val,0]),data['site'][val])if task=='quality'else balanced_brier(y[val],pred)
                        losses.append(value)
                    cache[index]=dict(parameters=grid[index],fold_losses=losses,loss=float(np.mean(losses)),seconds=time.monotonic()-began)
                    dump(trials_path,cache);print('CV',tag,index,cache[index]['loss'],'sec',round(cache[index]['seconds'],1),flush=True)
                    return cache[index]['loss']

                histories=[];budget=min(args.budget,len(grid))
                if budget<4:raise ValueError('at least4 equal-budget queries required')
                for seed in args.search_seeds:
                    rng=np.random.default_rng(seed);initial=rng.choice(len(grid),4,replace=False).tolist()
                    for method in ['random','fixed_ei','botorch_logei']:
                        history=[];losses=[];proposals=[]
                        for iteration in range(budget):
                            if method=='random':
                                choices=initial+np.random.default_rng(seed).permutation([i for i in range(len(grid))if i not in initial]).tolist()
                                index=int(choices[iteration]);detail=dict(stage='matched_random')
                            elif method=='fixed_ei':
                                index=next_expected_improvement(coords,history,losses,initial);detail=dict(stage='fixed_Matern52_EI')
                            else:index,detail=next_log_expected_improvement(coords,history,losses,initial,seed=seed)
                            history.append(index);losses.append(objective(index));proposals.append(detail)
                        histories.append(dict(method=method,seed=seed,initial=initial,indices=history,losses=losses,
                                              incumbent=np.minimum.accumulate(losses).tolist(),proposals=proposals))
                        dump(out/(tag+'_histories_partial.json'),histories)
                used={i for h in histories for i in h['indices']};best=min(used,key=lambda i:(cache[i]['loss'],i));defined=np.isfinite(y).all(1)
                model=fit(grid[best],x[defined],y[defined]);prefix=out/'models'/tag;context=dict(profile=schema['profile'],regions_sha256=schema['regions_sha256'],
                    checkpoint_sha256=identity['checkpoint_sha256'],perception_identity=identity,perception_weights='model_state_dict',target=task)
                names=['whole_frame_lane_iou']if task=='quality'else list(ERROR_TARGET_NAMES);files=[]
                if kind=='GP':
                    model.feature_names_=np.array(columns);model.target_names_=np.array(names);model.profile_version_=np.array(schema['profile']);model.regions_sha256_=np.array(schema['regions_sha256']);bind_model_perception(model,identity,'model_state_dict');path=prefix.with_suffix('.npz');model.save(path);files.append(path)
                else:
                    for j,m in enumerate(model):
                        path=prefix.with_name(prefix.name+'_'+str(j)+'.npz');portable=PortableForest.from_sklearn(m,columns,dict(context,target=names[j]));portable.save(path)
                        if np.max(abs(portable.predict(x[:32])-m.predict(x[:32])))>1e-10:raise ValueError('portable RF differs')
                        files.append(path)
                rec=dict(tag=tag,view=view,task=task,kind=kind,columns=columns,targets=names,parameters=grid[best],cv_loss=cache[best]['loss'],
                         fitted_rows=int(defined.sum()),perception_identity=identity,models=[str(p.relative_to(out))for p in files],model_sha256={str(p.relative_to(out)):sha(p)for p in files},
                         fit_pack_sha256=sha(args.fit_pack),source_sha256=sha(__file__))
                search=dict(query_budget_per_method_per_seed=budget,search_seeds=args.search_seeds,histories=histories,selected_index=best,
                            selected_from='All evaluated BO/random controls; original frozen policy is never overwritten.',unique_cv_trials=len(cache))
                dump(completed,rec);dump(out/(tag+'_search.json'),search);records.append(rec);all_search[tag]=search
                dump(out/'records_partial.json',dict(records=records,search=all_search));print('MODEL COMPLETE',tag,rec['cv_loss'],flush=True)
    dump(out/'fit_complete.json',dict(records=records,search=all_search,perception_identity=identity,fit_pack_sha256=sha(args.fit_pack),
         scope='Fit/CV only. New Dev/Cal selection and independent confirmation remain separate.',source_sha256=sha(__file__)))
    (out/'_SUCCESS').write_text(sha(out/'fit_complete.json')+'\n')


if __name__=='__main__':
    main()
