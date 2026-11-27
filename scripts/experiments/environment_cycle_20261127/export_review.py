"""Export portable Q/environment review with version-specific residual calibration."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from scipy.spatial.distance import cdist
from quality_estimation.environment_selection import residual_distribution
from quality_estimation.portable_forest import PortableForest
from quality_estimation.environment_review import EnvironmentReview
from study_data import read,dump,sha,load_sources,memberships,labels,lighting

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--quality-dir',default='f0_quality');p.add_argument('--output',type=Path,required=True);p.add_argument('--interval-study',type=Path);a=p.parse_args();S=a.study;out=a.output
 if out.exists():raise ValueError('bundle output exists')
 quality=read(S/a.quality_dir/'quality_selection_frozen.json');kind=quality['policy_predictor'];selected=quality['selected'][kind];prediction=np.load(S/selected['prediction']);ids=prediction['ids'];y=prediction['Y'];mean=prediction['mean'];scale=np.maximum(prediction['sd'],1e-3) if kind=='GP' else np.ones(len(ids));lookup={s:i for i,s in enumerate(ids)};index=read(S/'protocol/quality_index_frozen.json');cal=np.array([lookup[r['sample_id']] for r in index['calibration']['rows']]);cal=cal[np.isfinite(y[cal])];residuals=residual_distribution(y[cal],mean[cal],scale[cal]);identity=quality['perception_identity'];weights=quality.get('perception_weights','ema_state_dict' if identity['kind']=='baseline' else 'model_state_dict')
 env=read(S/'environment/environment_selection_frozen.json');classifier=env['selected_classifier'];cluster=np.load(S/env['selected_cluster_artifact']);assert env['selected_cluster']['representation']=='supervised_prob9';out.mkdir(parents=True)
 shutil.copy2(S/selected['model'],out/'quality_model.npz')
 for target in ['weather','lighting']:shutil.copy2(S/'environment'/f"{classifier['view']}_{target}.npz",out/f'{target}.npz')
 np.savez_compressed(out/'centers.npz',centers=cluster['centers'],active=cluster['active']);hz=[np.load(S/n) for n in ['environment_features.npz','environment_promoted_features.npz']];hids=np.concatenate([z['ids'] for z in hz]);hx=np.concatenate([z['X'] for z in hz]);hlookup={s:i for i,s in enumerate(hids)};hcols=hz[0]['columns'].tolist();data,schema,rows=load_sources(a.parent);pool=np.flatnonzero(data['role']=='candidate_pool');x=hx[[hlookup[s] for s in data['ids'][pool]]][:,[hcols.index(n) for n in classifier['columns']]]
 wp=PortableForest.load(out/'weather.npz').predict_proba(x);lp=PortableForest.load(out/'lighting.npz').predict_proba(x);embedding=np.sqrt(np.maximum(np.c_[wp,lp],0))[:,cluster['active']];dist=cdist(embedding,cluster['centers'],'sqeuclidean');assigned=dist.argmin(1);distance=np.sqrt(dist.min(1));limits={str(int(g)):float(max(np.quantile(distance[assigned==g],.99),.05)) for g in np.unique(assigned)}
 mapping={};trainlabels=cluster['train_assigned'];w=data['weather'][pool].astype(int);light=lighting(data['hour'][pool])
 for g in range(len(cluster['centers'])):
  mapping[str(g)]={}
  for name,t in [('weather',w),('lighting',light)]:
   values=np.unique(t);mapping[str(g)][name]=int(max(values,key=lambda v:((trainlabels==g)&(t==v)).sum()/max(int((t==v).sum()),1)))
 modelsha=sha(out/'quality_model.npz');calibration=dict(perception_identity=identity,model_sha256=modelsha,signed_residuals=residuals.tolist(),scale='GP posterior predictive SD' if kind=='GP' else 'unit scale; absolute signed residual CDF',calibration_rows=len(cal),calibration_ids_sha256=__import__('hashlib').sha256('\n'.join(ids[cal]).encode()).hexdigest(),interval='Central90% signed residual empirical interval; coverage evaluated on independent roads, not assumed guaranteed conditionally.')
 interval_path=a.interval_study if a.interval_study is not None else S/'g1_quality/interval_calibration_study.json' if a.quality_dir=='g1_quality' else None
 if interval_path is not None:
  interval=read(interval_path)
  if interval['quality_selection_sha256']!=sha(S/a.quality_dir/'quality_selection_frozen.json') or interval['models'][kind]['model_sha256']!=modelsha:raise ValueError('interval calibration belongs to another quality model')
  interval=interval['models'][kind];calibration.update(interval_method=interval['selected_method'],interval_residual_bounds=interval['interval_residual_bounds'],interval_selection_sha256=sha(interval_path),interval='Target90% empirical signed-residual interval; pooled or Cal-road envelope selected by Cal-CV/Dev interval score. Coverage and width evaluated independently; not a conditional guarantee.')
 dump(out/'calibration.json',calibration)
 # This study's relative-group BO did not improve the primary multibudget
 # objective. Freeze the common policy; retain learned tags for diagnostics.
 policy=dict(perception_identity=identity,model_sha256=modelsha,absolute_quality_threshold=.7,grouping='global',reference={'absolute_threshold':.7,'groups':{}},balanced_floor=0.,audit_fraction=.1,uncertainty_weight=0.,minimum_tag_confidence=.6,cluster_distance_limits=limits,cluster_semantic_mapping=mapping,environment_thresholds_adopted=False,decision='Development BO group references tied or underperformed common priority; no claimed environment-threshold benefit. Refit residuals for each perception version.',automatic_admission=False)
 if a.quality_dir=='g1_quality':policy.update(read(S/'g1_quality/review_policy_frozen.json')['runtime_policy'])
 dump(out/'policy.json',policy)
 manifest=dict(version='environment-quality-review-bundle-2' if interval_path is not None else 'environment-quality-review-bundle-1',quality_model_kind=kind,perception_identity=identity,perception_weights=weights,profile=schema['profile'],regions_sha256=schema['regions_sha256'],checkpoint_sha256=identity['checkpoint_sha256'],ordered_features=selected['columns'],target_names=['whole_frame_lane_iou'],environment_ordered_features=classifier['columns'],environment_representation='Supervised RGB weather/time classifier probabilities, square-root transform, official e-means fixed centers.',artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},quality_selection_sha256=sha(S/a.quality_dir/'quality_selection_frozen.json'),environment_selection_sha256=sha(S/'environment/environment_selection_frozen.json'),source_sha256=sha(__file__))
 dump(out/'bundle_manifest.json',manifest);review=EnvironmentReview(out,perception_identity=identity);print('EXPORTED',kind,out,flush=True)
if __name__=='__main__':main()
