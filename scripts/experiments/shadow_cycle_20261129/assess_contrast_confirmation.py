"""Evaluate frozen RF/GP/contrast policies once on the prospectively collected cohort."""
from pathlib import Path
import argparse,json,math
import numpy as np
from common.bridge import sha256_file
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.contrast_rank import ContrastRankPolicy
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.review_targets import support_aware_targets
from quality_estimation.support_review_selection import select_supported_review
from quality_estimation.spatial_identity import validate_pack_perception


def read(p):return json.loads(p.read_text())
def dump(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def metrics(chosen,data,target):
    pick=np.zeros(len(data['ids']),bool);pick[chosen]=True
    def recall(value,where=None):
        where=np.ones(len(pick),bool)if where is None else where
        denominator=float(np.sum(value[where]));return float(np.sum(value[pick&where])/denominator)if denominator else None
    def macro(value):
        scores=[recall(value,data['site']==road)for road in np.unique(data['site'])];return float(np.mean([s for s in scores if s is not None]))if any(s is not None for s in scores)else None
    near=target['near'];far=target['far'];utility=target['review_utility'];retained=(near['gt_pixels']>=50)&(near['predicted_pixels']>=50)&~near['error']
    def counts(where):
        return dict(rows=int(where.sum()),selected=int((where&pick).sum()),near_errors=int((where&near['error']).sum()),near_errors_selected=int((where&near['error']&pick).sum()),
          near_fn=int((where&near['false_negative']).sum()),near_fn_selected=int((where&near['false_negative']&pick).sum()),near_fp=int((where&near['false_positive']).sum()),near_fp_selected=int((where&near['false_positive']&pick).sum()),
          near_retained_far_error=int((where&retained&far['error']).sum()),near_retained_far_error_selected=int((where&retained&far['error']&pick).sum()),near_error_recall=recall(near['error'],where),utility_recall=recall(utility,where))
    return dict(rows=len(pick),selected=int(pick.sum()),near_errors=int(near['error'].sum()),near_error_segments=int(len(np.unique(data['site'][near['error']]))),near_error_recall=recall(near['error']),near_error_road_macro=macro(near['error']),utility_recall=recall(utility),utility_road_macro=macro(utility),
                near_fn_recall=recall(near['false_negative']),near_fp_recall=recall(near['false_positive']),no_supported_error_selected=int(np.sum(pick&(utility==0))),
                by_hour={str(int(h)):counts(data['hour']==h)for h in np.unique(data['hour'])},by_segment={str(s):counts(data['site']==s)for s in np.unique(data['site'])})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','index','pack','schema','environment-features','rf-bundle','gp-bundle','rf-only-bundle','candidate','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);plan=read(a.plan);receipt=read(a.pack.with_suffix('.json'))
    if plan['gp_candidate_sha256']!=sha256_file(a.candidate)or receipt['sha256']!=sha256_file(a.pack)or receipt['index_sha256']!=sha256_file(a.index):raise ValueError('frozen policy/data changed')
    if receipt['schema_sha256']!=sha256_file(a.schema):raise ValueError('schema changed')
    with np.load(a.pack)as z:data={k:z[k]for k in z.files}
    rows=read(a.index)['confirmation']['rows'];schema=read(a.schema);ids=data['ids']
    if ids.tolist()!=[r['sample_id']for r in rows]or data['pixel_sha256'].tolist()!=[r['pixel_sha256']for r in rows]or len(ids)!=plan['expected_frames']:raise ValueError('registered membership differs')
    with np.load(a.environment_features)as z:
        if z['ids'].tolist()!=ids.tolist():raise ValueError('environment RGB rows differ')
        hist=z['X'];histcols=z['columns'].tolist()
    result={};predictions={};bundles={'frozen_RF':a.rf_bundle,'frozen_GP':a.gp_bundle,'RF_only':a.rf_only_bundle};features={}
    for name,path in bundles.items():
        r=SupportAwareReview(path);validate_pack_perception(receipt,r.identity,'model_state_dict')
        if histcols!=r.environment.hist_columns:raise ValueError('environment column order differs')
        env=np.c_[hist,feature_matrix(data,schema,r.environment.columns[len(histcols):])]
        x=feature_matrix(data,schema,r.columns);features[name]=x;predictions[name]=r.predict(x,env)
        q=predictions[name]['estimated_iou'];gt=support_aware_targets(data['counts'])['whole_frame_iou'];valid=np.isfinite(gt)
        result[name]=dict(quality_kind=r.kind,error_kind=r.error_kind,quality_mae=float(np.mean(abs(q[valid]-gt[valid]))),quality_road_macro_mae=float(np.mean([np.mean(abs(q[(data['site']==road)&valid]-gt[(data['site']==road)&valid]))for road in np.unique(data['site'])if((data['site']==road)&valid).any()])),bundle_manifest_sha256=sha256_file(path/'bundle_manifest.json'))
    identity=SupportAwareReview(a.rf_bundle).identity;policy=ContrastRankPolicy(a.candidate,a.rf_bundle,a.gp_bundle,identity)
    if not np.array_equal(predictions['frozen_RF']['estimated_iou'],predictions['frozen_GP']['estimated_iou']):raise ValueError('quality GP controls differ')
    scores={name:pred['utility_raw_mean']for name,pred in predictions.items()};scores['RF_rank_plus_z_0.25']=policy.score(predictions['frozen_RF'],predictions['frozen_GP'])
    target=support_aware_targets(data['counts']);selections={};window_results={}
    for mode in ['global','window100']:
        selections[mode]={};window_results[mode]={}
        for name,score in scores.items():
            parts=[];windows=[]
            for start in range(0,len(ids),len(ids)if mode=='global'else 100):
                end=min(len(ids),start+(len(ids)if mode=='global'else 100));budget=int(math.ceil(.1*(end-start)))
                ix,_,detail=select_supported_review(ids[start:end],score[start:end],np.zeros(end-start),np.zeros(end-start,int),budget,audit_fraction=.1,seed=281128)
                parts.extend((ix+start).tolist());windows.append(dict(start=start,stop=end,budget=budget,selection=detail))
            chosen=np.asarray(parts);selections[mode][name]=chosen;window_results[mode][name]=dict(metrics=metrics(chosen,data,target),windows=windows,selected_ids=ids[chosen].tolist())
    rng=np.random.default_rng(291129);random=[];budget=int(math.ceil(len(ids)*.1))
    for _ in range(1000):random.append(metrics(rng.choice(len(ids),budget,replace=False),data,target))
    random_summary={k:float(np.mean([r[k]for r in random if r[k]is not None]))for k in ['near_error_recall','near_error_road_macro','utility_recall','utility_road_macro','near_fn_recall','near_fp_recall']if any(r[k]is not None for r in random)}
    candidate=window_results['global']['RF_rank_plus_z_0.25']['metrics'];base=window_results['global']['frozen_RF']['metrics'];gates=plan['acceptance'];failed=[]
    if candidate['near_errors']<gates['minimum_near_errors']:failed.append('insufficient_near_error_support')
    if candidate['near_error_segments']<gates['minimum_error_segments']:failed.append('insufficient_error_segments')
    utility_gain=None if candidate['utility_road_macro']is None or base['utility_road_macro']is None else candidate['utility_road_macro']-base['utility_road_macro']
    near_gain=None if candidate['near_error_recall']is None or base['near_error_recall']is None else candidate['near_error_recall']-base['near_error_recall']
    if utility_gain is None or utility_gain<gates['minimum_road_macro_utility_gain']-1e-12:failed.append('road_macro_utility_gain')
    if near_gain is None or near_gain<gates['minimum_near_recall_gain']-1e-12:failed.append('near_error_recall_gain')
    roads=np.unique(data['site']);deltas={}
    for name in ['utility_recall','near_error_recall']:
        left=np.array([candidate['by_segment'][str(r)][name]for r in roads],float);right=np.array([base['by_segment'][str(r)][name]for r in roads],float);valid=np.isfinite(left)&np.isfinite(right);diff=left[valid]-right[valid]
        if len(diff):
            boot=np.random.default_rng(291129).choice(diff,size=(10000,len(diff)),replace=True).mean(1)
            deltas[name]=dict(road_differences=diff.tolist(),roads=int(len(diff)),mean=float(diff.mean()),paired_road_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),scope='Fixed selected sets; resample physical segments, never treat each frame as independent. Six-block interval is imprecise.')
    summary=dict(schema='frozen-contrast-confirmation-assessment-1',rows=len(ids),unique_rgb_pixels=len(set(data['pixel_sha256'])),segments=len(roads),perception_identity=identity,
        policy_frozen_before_collection=True,retuned_on_confirmation=False,quality=result,selection=window_results,random_1000=random_summary,
        candidate_gate=dict(passes=not failed,failed=failed,utility_road_macro_gain=utility_gain,near_recall_gain=near_gain),paired_road_bootstrap=deltas,
        source_sha256=sha256_file(Path(__file__)),input_sha256={k:sha256_file(getattr(a,k))for k in ['plan','index','pack','schema','environment_features','candidate']},scope='Static newly sampled same-map segments, native pixel-supported event targets, no claim of physical visibility/safety truth or absent original perception-training geography. RF/GP family decision remains deferred.')
    for name,pred in predictions.items():np.savez_compressed(a.output/(name+'_predictions.npz'),ids=ids,**pred)
    np.savez_compressed(a.output/'scores.npz',ids=ids,**scores);dump(a.output/'summary.json',summary);(a.output/'_SUCCESS').write_text(sha256_file(a.output/'summary.json')+'\n')
    print(json.dumps(dict(quality=result,candidate_gate=summary['candidate_gate'],paired_road_bootstrap=deltas),indent=2))
if __name__=='__main__':main()
