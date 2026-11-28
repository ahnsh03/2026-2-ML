"""Score frozen version-matched review candidates; never retune on confirmation."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
from common.bridge import sha256_file
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.review_targets import support_aware_targets
from evaluate_development import metrics,environment_vectors


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def review_gate(value,random,gate):
    failed=[]
    if value['near_error_positives']<gate['minimum_near_error_support']:failed.append('insufficient_near_error_support')
    if value['near_error_positive_roads']<gate['minimum_error_roads']:failed.append('insufficient_error_segments')
    if value['utility_recall']is None or value['utility_recall']+1e-12<random['utility_recall']*gate['minimum_utility_recall_ratio_vs_random']:failed.append('utility_recall_vs_random')
    if(value['near_error_road_macro']is None or random['near_error_road_macro']is None
            or value['near_error_road_macro']+1e-12<random['near_error_road_macro']+gate['minimum_near_error_road_macro_recall_gain_vs_random']):failed.append('near_error_segment_recall_vs_random')
    return dict(passes=not failed,failed_gates=failed)


def support_summary(selected,target,where):
    """Offline denominators; near retained never means operationally safe."""
    where=np.asarray(where,bool);picked=np.zeros(len(where),bool);picked[np.asarray(selected,int)]=True
    near=target['near'];far=target['far'];retained=(near['gt_pixels']>=50)&(near['predicted_pixels']>=50)&~near['error']
    values=dict(near_gt_supported=near['gt_pixels']>=50,near_prediction_supported=near['predicted_pixels']>=50,
        near_error=near['error'],near_fn=near['false_negative'],near_fp=near['false_positive'],
        near_no_output_fn=near['false_negative']&(near['predicted_pixels']==0),
        near_gt_empty_fp=near['false_positive']&(near['gt_pixels']==0),
        near_retained=retained,near_retained_far_error=retained&far['error'],
        no_supported_error=target['review_utility']==0,offroad_fp=target['offroad']['false_positive'],
        whole_iou_defined=np.isfinite(target['whole_frame_iou']))
    return dict(rows=int(where.sum()),selected_rows=int((where&picked).sum()),
        **{name:int((where&value).sum())for name,value in values.items()},
        **{name+'_selected':int((where&picked&value).sum())for name,value in values.items()},
        scope='Fixed offline support/error definition. Retained near recall and precision are not visibility or safety ground truth.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['freeze','source-index','environment-features','schema','study-design','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);freeze=read(a.freeze);design=read(a.study_design);schema=read(a.schema)
    if freeze['study_design_sha256']!=sha256_file(a.study_design)or not freeze['confirmation_not_collected_at_freeze']:raise ValueError('confirmation design/freeze differs')
    cohort=a.source_index.parent.parent;complete=read(cohort/'COLLECTION_COMPLETE.json')
    if complete['plan_sha256']!=freeze['confirmation_plan_sha256']:raise ValueError('confirmation collection plan changed')
    before=list(cohort.glob('*_before.json'))
    if not before or any(datetime.fromisoformat(read(path)['utc'])<datetime.fromisoformat(freeze['frozen_utc'])for path in before):
        raise ValueError('confirmation collection began before model/policy freeze')
    source=read(a.source_index)['confirmation']['rows'];ids=[r['sample_id']for r in source];results={};predictions={};sets={};reference_gt=None
    for candidate in freeze['review_candidates']:
        name=candidate['name'];bundle=Path(candidate['bundle']);path=Path(candidate['pack'])
        if sha256_file(bundle/'bundle_manifest.json')!=candidate['bundle_manifest_sha256']:raise ValueError('frozen review bundle changed')
        review=SupportAwareReview(bundle,perception_identity=candidate['perception_identity']);receipt=read(path.with_suffix('.json'))
        validate_pack_perception(receipt,review.identity,'model_state_dict')
        if(receipt['sha256']!=sha256_file(path)or receipt['index_sha256']!=sha256_file(a.source_index)
                or receipt['schema_sha256']!=sha256_file(path.parent/'schema.json')):raise ValueError('confirmation feature/GT provenance differs')
        with np.load(path)as z:data={k:z[k]for k in z.files}
        if data['ids'].tolist()!=ids or data['pixel_sha256'].tolist()!=[r['pixel_sha256']for r in source]:raise ValueError('confirmation RGB membership differs')
        gt=data['counts'][:,:,0]+data['counts'][:,:,2]
        if reference_gt is not None and not np.array_equal(gt,reference_gt):raise ValueError('cross-version confirmation GT support differs')
        reference_gt=gt
        x=feature_matrix(data,schema,review.columns);ex=environment_vectors(data,a.environment_features,schema,review.environment.hist_columns)
        pred=review.predict(x,ex);budget=int(np.ceil(len(ids)*design['review_acceptance']['budget']))
        selected,_,details=review.select_with_features(data['ids'],pred,budget,x);target=support_aware_targets(data['counts']);value=metrics(selected,data,target)
        rng=np.random.default_rng(281128);random=[metrics(rng.choice(len(ids),budget,replace=False),data,target)for _ in range(design['review_acceptance']['random_trials'])]
        random_mean={k:float(np.mean([r[k]for r in random if r[k]is not None]))if any(r[k]is not None for r in random)else None for k in random[0]}
        quality=target['whole_frame_iou'];valid=np.isfinite(quality);error=np.abs(pred['estimated_iou']-quality);covered=(quality>=pred['interval90_lower'])&(quality<=pred['interval90_upper'])
        def by_site(values,population=None):
            population=valid if population is None else population
            return {str(site):float(np.mean(values[(data['site']==site)&population]))for site in np.unique(data['site'])if ((data['site']==site)&population).any()}
        mae_by_site=by_site(error);coverage_by_site=by_site(covered)
        supported=valid&(target['whole_gt_pixels']>=freeze.get('quality_handoff_gate',{}).get('minimum_whole_gt_pixels',50))
        supported_mae=by_site(error,supported);supported_coverage=by_site(covered,supported)
        result=dict(perception_identity=review.identity,quality_kind=review.kind,error_kind=review.error_kind,
            quality_columns=len(review.quality_columns),error_columns=len(review.error_columns),quality_defined_rows=int(valid.sum()),
            quality_mae=float(error[valid].mean()),quality_segment_macro_mae=float(np.mean(list(mae_by_site.values()))),quality_mae_by_segment=mae_by_site,
            interval90_coverage=float(covered[valid].mean()),interval90_segment_macro_coverage=float(np.mean(list(coverage_by_site.values()))),interval90_coverage_by_segment=coverage_by_site,
            interval90_mean_width=float(np.mean(pred['interval90_upper']-pred['interval90_lower'])),
            gt_supported_quality_rows=int(supported.sum()),gt_supported_quality_segment_macro_mae=float(np.mean(list(supported_mae.values())))if supported_mae else None,
            gt_supported_interval90_segment_macro_coverage=float(np.mean(list(supported_coverage.values())))if supported_coverage else None,
            policy=review.policy,review=value,random100_mean=random_mean,review_gate=review_gate(value,random_mean,design['review_acceptance']),
            by_hour={str(int(hour)):dict(rows=int((data['hour']==hour).sum()),near_error_positives=int(target['near']['error'][data['hour']==hour].sum()),
                near_error_selected=int(np.sum(target['near']['error'][selected]&(data['hour'][selected]==hour))),selected_rows=int((data['hour'][selected]==hour).sum()))for hour in np.unique(data['hour'])},
            support_by_hour={str(int(hour)):support_summary(selected,target,data['hour']==hour)for hour in np.unique(data['hour'])},
            support_by_segment={str(site):support_summary(selected,target,data['site']==site)for site in np.unique(data['site'])},
            support_by_weather_hour={str(int(weather))+'_'+str(int(hour)):support_summary(selected,target,(data['weather']==weather)&(data['hour']==hour))for weather,hour in sorted(set(zip(data['weather'],data['hour'])))},
            selected_ids=data['ids'][selected].tolist(),selection_details=details,
            source_pack_sha256=sha256_file(path),bundle_manifest_sha256=candidate['bundle_manifest_sha256'],
            diagnostic_scope=candidate.get('diagnostic_scope'),
            scope='Frozen independent same-map confirmation segments. Supported error labels are not operational visibility/safety labels.')
        results[name]=result;predictions[name]=pred;sets[name]=data
        np.savez_compressed(a.output/(name+'_predictions.npz'),ids=data['ids'],**pred)
        review.verify_unchanged();print('CONFIRMATION',name,result['quality_segment_macro_mae'],value,result['review_gate'],flush=True)
    primary=results[freeze['primary_review_name']];parent=results[freeze['parent_review_name']]
    complete_gate=freeze.get('quality_handoff_gate')
    handoff=None
    if complete_gate:
        stable_population=complete_gate.get('population')=='whole_gt_pixels_at_least50'
        error_key='gt_supported_quality_segment_macro_mae'if stable_population else'quality_segment_macro_mae'
        coverage_key='gt_supported_interval90_segment_macro_coverage'if stable_population else'interval90_segment_macro_coverage'
        for result in results.values():
            failed=[]
            if result[error_key]is None or parent[error_key]is None:failed.append('unsupported_quality_population')
            elif result[error_key]>parent[error_key]+complete_gate['maximum_macro_mae_increase_vs_parent_pipeline']+1e-12:failed.append('quality_mae_retention')
            if result[coverage_key]is None or result[coverage_key]+1e-12<complete_gate['minimum_interval90_macro_coverage']:failed.append('quality_interval_coverage')
            result['quality_handoff_gate']=dict(passes=not failed,failed_gates=failed,population=complete_gate.get('population','all_defined'),scope='Practical comparison across explicitly versioned quality targets; not statistical equivalence.')
        handoff=primary['quality_handoff_gate']
    document=dict(version='support-aware-independent-confirmation-1',results=results,primary_name=freeze['primary_review_name'],
        primary_review_passes=primary['review_gate']['passes'],quality_handoff_gate=handoff,
        confirmation_freeze_sha256=sha256_file(a.freeze),source_index_sha256=sha256_file(a.source_index),
        environment_features_sha256=sha256_file(a.environment_features),source_sha256=sha256_file(Path(__file__)),
        no_post_confirmation_retuning=True,automatic_adoption_performed=False,
        scope='Report fixed candidates and their denominators. A better control is not silently substituted after opening confirmation; perception adoption is assessed separately.')
    write(a.output/'summary.json',document);(a.output/'_SUCCESS').write_text(sha256_file(a.output/'summary.json')+'\n')


if __name__=='__main__':main()
