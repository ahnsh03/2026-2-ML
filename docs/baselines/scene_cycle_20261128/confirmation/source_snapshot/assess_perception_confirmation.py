"""Check a frozen cumulative neural update on new paired confirmation segments."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
from common.bridge import sha256_file
from quality_estimation.review_targets import support_aware_targets


def read(path):return json.loads(Path(path).read_text())


def validate_source_membership(rows,source):
    if len(rows)!=len(source):raise ValueError('neural/source confirmation counts differ')
    for row,expected in zip(rows,source):
        if(row['sample_id']!=expected['sample_id'] or row['pixel_sha256']!=expected['pixel_sha256']
                or row['adaptation_role']!='confirmation'):
            raise ValueError('neural confirmation membership is not the newly frozen RGB population')


def band_summary(counts):
    counts=np.asarray(counts,np.int64);target=support_aware_targets(counts);summary={}
    for name,regions in [('near',[6,7]),('far',[4,5]),('offroad',[0,1,2,3,8])]:
        band=target[name];tp,fp,fn=counts[:,regions].sum((0,1));n=len(counts)
        summary[name]=dict(rows=n,pooled_tp=int(tp),pooled_fp=int(fp),pooled_fn=int(fn),
            pooled_precision=float(tp/(tp+fp))if tp+fp else None,pooled_recall=float(tp/(tp+fn))if tp+fn else None,
            gt_supported_rows=int((band['gt_pixels']>=50).sum()),prediction_supported_rows=int((band['predicted_pixels']>=50).sum()),
            supported_fn_frames=int(band['false_negative'].sum()),supported_fp_frames=int(band['false_positive'].sum()),
            supported_error_frames=int(band['error'].sum()),supported_error_frame_rate=float(band['error'].mean()),
            empty_gt_fp_frames=int((band['false_positive']&(band['gt_pixels']==0)).sum()),
            missing_output_fn_frames=int((band['false_negative']&(band['predicted_pixels']==0)).sum()))
    return summary


def supported_metrics(rows,marked):
    values={};per_head={}
    for site in marked:
        current=[r for r in rows if r['site']==site and r['counts']['lane'][0]+r['counts']['lane'][2]>=50]
        if not current:raise ValueError('frozen marked segment lacks supported GT')
        values[site]=float(np.mean([r['counts']['lane'][0]/sum(r['counts']['lane'])for r in current]))
    for head in ['lane','drivable','white','yellow','stopline']:
        counts=np.asarray([r['counts'][head]for r in rows],np.int64).sum(0);per_head[head]=float(counts[0]/counts.sum())if counts.sum()else None
    counts=np.asarray([r['region_counts']for r in rows],np.int64)
    return dict(marked_macro=float(np.mean(list(values.values()))),marked_by_segment=values,all_frame_pooled=per_head,
        image_band_errors=band_summary(counts),
        image_band_errors_by_hour={str(int(hour)):band_summary(counts[np.array([r['hour']==hour for r in rows])])for hour in sorted({r['hour']for r in rows})},
        image_band_scope='Fixed offline support50px/precision-recall0.8 definition; near/far are image bands, not metric distance or safety/visibility labels. No online ROI IoU is used.')


def condition_bootstrap(selected,parent,marked):
    rng=np.random.default_rng(281128);blocks={}
    for site in marked:
        groups={}
        for left,right in zip(selected,parent):
            if left['site']!=site or left['counts']['lane'][0]+left['counts']['lane'][2]<50:continue
            gain=left['counts']['lane'][0]/sum(left['counts']['lane'])-right['counts']['lane'][0]/sum(right['counts']['lane'])
            groups.setdefault((left['weather'],left['hour']),[]).append(gain)
        blocks[site]=[np.asarray(v)for v in groups.values()]
    draws=[]
    for _ in range(1000):
        estimates=[]
        for values in blocks.values():estimates.append(float(np.concatenate([values[i]for i in rng.integers(len(values),size=len(values))]).mean()))
        draws.append(float(np.mean(estimates)))
    return dict(gain95_interval=np.quantile(draws,[.025,.975]).tolist(),resamples=1000,
        condition_blocks_by_segment={s:len(v)for s,v in blocks.items()},
        scope='Paired weather/hour block bootstrap conditional on these fixed map segments. Not a new-city or road-population confidence interval.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['freeze','evaluations','source-index','study-design','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();freeze=read(a.freeze);design=read(a.study_design);sources=read(a.evaluations);loaded={};summaries={};expected_ids=None;manifest=None
    if freeze['study_design_sha256']!=sha256_file(a.study_design):raise ValueError('neural confirmation design changed')
    cohort=a.source_index.parent.parent;complete=read(cohort/'COLLECTION_COMPLETE.json');before=list(cohort.glob('*_before.json'))
    if complete['plan_sha256']!=freeze['confirmation_plan_sha256'] or not before:
        raise ValueError('neural confirmation collection plan is not frozen')
    if any(datetime.fromisoformat(read(path)['utc'])<datetime.fromisoformat(freeze['frozen_utc'])for path in before):
        raise ValueError('neural confirmation collection began before candidate freeze')
    source=read(a.source_index)['confirmation']['rows']
    for name,directory in sources.items():
        directory=Path(directory);summary=read(directory/'summary.json')
        if(directory/'_SUCCESS').read_text().strip()!=sha256_file(directory/'summary.json')or summary['role']!='confirmation':raise ValueError('neural confirmation incomplete or wrong role')
        if summary['perception_identity']!=freeze['perception_models'][name]['perception_identity']:raise ValueError('frozen neural identity differs')
        rows=[json.loads(line)for line in (directory/'per_frame.jsonl').read_text().splitlines()];ids=[r['sample_id']for r in rows]
        validate_source_membership(rows,source)
        if len(set(ids))!=len(ids)or len(ids)!=summary['rows']:raise ValueError('neural confirmation membership differs')
        if expected_ids is not None and(ids!=expected_ids or summary['manifest_sha256']!=manifest):raise ValueError('neural models evaluated on different pixels')
        if summaries and any(summary[k]!=next(iter(summaries.values()))[k]for k in ['device','precision']):raise ValueError('paired neural backend differs')
        expected_ids=ids;manifest=summary['manifest_sha256'];loaded[name]=rows;summaries[name]=summary
    selected=loaded['selected'];parent=loaded['r1']
    for rows in loaded.values():
        for left,right in zip(rows,parent):
            for head in ['lane','drivable','white','yellow','stopline']:
                if left['counts'][head][0]+left['counts'][head][2]!=right['counts'][head][0]+right['counts'][head][2]:raise ValueError('paired neural GT differs')
            if not np.array_equal(np.asarray(left['region_counts'])[:,0]+np.asarray(left['region_counts'])[:,2],np.asarray(right['region_counts'])[:,0]+np.asarray(right['region_counts'])[:,2]):raise ValueError('paired image-band GT differs')
    marked=freeze['marked_confirmation_sites'];scores={name:supported_metrics(rows,marked)for name,rows in loaded.items()};gate=design['neural_acceptance']
    gain=scores['selected']['marked_macro']-scores['r1']['marked_macro'];deltas={site:scores['selected']['marked_by_segment'][site]-scores['r1']['marked_by_segment'][site]for site in marked}
    unchanged=freeze['perception_models']['selected']['perception_identity']==freeze['perception_models']['r1']['perception_identity'];fail=[]
    if not unchanged:
        if gain<gate['marked_macro_gain_min']:fail.append('marked_macro_gain')
        if sum(v>=gate['per_segment_gain_min']for v in deltas.values())<gate['marked_positive_segments_min']:fail.append('marked_positive_segments')
        if min(deltas.values())< -gate['per_segment_loss_max']:fail.append('marked_worst_segment')
        if not freeze['preconfirmation_perception_checks']['cpu_development_passes']:fail.append('development_check')
        if not freeze['preconfirmation_perception_checks']['gpu_original_retention_passes']:fail.append('original_retention_check')
    matched=None
    if 'risk_raw'in loaded and 'random_raw'in loaded:
        matched=dict(marked_macro_risk_minus_random=scores['risk_raw']['marked_macro']-scores['random_raw']['marked_macro'],
            marked_segment_deltas={site:scores['risk_raw']['marked_by_segment'][site]-scores['random_raw']['marked_by_segment'][site]for site in marked},
            paired_condition_bootstrap=condition_bootstrap(loaded['risk_raw'],loaded['random_raw'],marked),
            scope='Frozen raw18k endpoints, same parent/replay/environment counts and optimizer seed. Conditional segment evidence, not repeated-training evidence or a post-test adoption pool.')
    document=dict(version='cumulative-neural-confirmation-1',scores=scores,marked_gain_vs_parent=gain,marked_segment_deltas=deltas,
        selected_is_existing_parent=unchanged,new_child_passes=not fail and not unchanged,retained_parent_valid=unchanged,failed_gates=fail,
        paired_condition_bootstrap=condition_bootstrap(selected,parent,marked),source_freeze_sha256=sha256_file(a.freeze),source_sha256=sha256_file(Path(__file__)),
        matched_selection_effect=matched,
        source_index_sha256=sha256_file(a.source_index),
        evaluation_summary_sha256={name:sha256_file(Path(path)/'summary.json')for name,path in sources.items()},
        source_per_frame_sha256={name:sha256_file(Path(path)/'per_frame.jsonl')for name,path in sources.items()},
        no_post_confirmation_retuning=True,automatic_adoption_performed=False,
        scope='New same-map qualified segments, fixed before model prediction. Original retention checked separately before capture. Perception and review policy adoption remain separate; one optimizer seed is not repeated training evidence.')
    if a.output.exists():raise ValueError('neural confirmation report exists')
    a.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n');print('NEURAL CONFIRMATION',gain,fail,'unchanged',unchanged,flush=True)


if __name__=='__main__':main()
