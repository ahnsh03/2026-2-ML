"""Check a frozen cumulative neural update on new paired confirmation segments."""
from pathlib import Path
import argparse
import json
import numpy as np
from common.bridge import sha256_file


def read(path):return json.loads(Path(path).read_text())


def supported_metrics(rows,marked):
    values={};per_head={}
    for site in marked:
        current=[r for r in rows if r['site']==site and r['counts']['lane'][0]+r['counts']['lane'][2]>=50]
        if not current:raise ValueError('frozen marked segment lacks supported GT')
        values[site]=float(np.mean([r['counts']['lane'][0]/sum(r['counts']['lane'])for r in current]))
    for head in ['lane','drivable','white','yellow','stopline']:
        counts=np.asarray([r['counts'][head]for r in rows],np.int64).sum(0);per_head[head]=float(counts[0]/counts.sum())if counts.sum()else None
    return dict(marked_macro=float(np.mean(list(values.values()))),marked_by_segment=values,all_frame_pooled=per_head)


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
    for n in ['freeze','evaluations','study-design','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();freeze=read(a.freeze);design=read(a.study_design);sources=read(a.evaluations);loaded={};summaries={};expected_ids=None;manifest=None
    if freeze['study_design_sha256']!=sha256_file(a.study_design):raise ValueError('neural confirmation design changed')
    for name,directory in sources.items():
        directory=Path(directory);summary=read(directory/'summary.json')
        if(directory/'_SUCCESS').read_text().strip()!=sha256_file(directory/'summary.json')or summary['role']!='confirmation':raise ValueError('neural confirmation incomplete or wrong role')
        if summary['perception_identity']!=freeze['perception_models'][name]['perception_identity']:raise ValueError('frozen neural identity differs')
        rows=[json.loads(line)for line in (directory/'per_frame.jsonl').read_text().splitlines()];ids=[r['sample_id']for r in rows]
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
    document=dict(version='cumulative-neural-confirmation-1',scores=scores,marked_gain_vs_parent=gain,marked_segment_deltas=deltas,
        selected_is_existing_parent=unchanged,new_child_passes=not fail and not unchanged,retained_parent_valid=unchanged,failed_gates=fail,
        paired_condition_bootstrap=condition_bootstrap(selected,parent,marked),source_freeze_sha256=sha256_file(a.freeze),source_sha256=sha256_file(Path(__file__)),
        evaluation_summary_sha256={name:sha256_file(Path(path)/'summary.json')for name,path in sources.items()},
        source_per_frame_sha256={name:sha256_file(Path(path)/'per_frame.jsonl')for name,path in sources.items()},
        no_post_confirmation_retuning=True,automatic_adoption_performed=False,
        scope='New same-map qualified segments, fixed before model prediction. Original retention checked separately before capture. Perception and review policy adoption remain separate; one optimizer seed is not repeated training evidence.')
    if a.output.exists():raise ValueError('neural confirmation report exists')
    a.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n');print('NEURAL CONFIRMATION',gain,fail,'unchanged',unchanged,flush=True)


if __name__=='__main__':main()
