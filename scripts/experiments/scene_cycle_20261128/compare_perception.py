"""Choose a cumulative perception candidate using paired Dev/retention only."""
from pathlib import Path
import argparse
import json
import numpy as np
from common.bridge import sha256_file


def load(directory):
    run=json.loads((directory/'summary.json').read_text())
    if (directory/'_SUCCESS').read_text().strip()!=sha256_file(directory/'summary.json'):raise ValueError('perception evaluation incomplete')
    rows=[json.loads(line)for line in (directory/'per_frame.jsonl').read_text().splitlines()]
    if len(rows)!=run['rows']or len({r['sample_id']for r in rows})!=len(rows):raise ValueError('evaluation membership differs')
    return run,rows


def pooled(rows,head):
    c=np.array([r['counts'][head]for r in rows],np.int64).sum(0)
    return float(c[0]/c.sum())if c.sum()else None


def metrics(rows,marked):
    sites={}
    for site in marked:
        selected=[r for r in rows if r['site']==site and sum(r['counts']['lane'][::2])>=50]
        values=[r['counts']['lane'][0]/sum(r['counts']['lane'])for r in selected]
        if not values:raise ValueError('qualified marked segment lacks supported GT')
        sites[site]=float(np.mean(values))
    original=[r for r in rows if r['adaptation_role']=='retention']
    return dict(marked_by_segment=sites,marked_macro=float(np.mean(list(sites.values()))),
        retention={h:pooled(original,h)for h in ['lane','drivable','white','yellow','stopline']},
        retention_by_view={v:{h:pooled([r for r in original if r['view']==v],h)for h in ['lane','drivable']}for v in ['front','left','right']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['study','queue','parent-run','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();s=a.study;queue=json.loads(a.queue.read_text());design=json.loads((s/'protocol/study_design_frozen.json').read_text());gate=design['neural_acceptance']
    plan=json.loads((s/'development/plan_frozen.json').read_text());marked=[r['route_id']for r in plan['routes']if r['expected_support_stratum']=='marked']
    base_run,base_rows=load(s/'perception_evaluations/v6');parent_run,parent_rows=load(s/'perception_evaluations/r1')
    if [r['sample_id']for r in base_rows]!=[r['sample_id']for r in parent_rows]:raise ValueError('paired baseline IDs differ')
    baseline=metrics(base_rows,marked);parent=metrics(parent_rows,marked);records=[]
    for item in queue['candidates']:
        directory=Path(item['evaluation']);run,rows=load(directory)
        if (run['manifest_sha256']!=base_run['manifest_sha256']or run['precision']!=base_run['precision']or run['device']!=base_run['device']
                or[r['sample_id']for r in rows]!=[r['sample_id']for r in base_rows]):raise ValueError('candidate evaluated with different membership/backend')
        m=metrics(rows,marked);delta={site:m['marked_by_segment'][site]-parent['marked_by_segment'][site]for site in marked};fail=[]
        gain=m['marked_macro']-parent['marked_macro']
        if gain<gate['marked_macro_gain_min']:fail.append('marked_macro_gain')
        if sum(d>=gate['per_segment_gain_min']for d in delta.values())<gate['marked_positive_segments_min']:fail.append('marked_positive_segments')
        if min(delta.values())< -gate['per_segment_loss_max']:fail.append('marked_worst_segment')
        for h in ['lane','drivable','white','yellow','stopline']:
            limit=gate['original_pooled_loss_max']if h in ['lane','drivable']else gate['original_class_view_loss_max']
            if m['retention'][h]is not None and baseline['retention'][h]is not None and m['retention'][h]<baseline['retention'][h]-limit:fail.append('retention_'+h)
        for v in m['retention_by_view']:
            for h,value in m['retention_by_view'][v].items():
                ref=baseline['retention_by_view'][v][h]
                if value is not None and ref is not None and value<ref-gate['original_class_view_loss_max']:fail.append('retention_'+v+'_'+h)
        records.append(dict(item,metrics=m,marked_gain_vs_parent=gain,segment_deltas=delta,passes=not fail,failed_gates=fail,
            perception_identity=run['perception_identity'],per_frame_sha256=sha256_file(directory/'per_frame.jsonl')))
    passing=[r for r in records if r['passes']]
    if passing:
        best=max(r['metrics']['marked_macro']for r in passing);near=[r for r in passing if r['metrics']['marked_macro']>=best-.002]
        selected=max(near,key=lambda r:(r['alpha'],r['metrics']['marked_macro']))
    else:selected=dict(run=str(a.parent_run),arm='Retain_r1',alpha=1.,perception_identity=parent_run['perception_identity'],metrics=parent,passes=True,scope='No cumulative candidate passes; retain independently promoted parent rather than adopt a failed child.')
    document=dict(version='cumulative-perception-dev-selection-1',baseline=baseline,parent=parent,records=records,selected=selected,
        candidate_selected_from_passing=bool(passing),source_sha256=sha256_file(Path(__file__)),queue_sha256=sha256_file(a.queue),
        design_sha256=sha256_file(s/'protocol/study_design_frozen.json'),confirmation_accessed=False,
        scope='Paired same-backend development/retention decision only. Coverage24k differs from the matched18k selection controls. A new independent confirmation is still required.')
    if a.output.exists():raise ValueError('perception selection output exists')
    a.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    print('PERCEPTION DEVELOPMENT CHOICE',selected['arm'],selected['alpha'],selected['perception_identity']['checkpoint_sha256'],flush=True)


if __name__=='__main__':main()
