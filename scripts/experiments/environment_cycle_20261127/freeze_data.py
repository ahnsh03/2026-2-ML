"""Promote opened diagnostics explicitly; keep development/calibration separate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,collections
import numpy as np
from scipy.spatial import cKDTree

def lighting(hour):
    return 0 if 8<=int(hour)<18 else 1 if int(hour) in [6,7,18,19] else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;P=a.parent
    old=json.loads((P/'protocol/quality_index_frozen.json').read_text());z=np.load(P/'source_packs_v2.npz');lookup={s:i for i,s in enumerate(z['ids'])};q=np.divide(z['counts'][:,:,0].sum(1),z['counts'].sum((1,2)),out=np.full(len(z['ids']),np.nan),where=z['counts'].sum((1,2))>0)
    pool=old['quality_fit']['rows'];groups=collections.defaultdict(list);rng=np.random.default_rng(271127)
    for r in pool:
        if np.isfinite(q[lookup[r['sample_id']]]):groups[(r['weather'],lighting(r['hour']))].append(r)
    for k,v in groups.items():rng.shuffle(v)
    selected=[]
    while len(selected)<1666:
        for key in sorted(groups):
            if groups[key] and len(selected)<1666:selected.append(groups[key].pop())
    promoted=old['heldout_diagnostic']['rows'];assert len(promoted)==834 and all(np.isfinite(q[lookup[r['sample_id']]]) for r in promoted)
    fit=[dict(r,adaptation_role='quality_fit',previous_adaptation_role=r['adaptation_role'],role_transition='Previously opened diagnostics explicitly promoted; new final confirmation required') for r in selected+promoted]
    index=dict(quality_fit=dict(simulator_release='mixed; per-row release retained',rows=fit),development=old['development'],calibration=old['calibration'])
    for first,second in [('quality_fit','development'),('quality_fit','calibration'),('development','calibration')]:
        x=index[first]['rows'];y=index[second]['rows'];assert not {r['sample_id'] for r in x}&{r['sample_id'] for r in y}
        distance=float(cKDTree([r['xy'] for r in x]).query([r['xy'] for r in y])[0].min());assert distance>=25,(first,second,distance)
    target=S/'protocol/quality_index_frozen.json';assert not target.exists();target.write_text(json.dumps(index,indent=2)+'\n')
    (S/'protocol/promoted_environment_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in promoted))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),old_quality_fit_used=1666,opened_diagnostic_rows_promoted=834,quality_fit=2500,development=len(index['development']['rows']),calibration=len(index['calibration']['rows']),source_index_sha256=hashlib.sha256((P/'protocol/quality_index_frozen.json').read_bytes()).hexdigest(),new_index_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),old_diagnostics_no_longer_test=True,fresh_confirmation_required=True,scope='Q-only regression retains GT-sparse defined IoU; near/far undefined labels do not remove primary Q rows. All primary metadata and source hashes retained.')
    (S/'reports/role_transition.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

if __name__=='__main__':main()
