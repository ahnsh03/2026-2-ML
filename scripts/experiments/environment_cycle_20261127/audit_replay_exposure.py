"""Reconstruct frozen sampler draws and check complete-run exposure receipts."""
from pathlib import Path
from collections import Counter
import argparse,json
import numpy as np
from quality_estimation.retrain_streaming import BalancedBatches
from study_data import read,dump,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study
 manifest=S/'protocol/full_original_cached.jsonl';rows=[json.loads(line) for line in manifest.read_text().splitlines()];assert len(rows)==26001
 runs=np.array([r['run_id'] for r in rows]);views=np.array([r['view'] for r in rows]);groups=sorted(set(runs));reports=[]
 for seed,new_rows,new_batch,original_batch in [(271127,3072,2,14),(271128,3072,2,14),(271127,0,0,16)]:
  counts=np.zeros(len(rows),np.int64);new_counts=np.zeros(new_rows,np.int64)
  for batch in BalancedBatches(new_rows,len(rows),new_batch,original_batch,24000,seed):
   indices=np.asarray(batch);np.add.at(counts,indices[indices>=new_rows]-new_rows,1)
   if new_rows:np.add.at(new_counts,indices[indices<new_rows],1)
  assert counts.min()>0 and counts.max()-counts.min()<=1
  original_share=np.array([(runs==g).mean() for g in groups]);draw_share=np.array([counts[runs==g].sum()/counts.sum() for g in groups])
  reports.append(dict(seed=seed,new_rows=new_rows,original_rows=len(rows),new_draws=int(new_counts.sum()),original_draws=int(counts.sum()),original_min_draws_per_image=int(counts.min()),original_max_draws_per_image=int(counts.max()),new_min_draws_per_image=int(new_counts.min()) if new_rows else None,new_max_draws_per_image=int(new_counts.max()) if new_rows else None,source_total_variation=float(abs(original_share-draw_share).sum()/2),sources=[dict(source='Source'+str(i+1),rows=int((runs==g).sum()),population_fraction=float(original_share[i]),draw_fraction=float(draw_share[i])) for i,g in enumerate(groups)],views={v:dict(rows=int((views==v).sum()),draws=int(counts[views==v].sum())) for v in sorted(set(views))}))
 verified=[]
 for success in sorted((S/'runs').glob('*/_SUCCESS')):
  run=success.parent;summary=read(run/'summary.json');config=read(run/'protocol.json');matched=next(r for r in reports if r['seed']==summary['seed'] and r['new_rows']==summary['new_rows']);assert summary['new_draws']==matched['new_draws'] and summary['replay_draws']==matched['original_draws'];assert sha(run/'source/retrain_streaming.py')==summary['source_sha256']['retrain_streaming.py'];verified.append(run.name)
 dump(S/'reports/original_sampler_exposure_audit.json',dict(variants=reports,verified_complete_run_receipts=verified,original_manifest_sha256=sha(manifest),source_sha256=sha(__file__),scope='Exact deterministic sampler reconstruction, checked against complete-run draw totals and source snapshots; not independent per-image disk-read telemetry. Incomplete/future runs are not counted as completed.'))
 print('ORIGINAL EXPOSURE AUDIT',len(reports),'sampler variants;',len(verified),'complete receipts',flush=True)
if __name__=='__main__':main()
