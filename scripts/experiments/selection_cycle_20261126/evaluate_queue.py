"""Evaluate completed endpoints on reserved development and original retention only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--queue-file',default='training_queue_frozen.json');p.add_argument('--completion-name',default='development_evaluations_complete.json');a=p.parse_args();S=a.study;N=S.parent/'ml_roi9_drive_20261126'
 while not (S/'protocol/evaluation_cached.receipt.json').exists():time.sleep(10)
 if a.queue_file!='training_queue_frozen.json':
  while not (S/'reports/development_evaluations_complete.json').exists():time.sleep(15)
 rows=[json.loads(s) for s in (S/'protocol/evaluation_cached.jsonl').read_text().splitlines()];cached=[json.loads(s) for s in (S/'protocol/cached_sources.jsonl').read_text().splitlines()]
 devindex=json.loads((N/'registry_development_final/index_thinned.json').read_text());dev_ids={r['sample_id'] for v in devindex.values() for r in v['rows']}
 dev=[r for r in rows if r['adaptation_role']=='development' and (r['sample_id'] in dev_ids or r['source_key'] in ['z05','z10'])]
 retention=[r for r in cached if r['adaptation_role']=='retention'];combined=dev+retention
 jsonl(S/'protocol/development_retention_eval.jsonl',combined)
 freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),development=len(dev),retention=len(retention),
      manifest_sha256=__import__('hashlib').sha256((S/'protocol/development_retention_eval.jsonl').read_bytes()).hexdigest(),
      primary_development='26R1 three drives,3m-thinned, reported separately from historical static development',
      adoption_gate='Positive 26R1 development macro-road IoU gain; original pooled lane and drivable loss <=0.005. Group/condition diagnostics reported. Select by development only; heldout results opened after freeze.')
 if not (S/'protocol/evaluation_frozen.json').exists():dump(S/'protocol/evaluation_frozen.json',freeze)
 else:
  original=json.loads((S/'protocol/evaluation_frozen.json').read_text())
  if any(original[k]!=freeze[k] for k in ['development','retention','manifest_sha256']):raise ValueError('frozen evaluation membership changed')
 baseline=dict(name='f0',run=None);done=set()
 mount=['-v',str(a.repo)+':/repo','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src']
 while True:
  queuepath=S/'protocol'/a.queue_file
  if not queuepath.exists():time.sleep(10);continue
  queue=json.loads(queuepath.read_text())['queue'];models=[baseline]+[dict(name=Path(q['run']).name,run=q['run']) for q in queue if (Path(q['run'])/'_SUCCESS').exists()]
  for item in models:
   destination=S/'evaluations'/item['name']/'development_retention'
   if item['name'] in done or (destination/'_SUCCESS').exists():done.add(item['name']);continue
   command=['docker','run','--name','ml-selection-evaluation','--rm','--gpus','all','--shm-size','1g',*mount,'ml-perception:rtx50-cu128','python','-m','quality_estimation.evaluate_cached_perception',
            '--manifest',str(S/'protocol/development_retention_eval.jsonl'),'--cache',str(S/'cache'),'--output',str(destination),'--batch-size','4','--workers','2','--role','development_retention']
   command+=['--adaptation-run',item['run']] if item['run'] else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt']
   with (S/f"logs/evaluate_{item['name']}.log").open('a') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
   done.add(item['name']);print('EVALUATED',item['name'],flush=True)
  if len(done)==len(queue)+1:break
  time.sleep(15)
 dump(S/'reports'/a.completion_name,dict(models=sorted(done),heldout_accessed=False,completed_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
