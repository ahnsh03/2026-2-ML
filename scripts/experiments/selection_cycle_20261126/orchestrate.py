"""Durable serial-GPU matched-budget adaptation queue; no model adoption implicit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();S=a.study
 while not (S/'protocol/cached_sources.receipt.json').exists():
  dump(S/'WORK_IN_PROGRESS.json',dict(status='waiting_verified_cache',updated_utc=datetime.now(timezone.utc).isoformat()));time.sleep(5)
 cached=[json.loads(s) for s in (S/'protocol/cached_sources.jsonl').read_text().splitlines()];lookup={r['sample_id']:r for r in cached}
 if len(lookup)!=len(cached):raise ValueError('cache ID collision')
 replay=[r for r in cached if r['adaptation_role']=='replay']
 if len(replay)!=4096 or len({r['pixel_sha256'] for r in replay})!=4096:raise ValueError('replay duplicates or incomplete pool')
 jsonl(S/'protocol/replay_cached.jsonl',replay)
 for arm in ['Random','QualityRank','ClusterBO']:
  selected=[json.loads(s) for s in (S/f'protocol/{arm}_sources.jsonl').read_text().splitlines()]
  rows=[dict(lookup[r['sample_id']],selection=r['selection'],label_admission=r['label_admission']) for r in selected]
  if len(rows)!=3072 or len({r['pixel_sha256'] for r in rows})!=3072:raise ValueError('admission duplicates')
  jsonl(S/f'protocol/{arm}_cached.jsonl',rows)
 protocol=dict(schema_version='long-adaptation-1',selection_arms=['Random','QualityRank','ClusterBO'],seed=261126,steps=24000,
               new_per_batch=8,replay_per_batch=8,batch_size=16,learning_rate=1e-5,minimum_lr_factor=.1,warmup_steps=200,
               weight_decay=1e-4,gradient_clip_norm=10.,precision='BF16',loss_precision='FP32 reductions',workers=4,checkpoint_every=500,
               replay_distillation_weight=.25,distillation_temperature=2.,augmentation='none; identical native letterbox',
               normalization='BatchNorm running moments frozen; trainable affine',
               interpretation='Two matched seeds; fixed endpoints, common replay index stream per seed, 3072 distinct admissions/4096 replay; no test model selection',
               expected_wall_seconds_per_run=4500,expected_new_pool_passes=62.5,expected_replay_pool_passes=46.875)
 queue=[]
 for seed in [261126,261127]:
  # Alternate ordering to reduce any temperature/time confound across methods.
  for arm in (['Random','QualityRank','ClusterBO'] if seed==261126 else ['ClusterBO','QualityRank','Random']):
   path=S/f'protocol/training_s{seed}.json'
   if not path.exists():dump(path,dict(protocol,seed=seed))
   queue.append(dict(arm=arm,seed=seed,protocol=str(path),run=str(S/f'runs/{arm}_s{seed}')))
 if not (S/'protocol/training_queue_frozen.json').exists():dump(S/'protocol/training_queue_frozen.json',dict(queue=queue,created_utc=datetime.now(timezone.utc).isoformat(),primary_selection='Development gain with retention noninferiority; never final diagnostics'))
 mount=['-v',str(a.repo)+':/repo','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src:/repo/ros/src/data_collection/src']
 for j,item in enumerate(queue):
  run=Path(item['run'])
  if (run/'_SUCCESS').exists():continue
  dump(S/'WORK_IN_PROGRESS.json',dict(status='training',updated_utc=datetime.now(timezone.utc).isoformat(),current=item,completed=[q for q in queue if (Path(q['run'])/'_SUCCESS').exists()],remaining=len(queue)-j))
  command=['docker','run','--name','ml-selection-training','--rm','--gpus','all','--shm-size','1g',*mount,'ml-perception:rtx50-cu128','python','-m','quality_estimation.retrain_streaming',
           '--new-manifest',str(S/f"protocol/{item['arm']}_cached.jsonl"),'--replay-manifest',str(S/'protocol/replay_cached.jsonl'),'--protocol',item['protocol'],
           '--cache',str(S/'cache'),'--arm',item['arm'],'--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--output-dir',str(run)]
  if (run/'latest.pt').exists():command.append('--resume')
  with (S/f"logs/{item['arm']}_s{item['seed']}.log").open('a') as log:
   print('START',item,flush=True);subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True);print('COMPLETE',item,flush=True)
 dump(S/'WORK_IN_PROGRESS.json',dict(status='training_complete_evaluation_pending',updated_utc=datetime.now(timezone.utc).isoformat(),queue=queue))
if __name__=='__main__':main()
