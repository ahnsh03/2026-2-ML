"""Two matched replay-only endpoints after the primary queue releases the GPU."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();S=a.study
 while not (S/'protocol/replay_control_cached_v2.receipt.json').exists() or not (S/'protocol/training_queue_frozen.json').exists():time.sleep(10)
 primary=json.loads((S/'protocol/training_queue_frozen.json').read_text())['queue'];queue=[]
 for seed in [261126,261127]:
  protocol=json.loads((S/f'protocol/training_s{seed}.json').read_text());protocol.update(selection_arms=['ReplayOnly'],slot_source='new front-camera slots replaced by known f0 front train, 0 novel images',control_source='replay_control_v2_frozen.json')
  path=S/f'protocol/replay_only_v2_s{seed}.json'
  if not path.exists():dump(path,protocol)
  queue.append(dict(arm='ReplayOnly',seed=seed,protocol=str(path),run=str(S/f'runs/ReplayOnly_v2_s{seed}')))
 dump(S/'protocol/control_queue_v2_frozen.json',dict(queue=queue,created_utc=datetime.now(timezone.utc).isoformat(),new_images=0,camera_mixture_matched=True))
 while not all((Path(q['run'])/'_SUCCESS').exists() for q in primary):time.sleep(15)
 while subprocess.check_output(['docker','ps','-a','--filter','name=^/ml-selection-training$','--format','{{.Names}}'],text=True).strip():time.sleep(2)
 mounts=['-v',str(a.repo)+':/repo','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src']
 for item in queue:
  run=Path(item['run'])
  if (run/'_SUCCESS').exists():continue
  dump(S/'CONTROL_WORK_IN_PROGRESS.json',dict(status='training',current=item,updated_utc=datetime.now(timezone.utc).isoformat()))
  cmd=['docker','run','--name','ml-selection-training','--rm','--gpus','all','--shm-size','1g',*mounts,'ml-perception:rtx50-cu128','python','-m','quality_estimation.retrain_streaming',
       '--new-manifest',str(S/'protocol/replay_control_cached_v2.jsonl'),'--replay-manifest',str(S/'protocol/replay_cached.jsonl'),'--protocol',item['protocol'],
       '--cache',str(S/'cache'),'--arm','ReplayOnly','--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--output-dir',str(run)]
  if (run/'latest.pt').exists():cmd.append('--resume')
  with (S/f"logs/ReplayOnly_v2_s{item['seed']}.log").open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 dump(S/'CONTROL_WORK_IN_PROGRESS.json',dict(status='complete',queue=queue,completed_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
