"""Full-original-only continuation control, concurrent only after primary training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time
from study_data import read,dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();S=a.study;P=a.parent
 while not (S/'reports/development_evaluation_complete.json').exists():time.sleep(15)
 protocol=read(S/'protocol/training_271127.json');protocol.update(selection_arms=['ReplayOnly'],new_per_batch=0,replay_per_batch=16,new_rows=0,expected_new_pool_passes=0.,expected_original_pool_passes=384000/26001,interpretation='Full26001 originals only; cycling permutations,24kupdates,batch16, no new admission; teacher sees16 originals.')
 spec=S/'protocol/replay_full_control_frozen.json'
 if spec.exists():assert read(spec)==protocol
 else:dump(spec,protocol)
 empty=S/'protocol/no_new_admissions.jsonl';empty.write_text('');run=S/'runs/ReplayFull_s271127';mount=['-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src:/repo/ros/src/data_collection/src','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1']
 def call(name,cmd,gpu=False):
  with (S/'logs'/(name+'.log')).open('a') as log:subprocess.run(['docker','run','--rm','--name','ml-environment-replay-control',*(['--gpus','all','--shm-size','1g'] if gpu else []),*mount,'ml-perception:rtx50-cu128',*cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
 if not (run/'_SUCCESS').exists():
  cmd=['python','-m','quality_estimation.retrain_streaming','--new-manifest',str(empty),'--replay-manifest',str(S/'protocol/full_original_cached.jsonl'),'--protocol',str(spec),'--cache',str(P/'cache'),'--arm','ReplayOnly','--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--output-dir',str(run)]
  if (run/'latest.pt').exists():cmd.append('--resume')
  call('train_ReplayFull_s271127',cmd,True)
 items=[]
 for alpha in [1.,.25,.5,.75]:
  name='ReplayFull_s271127' if alpha==1. else f'ReplayFull_s271127_a{int(alpha*100):03d}';root=run if alpha==1. else S/'derived'/name
  if alpha!=1. and not (root/'_SUCCESS').exists():call('derive_'+name,['python','-m','quality_estimation.blend_perception','--initial-checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--parent-run',str(run),'--extension-protocol',str(S/'protocol/interpolation_frozen.json'),'--output',str(root),'--alpha',str(alpha)])
  out=S/'evaluations'/name/'development_retention'
  if not (out/'_SUCCESS').exists():call('dev_'+name,['python','-m','quality_estimation.evaluate_cached_perception','--manifest',str(P/'protocol/development_retention_eval.jsonl'),'--cache',str(P/'cache'),'--output',str(out),'--batch-size','4','--workers','2','--role','development_retention','--adaptation-run',str(root)],True)
  items.append(dict(name=name,alpha=alpha,run=str(root),evaluation=str(out)))
 dump(S/'reports/full_replay_control_complete.json',dict(models=items,optimizer_seeds=1,completed_utc=datetime.now(timezone.utc).isoformat(),fresh_confirmation_not_used=True));print('REPLAY CONTROL COMPLETE',flush=True)
if __name__=='__main__':main()
