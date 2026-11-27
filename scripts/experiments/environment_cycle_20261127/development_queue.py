"""Mutable training queue, symmetric Dev endpoints and predeclared repeat rule."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time,sys
from study_data import read,dump,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();S=a.study;P=a.parent
 mount=['-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1']
 def run(command,log,gpu=False):
  with (S/'logs'/log).open('a') as f:subprocess.run(['docker','run','--rm','--name','ml-environment-dev-eval',*(['--gpus','all','--shm-size','1g'] if gpu else []),*mount,'ml-perception:rtx50-cu128',*command],stdout=f,stderr=subprocess.STDOUT,check=True)
 def evaluate(name,path=None):
  out=S/'evaluations'/name/'development_retention'
  if not (out/'_SUCCESS').exists():
   cmd=['python','-m','quality_estimation.evaluate_cached_perception','--manifest',str(P/'protocol/development_retention_eval.jsonl'),'--cache',str(P/'cache'),'--output',str(out),'--batch-size','4','--workers','2','--role','development_retention']
   cmd+=['--adaptation-run',str(path)] if path else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt'];run(cmd,'dev_'+name+'.log',True)
  return out
 # f0 exact membership/weights were already evaluated, copy only tiny evidence.
 import shutil
 f0=S/'evaluations/f0/development_retention';f0.mkdir(parents=True,exist_ok=True)
 for filename in ['per_frame.jsonl','summary.json','_SUCCESS']:
  src=P/'evaluations/f0/development_retention'/filename
  if src.exists() and not (f0/filename).exists():shutil.copy2(src,f0/filename)
 if not (f0/'per_frame.jsonl').exists():evaluate('f0')
 while True:
  plan=read(S/'protocol/training_plan.json');changed=False
  for item in plan['queue']:
   root=Path(item['run'])
   if not (root/'_SUCCESS').exists():continue
   evaluate(item['name'],root)
   for alpha in [.25,.5,.75]:
    name=item['name']+f'_a{int(alpha*100):03d}';out=S/'derived'/name
    if not (out/'_SUCCESS').exists():
     run(['python','-m','quality_estimation.blend_perception','--initial-checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--parent-run',str(root),'--extension-protocol',str(S/'protocol/interpolation_frozen.json'),'--output',str(out),'--alpha',str(alpha)],'derive_'+name+'.log')
    evaluate(name,out)
  # Separate assessor owns metrics/gates and appends the repeat pair atomically.
  subprocess.run([sys.executable,str(Path(__file__).with_name('assess_development.py')),'--study',str(S)],check=True)
  plan=read(S/'protocol/training_plan.json')
  if plan.get('queue_closed') and all((S/'evaluations'/x['name']/'development_retention/_SUCCESS').exists() and all((S/'evaluations'/(x['name']+f'_a{int(k*100):03d}')/'development_retention/_SUCCESS').exists() for k in [.25,.5,.75]) for x in plan['queue']):break
  time.sleep(15)
 (S/'reports/development_evaluation_complete.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'queue':len(plan['queue']),'fresh_confirmation_opened':False})+'\n')
if __name__=='__main__':main()
