"""Symmetric development-only post-training retention study, no new gradient work."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();S=a.study
 protocol=S/'protocol/interpolation_extension_frozen.json'
 if not protocol.exists():
  dump(protocol,dict(created_utc=datetime.now(timezone.utc).isoformat(),alphas=[.25,.5,.75],formula='theta_alpha=(1-alpha)*f0_EMA+alpha*trained_parameters; frozen buffers must agree',
       motivation='First completed Random endpoint improved dev but missed pre-existing retention limit. Additional development experiment, registered after that result and before other f1 development outcomes.',
       scope='Same alpha grid for all methods/seeds including continuation control; no heldout selection; no extra training steps. Gates unchanged.',
       primary_reference='https://github.com/mlfoundations/wise-ft',reference_scope='Weight interpolation concept only; different model/task/data, no CLIP reproduction or promised benefit.'))
 queue=[]
 for filename in ['training_queue_frozen.json','control_queue_v2_frozen.json']:
  while not (S/'protocol'/filename).exists():time.sleep(10)
  queue.extend(json.loads((S/'protocol'/filename).read_text())['queue'])
 derived=[]
 for item in queue:
  for alpha in [.25,.5,.75]:
   derived.append(dict(arm=item['arm'],seed=item['seed'],parent_run=item['run'],alpha=alpha,run=str(S/'derived'/f"{Path(item['run']).name}_a{int(alpha*100):03d}")))
 dump(S/'protocol/interpolation_queue_frozen.json',dict(queue=derived,created_utc=datetime.now(timezone.utc).isoformat()))
 mount=['-v',str(a.repo)+':/repo','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1']
 done=set()
 while len(done)<len(derived):
  for item in derived:
   out=Path(item['run']);name=out.name;destination=S/'evaluations'/name/'development_retention'
   if name in done or (destination/'_SUCCESS').exists():done.add(name);continue
   # Do not compete with the raw endpoint's primary evaluation.
   raw_eval=S/'evaluations'/Path(item['parent_run']).name/'development_retention/_SUCCESS'
   if not raw_eval.exists():continue
   while subprocess.check_output(['docker','ps','--filter','name=^/ml-selection-evaluation$','--format','{{.Names}}'],text=True).strip():time.sleep(2)
   if not (out/'_SUCCESS').exists():
    command=['docker','run','--name','ml-selection-derive','--rm',*mount,'ml-perception:rtx50-cu128','python','-m','quality_estimation.blend_perception',
         '--initial-checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--parent-run',item['parent_run'],
         '--extension-protocol',str(protocol),'--output',str(out),'--alpha',str(item['alpha'])]
    with (S/f'logs/derive_{name}.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
   command=['docker','run','--name','ml-selection-interpolation-eval','--rm','--gpus','all','--shm-size','1g',*mount,'ml-perception:rtx50-cu128','python','-m','quality_estimation.evaluate_cached_perception',
          '--manifest',str(S/'protocol/development_retention_eval.jsonl'),'--cache',str(S/'cache'),'--output',str(destination),'--batch-size','4','--workers','2','--role','development_retention','--adaptation-run',str(out)]
   with (S/f'logs/evaluate_{name}.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
   done.add(name);print('INTERPOLATION EVALUATED',name,flush=True)
   primary=[q for q in derived if q['arm']!='ReplayOnly']
   if all((S/'evaluations'/Path(q['run']).name/'development_retention/_SUCCESS').exists() for q in primary):
    if not (S/'reports/primary_interpolation_complete.json').exists():dump(S/'reports/primary_interpolation_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),models=len(primary)))
  time.sleep(10)
 dump(S/'reports/interpolation_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),models=len(done)))
if __name__=='__main__':main()
