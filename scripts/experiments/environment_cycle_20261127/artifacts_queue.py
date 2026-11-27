"""After numeric assessment, export verified figures/aggregates without Git writes."""
from pathlib import Path
import argparse,os,subprocess,sys,time
from study_data import dump,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--public-output',type=Path,required=True);a=p.parse_args();S=a.study
 while not (S/'reports/final_evidence_complete.json').exists():time.sleep(15)
 if (S/'reports/artifacts_complete.json').exists():return
 env=os.environ.copy();env.update(PYTHONPATH=str(a.repo/'src')+':'+str(a.repo/'ros/src/data_collection/src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
 command=['docker','run','--rm','--gpus','all','--name','ml-environment-artifacts','-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v',str(a.repo.parent/'checkpoints')+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src:/repo/ros/src/data_collection/src','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','ml-perception:rtx50-cu128','python','scripts/experiments/environment_cycle_20261127/export_cases.py','--study',str(S),'--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt']
 with (S/'logs/export_final_cases.log').open('a') as log:subprocess.run(command,cwd=a.repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 for script,args in [('figures.py',['--study',str(S),'--parent',str(a.parent)]),('export_evidence.py',['--study',str(S),'--parent',str(a.parent),'--output',str(a.public_output)])]:
  with (S/'logs'/('artifacts_'+script.replace('.py','.log'))).open('a') as log:subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],cwd=a.repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 dump(S/'reports/artifacts_complete.json',dict(assessment_sha256=sha(S/'reports/fresh_confirmation_assessment.json'),figure_inventory_sha256=sha(S/'figures/inventory.json'),public_inventory_sha256=sha(a.public_output/'evidence_inventory.json'),source_sha256=sha(__file__),git_commit_or_push_performed=False))
 print('VERIFIED FIGURES AND AGGREGATES EXPORTED; documentation/review/Git handoff remains',flush=True)
if __name__=='__main__':main()
