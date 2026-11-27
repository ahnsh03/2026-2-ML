"""Regenerate X/y and fit GP/RF for the selected perception version before test."""
from pathlib import Path
import argparse,json,subprocess,sys,time,os
from study_data import read,sha,dump


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();S=a.study
 while not (S/'protocol/provisional_f1_frozen.json').exists() or not (S/'reports/full_replay_control_complete.json').exists():time.sleep(15)
 choice=read(S/'protocol/provisional_f1_frozen.json');run=choice['selected']['run'];packs=S/'g1_packs';out=S/'g1_quality'
 env=os.environ.copy();env.update(PYTHONPATH=str(a.repo/'src')+':'+str(a.repo/'ros/src/data_collection/src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
 cmd=['docker','run','--rm','--gpus','all','--shm-size','1g','--name','ml-environment-f1-extract','-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v','/mnt/d:/mnt/d:ro','-w','/repo','-e','PYTHONPATH=/repo/src:/repo/ros/src/data_collection/src','-e','OPENBLAS_NUM_THREADS=2','-e','OMP_NUM_THREADS=2','ml-perception:rtx50-cu128','python','scripts/experiments/roi9_gp_20261125/extract.py','--index',str(S/'protocol/quality_index_frozen.json'),'--data-root','/mnt','--adaptation-run',run,'--output',str(packs),'--batch-size','4']
 with (S/'logs/g1_extract.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 if not (out/'_SUCCESS').exists():
  cmd=[sys.executable,str(Path(__file__).with_name('quality_screen.py')),'--study',str(S),'--parent',str(a.parent),'--packs',str(packs),'--index',str(S/'protocol/quality_index_frozen.json'),'--adaptation-run',run,'--output-name','g1_quality']
  with (S/'logs/g1_refit.log').open('a') as log:subprocess.run(cmd,env=env,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('G1 FIT COMPLETE',flush=True)
if __name__=='__main__':main()
