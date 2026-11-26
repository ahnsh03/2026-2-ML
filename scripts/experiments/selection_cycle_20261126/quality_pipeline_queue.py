"""Regenerate/refit a provisional f1/g1 while continuation controls finish."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--python',type=Path,required=True);a=p.parse_args();S=a.study
 while not (S/'reports/development_evaluations_complete.json').exists():time.sleep(15)
 if (S/'protocol/interpolation_extension_frozen.json').exists():
  while not (S/'reports/primary_interpolation_complete.json').exists():time.sleep(15)
 compare=[str(a.python),'scripts/experiments/selection_cycle_20261126/compare_perception.py','--study',str(S),'--freeze-provisional']
 if not (S/'protocol/provisional_f1_frozen.json').exists():subprocess.run(compare,cwd=a.repo,check=True)
 provisional=json.loads((S/'protocol/provisional_f1_frozen.json').read_text());run=Path(provisional['selected']['run']);out=S/'g1_quality'
 dump(S/'QUALITY_WORK_IN_PROGRESS.json',dict(status='regenerating_f1_X_y',adaptation_run=str(run),updated_utc=datetime.now(timezone.utc).isoformat()))
 mounts=['-v',str(a.repo)+':/repo','-v','/mnt/e:/mnt/e','-v','/mnt/d:/mnt/d:ro','-w','/repo','-e','PYTHONPATH=/repo/src','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1']
 extract=['docker','run','--name','ml-cycle-f1-quality-packs','--rm','--gpus','all','--shm-size','1g',*mounts,'ml-perception:rtx50-cu128','python','scripts/experiments/roi9_gp_20261125/extract.py',
          '--index',str(S/'protocol/quality_index_frozen.json'),'--data-root','/mnt','--adaptation-run',str(run),'--output',str(out/'packs'),'--keys','quality_fit','development','calibration','--batch-size','4']
 with (S/'logs/g1_quality_packs.log').open('a') as log:subprocess.run(extract,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 dump(S/'QUALITY_WORK_IN_PROGRESS.json',dict(status='refitting_g1_GP_RF',adaptation_run=str(run),updated_utc=datetime.now(timezone.utc).isoformat()))
 refit=[str(a.python),'scripts/experiments/selection_cycle_20261126/refit_quality.py','--study',str(S),'--packs',str(out/'packs'),'--index',str(S/'protocol/quality_index_frozen.json'),
        '--parent-recipe',str(S.parent/'ml_roi9_gp_20261125/frozen_bundle_final/feature_recipe.json'),'--adaptation-run',str(run),'--output',str(out/'refit')]
 if not (out/'refit/_SUCCESS').exists():
  with (S/'logs/g1_quality_refit.log').open('a') as log:subprocess.run(refit,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 policy=[str(a.python),'scripts/experiments/selection_cycle_20261126/refit_policy.py','--study',str(S),'--refit',str(out/'refit'),'--packs',str(out/'packs'),'--output',str(out/'bundle')]
 if not (out/'bundle/bundle_manifest.json').exists():
  with (S/'logs/g1_quality_policy.log').open('a') as log:subprocess.run(policy,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 relative=[str(a.python),'scripts/experiments/selection_cycle_20261126/versioned_relative_study.py','--packs',str(out/'packs'),'--bundle',str(out/'bundle'),'--output',str(out/'relative_reference')]
 if not (out/'relative_reference/_SUCCESS').exists():
  with (S/'logs/g1_relative_reference.log').open('a') as log:subprocess.run(relative,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
 dump(S/'QUALITY_WORK_IN_PROGRESS.json',dict(status='refit_complete_heldout_pending',adaptation_run=str(run),bundle=str(out/'bundle'),updated_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
