"""Open fresh confirmation only after f1, g1 and review policy are frozen."""
from pathlib import Path
import argparse,json,subprocess,sys,time,os
from study_data import read,dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();S=a.study;C=S/'confirmation';env=os.environ.copy();env.update(PYTHONPATH=str(a.repo/'src')+':'+str(a.repo/'ros/src/data_collection/src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 while not (S/'g1_quality/_SUCCESS').exists() or not (C/'registry/index_frozen.json').exists():time.sleep(15)
 choice=read(S/'protocol/provisional_f1_frozen.json');run=choice['selected']['run'];mount=['-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v','/mnt/d:/mnt/d:ro','-v',str(a.repo.parent/'checkpoints')+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src:/repo/ros/src/data_collection/src','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2']
 def host(script,args,log):
  with (S/'logs'/log).open('a') as f:subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],env=env,cwd=a.repo,stdout=f,stderr=subprocess.STDOUT,check=True)
 def gpu(name,command):
  with (S/'logs'/(name+'.log')).open('a') as f:subprocess.run(['docker','run','--rm','--gpus','all','--shm-size','1g','--name','ml-environment-final-eval',*mount,'ml-perception:rtx50-cu128',*command],stdout=f,stderr=subprocess.STDOUT,check=True)
 if not (S/'g1_quality/review_policy_frozen.json').exists():host('refit_policy.py',['--study',str(S),'--parent',str(a.parent)],'refit_g1_policy.log')
 if not (S/'g1_quality/bundle/bundle_manifest.json').exists():host('export_review.py',['--study',str(S),'--parent',str(a.parent),'--quality-dir','g1_quality','--output',str(S/'g1_quality/bundle')],'export_g1_bundle.log')
 frozen=dict(perception=choice,quality_selection_sha256=sha(S/'g1_quality/quality_selection_frozen.json'),review_manifest_sha256=sha(S/'g1_quality/bundle/bundle_manifest.json'),confirmation_index_sha256=sha(C/'registry/index_frozen.json'),opened_after_complete_freeze=True,replay_control_alpha=choice['selected']['alpha'],replay_control_rule='Predeclared first-seed full26001-only24k endpoint at the selected f1 coefficient; no control checkpoint/alpha selected on confirmation.')
 dump(S/'protocol/fresh_confirmation_opening_frozen.json',frozen)
 if not (C/'alignment/summary.json').exists():
  command=[sys.executable,str(Path(__file__).with_name('alignment_check.py')),'--study',str(S)]
  with (S/'logs/fresh_alignment.log').open('a') as log:subprocess.run(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
 for name,adaptation in [('f0',None),('f1',run)]:
  packs=C/('packs_'+name);cmd=['python','scripts/experiments/roi9_gp_20261125/extract.py','--index',str(C/'registry/index_frozen.json'),'--data-root','/mnt','--output',str(packs),'--batch-size','4']
  cmd+=['--adaptation-run',str(adaptation)] if adaptation else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt'];gpu('fresh_extract_'+name,cmd)
 # Native supervised two-head evaluation uses the same thinned input/cache for
 # f0/f1, and retains all rows regardless of GT lane-pixel support.
 manifest=C/'registry/sources.jsonl';manifest.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in read(C/'registry/index_frozen.json')['fresh_confirmation']['rows']))
 cached=C/'registry/cached.jsonl'
 if not cached.exists():gpu('fresh_cache',['python','-m','quality_estimation.adaptation_cache','--manifest',str(manifest),'--data-root','/mnt','--cache',str(a.parent/'cache'),'--output',str(cached),'--workers','8'])
 for name,adaptation in [('f0',None),('f1',run)]:
  out=C/'evaluations'/name
  if (out/'_SUCCESS').exists():continue
  cmd=['python','-m','quality_estimation.evaluate_cached_perception','--manifest',str(cached),'--cache',str(a.parent/'cache'),'--output',str(out),'--batch-size','4','--workers','2','--role','heldout_diagnostic'];cmd+=['--adaptation-run',str(adaptation)] if adaptation else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt'];gpu('fresh_perception_'+name,cmd)
 host('check_runtime.py',['--study',str(S),'--parent',str(a.parent),'--repo',str(a.repo),'--quality-dir','g1_quality','--adaptation-run',run],'check_runtime_g1.log')
 while not (S/'reports/full_replay_control_complete.json').exists():time.sleep(15)
 control=read(S/'reports/full_replay_control_complete.json');alpha=choice['selected']['alpha'];control_run=next(r['run'] for r in control['models'] if r['alpha']==alpha);control_out=C/'evaluations/replay_control'
 if not (control_out/'_SUCCESS').exists():
  gpu('fresh_replay_control',['python','-m','quality_estimation.evaluate_cached_perception','--manifest',str(cached),'--cache',str(a.parent/'cache'),'--output',str(control_out),'--batch-size','4','--workers','2','--role','heldout_diagnostic','--adaptation-run',control_run])
 host('assess_confirmation.py',['--study',str(S),'--parent',str(a.parent)],'assess_confirmation.log')
 (S/'reports/final_evidence_complete.json').write_text(json.dumps({'status':'complete','fresh_confirmation_after_freeze':True})+'\n');print('FINAL EVIDENCE COMPLETE',flush=True)
if __name__=='__main__':main()
