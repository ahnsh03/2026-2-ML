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
 gpu('original_exposure_audit',['python','scripts/experiments/environment_cycle_20261127/audit_replay_exposure.py','--study',str(S)])
 alpha=choice['selected']['alpha'];matched=[]
 for item in read(S/'protocol/training_plan.json')['queue']:
  root=Path(item['run']) if alpha==1. else S/'derived'/(item['name']+f'_a{int(alpha*100):03d}')
  matched.append(dict(name=item['name'],arm=item['arm'],seed=item['seed'],run=str(root),checkpoint_sha256=sha(root/'checkpoint.pt'),evaluation='f1' if str(root)==run else item['name']+'_common_alpha'))
 control=read(S/'reports/full_replay_control_complete.json');control_run=next(r['run'] for r in control['models'] if r['alpha']==alpha)
 matched.append(dict(name='ReplayFull_s271127',arm='ReplayFull',seed=271127,run=control_run,checkpoint_sha256=sha(Path(control_run)/'checkpoint.pt'),evaluation='replay_control'))
 frozen=dict(source_sha256=sha(__file__),study_design_sha256=sha(S/'protocol/study_design_frozen.json'),perception=choice,matched_neural_models=matched,matched_alpha=alpha,quality_selection_sha256=sha(S/'g1_quality/quality_selection_frozen.json'),review_manifest_sha256=sha(S/'g1_quality/bundle/bundle_manifest.json'),confirmation_index_sha256=sha(C/'registry/index_frozen.json'),opened_after_complete_freeze=True,replay_control_alpha=choice['selected']['alpha'],replay_control_rule='Predeclared first-seed full26001-only24k endpoint at the selected f1 coefficient; no control checkpoint/alpha selected on confirmation.')
 opening=S/'protocol/fresh_confirmation_opening_frozen.json'
 if opening.exists():
  if read(opening)!=frozen:raise ValueError('frozen perception/quality/policy or confirmation membership changed')
 else:dump(opening,frozen)
 if not (S/'verification/g1_quality/runtime_verification.json').exists():host('check_runtime.py',['--study',str(S),'--parent',str(a.parent),'--repo',str(a.repo),'--quality-dir','g1_quality','--adaptation-run',run],'verify_g1_runtime.log')
 if not (C/'alignment/summary.json').exists():
  command=[sys.executable,str(Path(__file__).with_name('alignment_check.py')),'--study',str(S)]
  with (S/'logs/fresh_alignment.log').open('a') as log:subprocess.run(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
 index=read(C/'registry/index_frozen.json');raw_manifest=C/'registry/evaluation_sources.jsonl';cached_manifest=C/'registry/evaluation_cached.jsonl'
 if not raw_manifest.exists():raw_manifest.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in index['fresh_confirmation']['rows']))
 if not cached_manifest.with_suffix('.receipt.json').exists():
  if cached_manifest.exists():cached_manifest.rename(cached_manifest.with_suffix('.incomplete.jsonl'))
  gpu('fresh_prepare_cache',['python','-m','quality_estimation.adaptation_cache','--manifest',str(raw_manifest),'--data-root','/mnt','--cache',str(a.parent/'cache'),'--output',str(cached_manifest),'--workers','4'])
 eval_freeze=dict(opening_sha256=sha(opening),raw_manifest_sha256=sha(raw_manifest),cached_manifest_sha256=sha(cached_manifest),cache_receipt_sha256=sha(cached_manifest.with_suffix('.receipt.json')))
 eval_frozen=C/'registry/evaluation_manifest_frozen.json'
 if eval_frozen.exists():
  if read(eval_frozen)!=eval_freeze:raise ValueError('fresh evaluation input changed')
 else:dump(eval_frozen,eval_freeze)
 # Compare every frozen neural method at the same coefficient, including the
 # second optimizer seeds and the original-only continuation control.
 for item in [dict(evaluation='f0',run=None),*matched]:
  output=C/'evaluations'/item['evaluation']
  if (output/'_SUCCESS').exists():continue
  command=['python','-m','quality_estimation.evaluate_cached_perception','--manifest',str(cached_manifest),'--cache',str(a.parent/'cache'),'--output',str(output),'--batch-size','4','--workers','2','--role','heldout_diagnostic']
  command+=['--adaptation-run',str(item['run'])] if item['run'] else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt']
  gpu('fresh_neural_'+item['evaluation'],command)
 for name,adaptation in [('f0',None),('f1',run)]:
  packs=C/('packs_'+name);cmd=['python','scripts/experiments/roi9_gp_20261125/extract.py','--index',str(C/'registry/index_frozen.json'),'--data-root','/mnt','--output',str(packs),'--batch-size','4']
  cmd+=['--adaptation-run',str(adaptation)] if adaptation else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt'];gpu('fresh_extract_'+name,cmd)
 host('assess_confirmation.py',['--study',str(S),'--parent',str(a.parent)],'assess_confirmation.log')
 (S/'reports/final_evidence_complete.json').write_text(json.dumps({'status':'complete','fresh_confirmation_after_freeze':True})+'\n');print('FINAL EVIDENCE COMPLETE',flush=True)
if __name__=='__main__':main()
