"""Dev-only RGB/no-GT replay and deliberate model-version mismatch checks."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
from quality_estimation.environment_review import EnvironmentReview
from quality_estimation.perception_identity import baseline_identity
from quality_estimation.region_feature_views import feature_matrix
from study_data import read,dump,sha,load_sources,memberships

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--quality-dir',default='f0_quality');p.add_argument('--adaptation-run',type=Path);a=p.parse_args();S=a.study;V=S/'verification'/a.quality_dir;V.mkdir(parents=True,exist_ok=True);review=EnvironmentReview(S/a.quality_dir/'bundle');index=read(S/'protocol/quality_index_frozen.json');rows=[]
 for site in sorted({r['site'] for r in index['development']['rows'] if r['source_key'].startswith('drive_development_')}):
  group=[r for r in index['development']['rows'] if r['site']==site and r['source_key'].startswith('drive_development_')];rows.extend(group[i] for i in np.linspace(0,len(group)-1,16,dtype=int))
 assert len(rows)==48;manifest=V/'rgb_only.jsonl';manifest.write_text(''.join(json.dumps(dict(sample_id=r['sample_id'],image=r['image'],expected_sha256={'rgb':r['expected_sha256']['rgb']},view=r['view']),allow_nan=False)+'\n' for r in rows));mounts=['-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v','/mnt/d:/mnt/d:ro','-v',str(a.repo.parent/'checkpoints')+':/checkpoints:ro','-w','/repo','-e','PYTHONPATH=/repo/src','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1']
 for profile,device in [('rtx50-cu128','cuda'),('rtx40-cu118','cpu')]:
  out=V/profile
  if (out/'_SUCCESS').exists():continue
  cmd=['docker','run','--rm','--name','ml-environment-runtime',*(['--gpus','all'] if device=='cuda' else []),'--shm-size','1g',*mounts,'ml-perception:'+profile,'python','-m','quality_estimation.infer_versioned_review','--bundle',str(S/a.quality_dir/'bundle'),'--manifest',str(manifest),'--data-root','/mnt','--output',str(out),'--batch-size','4','--device',device,'--budget','8']
  cmd+=['--adaptation-run',str(a.adaptation_run)] if a.adaptation_run else ['--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt']
  with (S/'logs'/f'check_runtime_{a.quality_dir}_{profile}.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 gpu=np.load(V/'rtx50-cu128/features.npz');cpu=np.load(V/'rtx40-cu118/features.npz');assert gpu['ids'].tolist()==cpu['ids'].tolist();delta=float(np.max(abs(gpu['X']-cpu['X'])));g=[json.loads(x) for x in (V/'rtx50-cu128/predictions.jsonl').read_text().splitlines()];c=[json.loads(x) for x in (V/'rtx40-cu118/predictions.jsonl').read_text().splitlines()];pred_delta=max(abs(x['estimated_iou']-y['estimated_iou']) for x,y in zip(g,c));tag_equal=all(x['environment_cluster']==y['environment_cluster'] for x,y in zip(g,c));mismatch=dict(review.spec['perception_identity'],checkpoint_sha256='0'*64)
 try:EnvironmentReview(S/a.quality_dir/'bundle',perception_identity=mismatch)
 except ValueError:rejected=True
 else:rejected=False
 assert rejected and tag_equal and pred_delta<.01
 gpu_selected=[json.loads(x)['sample_id'] for x in (V/'rtx50-cu128/candidates.jsonl').read_text().splitlines()];cpu_selected=[json.loads(x)['sample_id'] for x in (V/'rtx40-cu118/candidates.jsonl').read_text().splitlines()]
 dump(V/'runtime_verification.json',dict(selection_order_equal=gpu_selected==cpu_selected,selection_overlap=len(set(gpu_selected)&set(cpu_selected)),rows=48,uses_gt=False,manifest_has_semantic_field=False,max_feature_profile_difference=delta,max_iou_prediction_profile_difference=pred_delta,environment_cluster_profile_equal=tag_equal,wrong_perception_identity_rejected=rejected,scope='40-profile CPU and50-profile GPU execution;40-series GPU hardware unavailable. Native CUDA batch arithmetic can alter near-tied hard output pixels.'))
 if a.adaptation_run:
  gt=V/'joined_gt'
  if not (gt/'_SUCCESS').exists():
   command=[sys.executable,'-m','quality_estimation.join_spatial_gt','--queue-dir',str(V/'rtx50-cu128'),'--packs',str(S/'g1_packs'),'--source-index',str(S/'protocol/quality_index_frozen.json'),'--all-keys','--data-root','/mnt','--output-dir',str(gt)]
   subprocess.run(command,check=True)
  joined=[json.loads(x) for x in (gt/'gt_review.jsonl').read_text().splitlines()]
  assert len(joined)==8 and all(not r['admission_approved'] and r['source_context']['adaptation_role']=='development' for r in joined)
 print('RUNTIME VERIFIED',a.quality_dir,delta,pred_delta,flush=True)
if __name__=='__main__':main()
