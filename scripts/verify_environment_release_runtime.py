"""Run the handoff with only its own read-only volume and a fresh output volume."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
from verify_environment_release import verify
from verify_inference_release import digest


def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int);a=p.parse_args();release=a.release.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);checked=verify(release);source=['--adaptation-run','/release/models/perception'] if checked['perception_identity']['kind']=='adaptation' else ['--checkpoint','/release/models/perception/best_mean.pt'];results={}
 budget=min(4,checked['rgb_examples'])if a.budget is None else a.budget
 if not 0<=budget<=checked['rgb_examples']:raise ValueError('standalone review budget differs from buffer')
 for profile,device in [('rtx40-cu118','cpu'),('rtx50-cu128','cuda')]:
  target=out/profile
  if not (target/'_SUCCESS').exists():
   command=['docker','run','--rm','--name','ml-environment-release-verify',*(['--gpus','all'] if device=='cuda' else []),'--shm-size','1g','-v',str(release)+':/release:ro','-v',str(out)+':/results','-w','/release/code','-e','PYTHONPATH=/release/code/src:/release/code/ros/src/data_collection/src','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','ml-perception:'+profile,'python','-B','-m','quality_estimation.infer_versioned_review','--bundle','/release/models/review',*source,'--manifest','/release/examples/manifest.jsonl','--data-root','/release/examples/rgb','--output','/results/'+profile,'--device',device,'--batch-size','4','--budget',str(budget)]
   with (out/(profile+'.log')).open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  run=json.loads((target/'run.json').read_text())
  if run['budget']!=budget or run['bundle_manifest_sha256']!=digest(release/'models/review/bundle_manifest.json') or run['manifest_sha256']!=digest(release/'examples/manifest.jsonl') or run['perception_identity']!=checked['perception_identity']:raise ValueError('cached standalone replay belongs to another release')
  if (target/'_SUCCESS').read_text().strip()!=digest(target/'run.json'):raise ValueError('standalone replay incomplete')
  results[profile]=[json.loads(line) for line in (target/'predictions.jsonl').read_text().splitlines()]
 cpu,gpu=results.values();assert [r['sample_id'] for r in cpu]==[r['sample_id'] for r in gpu];delta=max(abs(x['estimated_iou']-y['estimated_iou']) for x,y in zip(cpu,gpu));assert delta<.01 and all(not r['uses_ground_truth'] for r in cpu+gpu);tag_equal=all(x['environment_cluster']==y['environment_cluster'] for x,y in zip(cpu,gpu));assert tag_equal
 picks=[[json.loads(line)['sample_id'] for line in (out/profile/'candidates.jsonl').read_text().splitlines()] for profile in results];f0=np.load(out/'rtx40-cu118/features.npz');f1=np.load(out/'rtx50-cu128/features.npz');feature_delta=float(np.max(abs(f0['X']-f1['X'])));verify(release)
 report=dict(status='verified',adoption_status=checked['adoption_status'],perception_identity=checked['perception_identity'],rows=checked['rgb_examples'],source_git=checked['source_git'],release_manifest_sha256=digest(release/'release_manifest.json'),max_iou_prediction_difference=delta,max_feature_difference=feature_delta,environment_cluster_equal=tag_equal,selection_order_equal=picks[0]==picks[1],selection_overlap=len(set(picks[0])&set(picks[1])),uses_ground_truth=False,original_workspace_mounted=False,original_data_volumes_mounted=False,upstream_objects_self_contained=True,actual_40_gpu_tested=False,actual_50_gpu_tested=True,scope='Only the release volume and a fresh output directory mounted. File/context verification repeated after both runs. Does not imply offline candidate adoption or live ROS acceptance.')
 (out/'runtime_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
