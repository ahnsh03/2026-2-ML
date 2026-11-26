"""Check RGB replay equivalence/compatibility and attach GT after queue freezing."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,numpy as np
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.join_spatial_gt import join_candidates
from feature_study import load,read,save,sha

def lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','output']:p.add_argument('--'+name,type=Path,required=True)
 for name in ['registry','packs','reference','replay','cpu-replay','queue','data-root']:p.add_argument('--'+name,type=Path)
 a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False)
 registry=a.registry or n/'registry_test_final';packs=a.packs or n/'packs_test';reference_path=a.reference or n/'prospective_test_v1/predictions.npz';replay_path=a.replay or n/'rgb_replay50_v1';cpu_path=a.cpu_replay or n/'rgb_replay40_cpu8_v1';queue_path=a.queue or n/'rgb_review_queue_v1';data_root=a.data_root or n.parent
 index=read(registry/'index_full.json');data,schema=load(packs,list(index));lookup={sid:i for i,sid in enumerate(data['ids'])}
 with np.load(replay_path/'features.npz',allow_pickle=False) as z:
  ids=z['ids'];columns=z['ordered_features'].tolist();actual=z['X'];ix=[lookup[sid] for sid in ids];expected=feature_matrix(data,schema,columns)[ix]
 replay=lines(replay_path/'predictions.jsonl');byid={r['sample_id']:r for r in replay}
 with np.load(reference_path,allow_pickle=False) as z:
  reference={sid:(m,c) for sid,m,c in zip(z['ids'],z['raw_mean'],z['raw_covariance'])}
 raw_error=max(float(np.max(abs(np.asarray(r['raw_mean'])-reference[r['sample_id']][0]))) for r in replay)
 covariance_error=max(float(np.max(abs(np.asarray(r['raw_covariance'])-reference[r['sample_id']][1]))) for r in replay)
 cpu=lines(cpu_path/'predictions.jsonl');mean_error=max(float(np.max(abs(np.asarray(r['raw_mean'])-byid[r['sample_id']]['raw_mean']))) for r in cpu)
 spatial_error=max(float(np.max(abs(np.asarray(r['spatial'])-byid[r['sample_id']]['spatial']))) for r in cpu)
 with np.load(cpu_path/'features.npz',allow_pickle=False) as z:
  assert z['ordered_features'].tolist()==columns
  rlookup={sid:i for i,sid in enumerate(ids)};feature_error=float(np.max(abs(z['X']-actual[[rlookup[sid] for sid in z['ids']]])))
 protocol=read(n/'runtime_compatibility_protocol.json');passed=mean_error<=protocol['maximum_primary_point_mean_difference'] and spatial_error<=protocol['maximum_spatial_probability_difference']
 queue=read(queue_path/'run.json');assert sha(queue_path/'candidates.jsonl')==queue['candidates_sha256'];candidates=lines(queue_path/'candidates.jsonl')
 source_rows=[r for pack in index.values() for r in pack['rows']];source_lookup={r['sample_id']:r for r in source_rows}
 for candidate in candidates:
  source=source_lookup[candidate['sample_id']]
  assert sha(data_root/source['image'])==source['expected_sha256']['rgb'] and sha(data_root/source['semantic'])==source['expected_sha256']['semantic']
 receipts=[read(packs/(key+'.json')) for key in index]
 assert all(r['index_sha256']==sha(registry/'index_full.json') for r in receipts)
 receipt=receipts[0]
 joined=join_candidates(candidates,data,source_rows,checkpoint_sha256=receipt['checkpoint_sha256'],profile=schema['profile'],regions_sha256=schema['regions_sha256'],perception_identity=receipt['perception_identity'],perception_weights=receipt['perception_weights'])
 assert len(joined)==len(candidates) and not any(r['admission_approved'] for r in joined)
 (a.output/'candidates_with_gt.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in joined))
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),rgb_replay_rows=len(replay),ordered_feature_union=len(columns),replay_cached_feature_maximum_difference=float(np.max(abs(actual-expected))),replay_raw_GP_mean_maximum_difference=raw_error,replay_raw_covariance_maximum_difference=covariance_error,profile40_cpu_rows=len(cpu),profile40_cpu_mean_maximum_difference=mean_error,profile40_cpu_spatial_probability_maximum_difference=spatial_error,profile40_cpu_feature_maximum_difference=feature_error,profile_compatibility_passed=passed,IoU07_prediction_switches=sum((r['raw_mean'][0]<.7)!=(byid[r['sample_id']]['raw_mean'][0]<.7) for r in cpu),GT_joined_candidates=len(joined),admission_approved=0,queue_actual_budget_fraction=queue['actual_budget_fraction'],queue_sha256=sha(n/'rgb_review_queue_v1/run.json'),scope='GT was attached only after RGB-only inference and queue freezing. Session/window queue budget differs from the main per-segment pooled comparison; no40GPU, live ROS or perception-retraining claim.')
 result['queue_sha256']=sha(queue_path/'run.json')
 result['cached_vs_RGB_IoU07_prediction_switches']=int(sum((r['raw_mean'][0]<.7)!=(reference[r['sample_id']][0][0]<.7) for r in replay))
 save(a.output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
 if not passed:raise SystemExit('Runtime differences exceed predeclared tolerance; artifacts remain for inspection')
if __name__=='__main__':main()
