"""Audit all immutable drive registries and cross-role spatial/pixel separation."""
from pathlib import Path
import argparse,json,hashlib
from datetime import datetime,timezone
import numpy as np
from scipy.spatial import cKDTree

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-followups',action='store_true');a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False)
 roles={};points={};pixels={};ids={};summary={};runs=[];raw_files={}
 cohorts=[(role,n/('registry_'+role+'_final'),n) for role in ['development','calibration','test']]
 if a.include_followups:
  cohorts.extend((role,n/role/'registry',n/role) for role in ['challenge','marked_support','confirmation'])
 for role,root,raw_root in cohorts:
  full=read(root/'index_full.json');thin=read(root/'index_thinned.json');audit=read(root/'audit.json');inventory=read(root/'inventory.json')
  assert not audit['rejected'] and not audit['missing_conditions'];rows=[r for v in full.values() for r in v['rows']];roles[role]=rows;points[role]=np.array([r['xy'] for r in rows]);pixels[role]={r['expected_pixel_sha256'] for r in rows};ids[role]={r['sample_id'] for r in rows}
  assert len(ids[role])==len(rows)==len(pixels[role])
  for item in inventory:
   for channel,record in item['files'].items():
    path=raw_root/record['path'];assert sha(path)==record['sha256'];raw_files[str(path.relative_to(n))]=record['sha256']
  summary[role]=dict(full_frames=len(rows),thinned_frames=sum(len(v['rows']) for v in thin.values()),road_segments=len(full),runs=len(audit['runs']),actual_minimum_prior_camera_m=audit['minimum_actual_prior_m'],source_binding_mismatches=sum(not r['source_binding_matches'] for r in audit['runs']),index_sha256=sha(root/'index_full.json'),thin_index_sha256=sha(root/'index_thinned.json'))
  for r in audit['runs']:
   run=raw_root/'runs'/r['run_id'];s=read(run/'summary.json');first=[json.loads(line) for line in (run/'manifest.jsonl').read_text().splitlines()];v=[m['first_seen']['Intensity']['state']['speed_mps'] for m in first if m['accepted']]
   control=[json.loads(line) for line in (run/'control.jsonl').read_text().splitlines()]
   xy=np.array([[c['state']['transform']['location'][axis] for axis in 'xy'] for c in control]);actual=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
   summary_run=dict(role=role,route=s['route_id'],weather=s['weather'],hour=s['hour'],frames=s['frames'],projected_route_progress_m=s['last_station_m']-s['start_station_m'],telemetry_xy_distance_m=actual,maximum_cte_m=s['maximum_cte_m'],maximum_speed_kph=s['maximum_speed_mps']*3.6,minimum_capture_speed_kph=min(v)*3.6,maximum_file_first_seen_latency_s=max(m['maximum_first_seen_latency_s'] for m in first if m['accepted']))
   runs.append(summary_run)
 separation={}
 names=list(roles)
 for i,left in enumerate(names):
  for right in names[i+1:]:
   assert not pixels[left]&pixels[right] and not ids[left]&ids[right]
   distance=float(cKDTree(points[left]).query(points[right])[0].min())
   separation[left+'__'+right]=dict(minimum_camera_xy_m=distance,duplicate_ids=0,duplicate_decoded_pixels=0)
 assert len(raw_files)==4*sum(v['full_frames'] for v in summary.values())
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),roles=summary,cross_role_separation=separation,raw_channels_verified=len(raw_files),total_frames=sum(v['full_frames'] for v in summary.values()),total_thinned=sum(v['thinned_frames'] for v in summary.values()),projected_route_progress_m=sum(r['projected_route_progress_m'] for r in runs),telemetry_xy_distance_m=sum(r['telemetry_xy_distance_m'] for r in runs),maximum_cte_m=max(r['maximum_cte_m'] for r in runs),maximum_speed_kph=max(r['maximum_speed_kph'] for r in runs),minimum_capture_speed_kph=min(r['minimum_capture_speed_kph'] for r in runs),maximum_file_first_seen_latency_s=max(r['maximum_file_first_seen_latency_s'] for r in runs),run_condition_records=runs,scope='Same-map separated short segments; vehicle telemetry at file first-seen is not camera exposure pose. Pixel uniqueness does not prove unseen perception-training geography.')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output/'raw_files_sha256.json').write_text(json.dumps(raw_files,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['run_condition_records','roles']},indent=2),flush=True)
if __name__=='__main__':main()
