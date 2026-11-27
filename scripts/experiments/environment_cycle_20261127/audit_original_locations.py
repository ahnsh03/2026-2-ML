"""Audit original capture metadata without inferring weather or simulator hour."""
from pathlib import Path
import argparse,json
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from study_data import read,dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--train-manifest',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--positions-output',type=Path);a=p.parse_args();runs=defaultdict(set)
 for line in a.train_manifest.read_text().splitlines():
  row=json.loads(line);runs[row['run_id']].add(int(row['frame_id']))
 all_xy=[];reports=[]
 for j,(run,frames) in enumerate(sorted(runs.items())):
  root=a.data_root/'datasets'/run;manifest={}
  with (root/'manifest.jsonl').open() as stream:
   for line in stream:
    r=json.loads(line);manifest[int(r['frame_id'])]=r['custom_name']
  states={};field_counts=defaultdict(int)
  with (root/'lineage/capture_manifest.jsonl').open() as stream:
   for line in stream:
    r=json.loads(line);state=r.get('state') or {};ego=state.get('ego') or {};pos=ego.get('position')
    if pos is not None:states[r['custom_name']]=[pos['x'],pos['y']]
    for field in ['weather','hour','environment','simulator_hour']:
     if field in r or field in state:field_counts[field]+=1
  xy=[states[manifest[i]] for i in sorted(frames) if i in manifest and manifest[i] in states];all_xy.extend(xy);reports.append(dict(run='Run'+str(j+1),original_train_poses=len(frames),matched_recorded_ego_positions=len(xy),explicit_environment_fields=dict(field_counts)))
 xy=np.array(all_xy)
 if a.positions_output:np.savez_compressed(a.positions_output,ego_xy=xy,train_manifest_sha256=sha(a.train_manifest))
 tree=cKDTree(xy);route_reports=[]
 for i,route in enumerate(read(a.plan)['routes']):
  distance=tree.query(np.array(route['points'])[:,:2])[0];route_reports.append(dict(route='ConfirmationRoad'+str(i+1),minimum_nearest_original_ego_m=float(distance.min()),median_nearest_original_ego_m=float(np.median(distance)),fraction_planned_points_within5m_of_original_ego=float((distance<=5).mean())))
 dump(a.output,dict(original_train_unique_pose_rows=sum(map(len,runs.values())),recorded_ego_pose_rows=len(xy),runs=reports,confirmation_original_training_proximity=route_reports,source_sha256=sha(__file__),train_manifest_sha256=sha(a.train_manifest),plan_sha256=sha(a.plan),scope='Original recorded ego positions are a location-overlap diagnostic, not camera extrinsic/exposure proof. Original weather/hour are not reconstructed from filenames or host timestamps. Fresh buffers from current quality/adaptation sources do not imply exclusion from original v6 training routes.'))
 print('ORIGINAL CAPTURE AUDIT',len(xy),'/',sum(map(len,runs.values())),route_reports,flush=True)
if __name__=='__main__':main()
