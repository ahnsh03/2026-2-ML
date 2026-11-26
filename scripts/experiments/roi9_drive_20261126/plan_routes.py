"""Plan short continuous drives using geometry only, before opening new labels."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['map','prior-index','membership','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--exclude-plan',type=Path,action='append',default=[])
 p.add_argument('--routes',type=int,default=9);p.add_argument('--length',type=float,default=32)
 p.add_argument('--prior-clearance',type=float,default=25);p.add_argument('--training-clearance',type=float,default=30)
 p.add_argument('--between-clearance',type=float,default=30)
 a=p.parse_args(); assert a.routes%3==0
 rows=[r for v in json.loads(a.prior_index.read_text()).values() for r in v['rows']]
 members={r['sample_id'] for r in json.loads(a.membership.read_text())['rows']}
 prior=cKDTree([r['xy'] for r in rows]); train=cKDTree([r['xy'] for r in rows if r['sample_id'] in members])
 excluded={r['requested_link'] for path in a.exclude_plan for r in json.loads(path.read_text()).get('poses',[]) if 'requested_link' in r}
 candidates=[]
 for link in json.loads(a.map.read_text()):
  if link['idx'] in excluded or link.get('lazy_init') or link.get('opp_traffic'):continue
  xyz=np.asarray(link['points']); s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1))]
  if s[-1]<a.length+10:continue
  for center in np.arange(5+a.length/2,s[-1]-5-a.length/2,8):
   station=np.arange(center-a.length/2,center+a.length/2+.01,.5)
   pts=np.column_stack([np.interp(station,s,xyz[:,i]) for i in range(3)])
   pd=float(prior.query(pts[:,:2])[0].min());td=float(train.query(pts[:,:2])[0].min())
   if pd<a.prior_clearance+.3 or td<a.training_clearance+.3:continue
   headings=np.unwrap(np.arctan2(np.diff(pts[:,1]),np.diff(pts[:,0])))
   turn=float(np.rad2deg(np.ptp(headings)))
   grade=float(np.max(np.abs(np.diff(pts[:,2]))/.5))
   if turn>45 or grade>.07:continue
   candidates.append(dict(link=link['idx'],station_start=float(station[0]),station_end=float(station[-1]),points=pts.tolist(),minimum_prior_m=pd,minimum_training_m=td,turn_degrees=turn))
 chosen=[]
 while len(chosen)<a.routes:
  eligible=[r for r in candidates if r['link'] not in {c['link'] for c in chosen} and all(cKDTree(np.asarray(c['points'])[:,:2]).query(np.asarray(r['points'])[:,:2])[0].min()>=a.between_clearance for c in chosen)]
  if not eligible:break
  # Geometric spread and curve/straight balance, with no RGB or target inspection.
  def rank(r):
   spread=min((np.linalg.norm(np.asarray(r['points'])[len(r['points'])//2,:2]-np.asarray(c['points'])[len(c['points'])//2,:2]) for c in chosen),default=200)
   return (min(spread,250)+min(r['minimum_prior_m'],100),r['link'],r['station_start'])
  chosen.append(max(eligible,key=rank))
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(Path(__file__)),map_sha256=digest(a.map),prior_index_sha256=digest(a.prior_index),training_membership_sha256=digest(a.membership),excluded_plan_sha256=[digest(p) for p in a.exclude_plan],excluded_links=sorted(excluded),criteria=dict(length_m=a.length,prior_camera_clearance_m=a.prior_clearance,training_camera_clearance_m=a.training_clearance,between_drive_clearance_m=a.between_clearance,role='Geometry-only selection. New short link segments on the same map; prior complete link clearance is not claimed.'),candidate_count=len(candidates),requested_routes=a.routes,routes=[])
 roles=['development','calibration','test']
 for i,r in enumerate(chosen):
  r.update(route_id=f'drive26_r{i:02d}',role=roles[i%3],conditions=[[w,h] for w in [0,1,2,4,5] for h in [8,13,20]],target_speed_kph=4.0)
  pts=np.asarray(r['points']);delta=pts[4]-pts[0]
  r['start_pose']=dict(location=dict(zip('xyz',[*pts[0,:2],pts[0,2]+.35])),rotation=dict(x=0,y=0,z=float(np.rad2deg(np.arctan2(delta[1],delta[0])))))
  result['routes'].append(r)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(candidates=len(candidates),chosen=len(chosen),routes=[{k:r[k] for k in ['route_id','role','link','minimum_prior_m','minimum_training_m','turn_degrees']} for r in chosen]),indent=2))
 if len(chosen)!=a.routes:raise SystemExit('Insufficient eligible routes: inspect geometry before changing explicit criteria')
if __name__=='__main__':main()
