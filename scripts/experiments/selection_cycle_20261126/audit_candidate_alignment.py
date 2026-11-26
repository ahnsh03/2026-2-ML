"""Daytime content check for moving RGB/Semantic pairs; not exposure timestamp certification."""
from pathlib import Path
import argparse,json,collections,hashlib
import cv2,numpy as np
from quality_estimation.spatial_features import SpatialProfile
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;cv2.setNumThreads(1)
 rows=[json.loads(s) for s in (S/'protocol/all_sources_v2.jsonl').read_text().splitlines()];pool=[r for r in rows if r['adaptation_role']=='candidate_pool' and r['weather']==0 and r['hour']==13];chosen=[pool[i] for i in np.linspace(0,len(pool)-1,48,dtype=int)];groups=collections.defaultdict(list)
 for r in rows:groups[r['source_key']].append(r)
 roi=SpatialProfile.load().native;road=(roi>=4)&(roi<=7);records=[]
 def target(row):
  path=Path('/mnt')/row['semantic'];data=path.read_bytes()
  if hashlib.sha256(data).hexdigest()!=row['expected_sha256']['semantic']:raise ValueError('semantic changed')
  sem=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);return (np.all(sem==(255,255,255),axis=2)|np.all(sem==(0,255,255),axis=2)).astype(np.uint8)
 def boundary(mask):return (mask.astype(bool)&~cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool))&road
 for row in chosen:
  data=(Path('/mnt')/row['image']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=row['expected_sha256']['rgb']:raise ValueError('RGB changed')
  bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,50,150,L2gradient=True)>0;distance=cv2.distanceTransform((~edges).astype(np.uint8),cv2.DIST_L2,5);gt=target(row);own=boundary(gt)
  if own.sum()<50:continue
  cost=float(np.minimum(distance[own],12).mean());g=groups[row['source_key']];idx=next(i for i,r in enumerate(g) if r['sample_id']==row['sample_id']);neighbors=[]
  for i in [idx-1,idx+1]:
   if 0<=i<len(g):
    other=g[i];d=float(np.linalg.norm(np.array(other['xy'])-row['xy']))
    if .5<=d<=30:
     b=boundary(target(other))
     if b.sum()>=50:neighbors.append(dict(sample_id=other['sample_id'],distance_m=d,cost=float(np.minimum(distance[b],12).mean())))
  shifts=[]
  for dy in [-4,0,4]:
   for dx in [-4,0,4]:
    shifted=cv2.warpAffine(gt,np.array([[1,0,dx],[0,1,dy]],np.float32),(1280,720),flags=cv2.INTER_NEAREST);b=boundary(shifted)
    if b.any():shifts.append(dict(dx=dx,dy=dy,cost=float(np.minimum(distance[b],12).mean())))
  best=min(shifts,key=lambda s:s['cost']);records.append(dict(sample_id=row['sample_id'],own_cost_px=cost,neighbors=neighbors,best_translation=best,translation_gain=cost-best['cost']))
 paired=[r for r in records if r['neighbors']];report=dict(frames=len(records),paired_neighbor_frames=len(paired),median_own_cost_px=float(np.median([r['own_cost_px'] for r in records])),
    median_neighbor_cost_px=float(np.median([np.mean([n['cost'] for n in r['neighbors']]) for r in paired])) if paired else None,
    own_better_than_neighbor_fraction=float(np.mean([r['own_cost_px']<np.mean([n['cost'] for n in r['neighbors']]) for r in paired])) if paired else None,
    median_best_translation_gain_px=float(np.median([r['translation_gain'] for r in records])),records=records,
    scope='Sunny13 content-edge diagnostic only. Lane visibility/anti-aliasing affect this score. It is not hardware time synchronization or all-weather GT correctness proof.')
 dump(S/'reports/candidate_alignment_audit.json',report);print({k:v for k,v in report.items() if k!='records'},flush=True)
if __name__=='__main__':main()
