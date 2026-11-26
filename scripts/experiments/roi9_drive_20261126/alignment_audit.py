"""Audit native moving RGB/Semantic registration without perception predictions."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from quality_estimation.spatial_features import SpatialProfile

def edge_distance(rgb):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 edges=cv2.Canny(gray,30,90,L2gradient=True)>0
 return cv2.distanceTransform((~edges).astype(np.uint8),cv2.DIST_L2,3)
def boundary(semantic,roi):
 lane=np.all(semantic==[255,255,255],2)|np.all(semantic==[255,255,0],2)
 edge=lane^(cv2.erode(lane.astype(np.uint8),np.ones((3,3),np.uint8))>0)
 edge[:338]=False;edge[roi==8]=False;return edge
def cost(distance,edge,dx=0,dy=0):
 y,x=np.nonzero(edge);x=x+dx;y=y+dy;ok=(x>=0)&(x<distance.shape[1])&(y>=0)&(y<distance.shape[0])
 return float(np.minimum(distance[y[ok],x[ok]],5).mean()) if ok.any() else None
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 index=json.loads((a.registry/'index_full.json').read_text());profile=SpatialProfile.load();records=[];images=[];skipped=[]
 for key,pack in index.items():
  rows=[r for r in pack['rows'] if r['weather']==0 and r['hour']==13]
  if len(rows)<3:continue
  for i in range(1,len(rows)-1):
   row=rows[i];rgb=cv2.cvtColor(cv2.imread(str(a.root/row['image'])),cv2.COLOR_BGR2RGB);dist=edge_distance(rgb)
   sem=cv2.cvtColor(cv2.imread(str(a.root/row['semantic'])),cv2.COLOR_BGR2RGB);edge=boundary(sem,profile.native)
   if edge.sum()<50:
    skipped.append(dict(id=row['sample_id'],reason='fewer than50 native lane-boundary pixels'));continue
   shifts=[(cost(dist,edge,dx,dy),dx,dy) for dx in range(-3,4) for dy in range(-3,4)];best=min(shifts)
   neighbors=[]
   for j in [i-1,i+1]:
    other=cv2.cvtColor(cv2.imread(str(a.root/rows[j]['semantic'])),cv2.COLOR_BGR2RGB);other_edge=boundary(other,profile.native)
    if other_edge.sum()<50:break
    neighbors.append(cost(dist,other_edge))
   if len(neighbors)!=2:
    skipped.append(dict(id=row['sample_id'],reason='adjacent frame lacks50 native lane-boundary pixels'));continue
   own=cost(dist,edge);records.append(dict(id=row['sample_id'],group=row['group'],own_cost_px=own,neighbor_mean_cost_px=float(np.mean(neighbors)),best_shift_xy=[best[1],best[2]],best_cost_px=best[0],improvement_px=own-best[0],lane_boundary_pixels=int(edge.sum())))
   if i==len(rows)//2:images.append((row,rgb,sem,edge))
 summary=dict(method='RGB Canny-edge distance at exact native white/yellow GT boundaries; near/far road bands, Sunny13 only. Compare own GT with adjacent moving-frame GT and translation sensitivity. This is content evidence, not hardware exposure synchronization proof.',frames=len(records),groups=len({r['group'] for r in records}),median_own_cost_px=float(np.median([r['own_cost_px'] for r in records])),median_neighbor_cost_px=float(np.median([r['neighbor_mean_cost_px'] for r in records])),own_better_than_neighbors_fraction=float(np.mean([r['own_cost_px']<r['neighbor_mean_cost_px'] for r in records])),median_best_shift_linf_px=float(np.median([max(abs(x) for x in r['best_shift_xy']) for r in records])),maximum_translation_cost_improvement_px=max(r['improvement_px'] for r in records))
 summary.update(skipped=skipped,minimum_native_boundary_pixels=50)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(a.output/'frames.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(images),3,figsize=(16,9),squeeze=False)
 for i,(row,rgb,sem,edge) in enumerate(images):
  overlay=rgb.copy();overlay[edge]=[255,0,90]
  for j,img in enumerate([rgb,sem,overlay]):axes[i,j].imshow(img);axes[i,j].axis('off');axes[i,j].set_title(['Moving RGB','Native Semantic GT','Exact lane-boundary overlay'][j]+' / '+row['group'],fontsize=9)
  component=a.output/row['group'];component.mkdir();
  for name,img in [('rgb',rgb),('semantic',sem),('overlay',overlay)]:cv2.imwrite(str(component/(name+'.png')),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
 fig.tight_layout();fig.savefig(a.output/'alignment.png',dpi=160);fig.savefig(a.output/'alignment.svg');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
