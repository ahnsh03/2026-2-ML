"""Independent RGB/GT registration check on fresh Sunny13 moving frames."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
from study_data import read,dump,sha
sys.path.insert(0,str(Path(__file__).parents[1]/'roi9_drive_20261126'))
from alignment_audit import edge_distance,boundary,cost
from quality_estimation.spatial_features import SpatialProfile

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);S=p.parse_args().study;C=S/'confirmation';out=C/'alignment';out.mkdir(exist_ok=False);index=read(C/'registry/alignment/index_full.json');profile=SpatialProfile.load();records=[];skips=[];examples=[]
 for key,part in index.items():
  rows=[r for r in part['rows'] if r['weather']==0 and r['hour']==13]
  for i in range(1,len(rows)-1):
   row=rows[i];rgb=cv2.cvtColor(cv2.imread(str(Path('/mnt')/row['image'])),cv2.COLOR_BGR2RGB);sem=cv2.cvtColor(cv2.imread(str(Path('/mnt')/row['semantic'])),cv2.COLOR_BGR2RGB);dist=edge_distance(rgb);edge=boundary(sem,profile.native)
   if edge.sum()<50:skips.append(dict(sample_id=row['sample_id'],reason='Insufficient visible lane boundary for RGB-edge registration audit'));continue
   neighbors=[]
   for j in [i-1,i+1]:
    other=cv2.cvtColor(cv2.imread(str(Path('/mnt')/rows[j]['semantic'])),cv2.COLOR_BGR2RGB);e=boundary(other,profile.native)
    if e.sum()<50:break
    neighbors.append(cost(dist,e))
   if len(neighbors)!=2:continue
   shifts=[(cost(dist,edge,dx,dy),dx,dy) for dx in range(-3,4) for dy in range(-3,4)];best=min(shifts);own=cost(dist,edge);records.append(dict(site=row['site'],sample_id=row['sample_id'],own_cost_px=own,neighbor_mean_cost_px=float(np.mean(neighbors)),best_shift_xy=[best[1],best[2]],shift_improvement_px=own-best[0]))
   if i==len(rows)//2:
    overlay=rgb.copy();overlay[edge]=[255,0,90];examples.append((row['site'],rgb,sem,overlay))
 summary=dict(frames=len(records),roads=len({r['site'] for r in records}),own_better_than_neighbor_fraction=float(np.mean([r['own_cost_px']<r['neighbor_mean_cost_px'] for r in records])) if records else None,median_own_cost_px=float(np.median([r['own_cost_px'] for r in records])) if records else None,median_neighbor_cost_px=float(np.median([r['neighbor_mean_cost_px'] for r in records])) if records else None,median_best_shift_linf_px=float(np.median([max(map(abs,r['best_shift_xy'])) for r in records])) if records else None,skipped=len(skips),scope='Sunny13 native white/yellow GT boundary compared to RGB edges, adjacent-frame GT and +-3pixel translations. Content registration evidence, not camera exposure synchronization proof.',source_sha256=sha(__file__))
 dump(out/'summary.json',summary);dump(out/'frames.json',records)
 if examples:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(len(examples),3,figsize=(15,3.2*len(examples)),squeeze=False)
  for i,(site,rgb,sem,overlay) in enumerate(examples):
   for j,img in enumerate([rgb,sem,overlay]):axes[i,j].imshow(img);axes[i,j].axis('off');axes[i,j].set_title(['RGB','Native Semantic GT','GT boundary on RGB'][j]+' / '+site)
  fig.tight_layout();fig.savefig(out/'alignment.png',dpi=180);plt.close(fig)
 print('ALIGNMENT CHECK',summary,flush=True)
if __name__=='__main__':main()
