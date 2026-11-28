"""Check content registration within moving epochs, never across route boundaries."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import sys
import cv2
import numpy as np
from common.bridge import sha256_file
from quality_estimation.image_io import read_image,color_channels
from quality_estimation.spatial_features import SpatialProfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'roi9_drive_20261126'))
from alignment_audit import edge_distance,boundary,cost


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['index','data-root','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    index=json.loads(a.index.read_text());groups=defaultdict(list);profile=SpatialProfile.load();records=[];skips=[];examples=[]
    def image(row,field,label):
        value,digest=read_image(a.data_root,row[field],cv2.IMREAD_UNCHANGED)
        if digest!=row['expected_sha256'][label]:raise ValueError('alignment source bytes changed')
        return cv2.cvtColor(color_channels(value,label),cv2.COLOR_BGR2RGB)
    for part in index.values():
        for row in part['rows']:
            if row['weather']==0 and row['hour']==13:groups[row['parent_session_id']].append(row)
    for group,rows in groups.items():
        for i in range(1,len(rows)-1):
            row=rows[i];rgb=image(row,'image','rgb');semantic=image(row,'semantic','semantic');edge=boundary(semantic,profile.native)
            if edge.sum()<50:skips.append(dict(sample_id=row['sample_id'],reason='insufficient_native_boundary'));continue
            distance=edge_distance(rgb);neighbors=[]
            for j in [i-1,i+1]:
                delta=float(np.linalg.norm(np.asarray(rows[j]['xy'])-row['xy']))
                if not .1<=delta<=5:break
                other=boundary(image(rows[j],'semantic','semantic'),profile.native)
                if other.sum()<50:break
                neighbors.append(cost(distance,other))
            if len(neighbors)!=2:skips.append(dict(sample_id=row['sample_id'],reason='insufficient_same_epoch_neighbors'));continue
            own=cost(distance,edge);best=min((cost(distance,edge,dx,dy),dx,dy)for dx in range(-3,4)for dy in range(-3,4))
            records.append(dict(sample_id=row['sample_id'],site=row['site'],epoch=group,own_cost_px=own,neighbor_mean_cost_px=float(np.mean(neighbors)),best_shift_xy=[best[1],best[2]],translation_gain_px=own-best[0],boundary_pixels=int(edge.sum())))
            if i==len(rows)//2:
                overlay=rgb.copy();overlay[edge]=[255,0,90];examples.append((row,rgb,semantic,overlay))
    paired=[r['own_cost_px']<r['neighbor_mean_cost_px']for r in records]
    report=dict(frames=len(records),supported_segments=len({r['site']for r in records}),
        own_better_than_neighbor_fraction=float(np.mean(paired))if paired else None,
        median_own_cost_px=float(np.median([r['own_cost_px']for r in records]))if records else None,
        median_neighbor_cost_px=float(np.median([r['neighbor_mean_cost_px']for r in records]))if records else None,
        median_best_shift_linf_px=float(np.median([max(map(abs,r['best_shift_xy']))for r in records]))if records else None,
        records=records,skipped=skips,index_sha256=sha256_file(a.index),source_sha256=sha256_file(Path(__file__)),
        scope='Sunny13 moving content-edge evidence only. Same epoch and <=5m neighbors; white/yellow GT, no perception predictions. Does not certify exposure synchronization, visibility or all-weather GT correctness.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if examples:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(len(examples),3,figsize=(15,3*len(examples)),squeeze=False,layout='constrained')
        for i,(row,rgb,semantic,overlay)in enumerate(examples):
            for j,value in enumerate([rgb,semantic,overlay]):
                axes[i,j].imshow(value);axes[i,j].axis('off');axes[i,j].set_title(['RGB','Native Semantic GT','Exact GT boundary on RGB'][j]+' / '+row['site'],fontsize=10)
        fig.savefig(a.output/'alignment.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('ALIGNMENT AUDIT',len(records),report['supported_segments'],report['own_better_than_neighbor_fraction'],flush=True)


if __name__=='__main__':main()
