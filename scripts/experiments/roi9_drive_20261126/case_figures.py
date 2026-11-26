"""Render each moving RGB with its own hash-bound native GT and prediction."""
from pathlib import Path
import argparse,json,hashlib
import cv2,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','data-root','replay','pairs','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.study
 index=read(n/'registry_test_final/index_thinned.json');rows={r['sample_id']:r for v in index.values() for r in v['rows']}
 pred={r['sample_id']:r for r in map(json.loads,(a.replay/'predictions.jsonl').read_text().splitlines())}
 with np.load(n/'prospective_test_v1/predictions.npz',allow_pickle=False) as z:truth={sid:q for sid,q in zip(z['ids'],z['truth'])}
 records=[]
 for pair in read(a.pairs):
  fig,axes=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(left=.015,right=.985,bottom=.08,top=.88,wspace=.02,hspace=.17)
  for j,key in enumerate(['day','night']):
   sid=pair[key];row=rows[sid];r=pred[sid];rgbfile=a.data_root/row['image'];gtfile=a.data_root/row['semantic']
   assert sha(rgbfile)==row['expected_sha256']['rgb'] and sha(gtfile)==row['expected_sha256']['semantic']
   rgb=np.array(Image.open(rgbfile).convert('RGB'));semantic=np.array(Image.open(gtfile).convert('RGB'));lane=np.all(semantic==[255,255,255],2)|np.all(semantic==[255,255,0],2)
   with np.load(a.replay/r['heads_file'],allow_pickle=False) as z:mark=z['marking_class'][12:372]
   mask=cv2.resize(np.isin(mark,[1,2]).astype(np.uint8),(1280,720),interpolation=cv2.INTER_NEAREST)>0
   target=np.zeros_like(rgb);target[lane]=semantic[lane];overlay=rgb.copy();overlay[mask]=(overlay[mask]*.35+np.array([0,255,80])*.65).astype(np.uint8)
   root=a.output/sid;root.mkdir();Image.fromarray(rgb).save(root/'rgb.png');Image.fromarray(semantic).save(root/'semantic_native.png');Image.fromarray(target).save(root/'lane_gt_native.png');Image.fromarray(overlay).save(root/'prediction_overlay.png')
   for k,(image,title) in enumerate(zip([rgb,target,overlay],['RGB','Own-frame native lane GT','Fixed v6 prediction'])):
    axes[j,k].imshow(image);axes[j,k].axis('off');axes[j,k].set_title(f'{key.title()} | {title}',fontsize=12)
   q=truth[sid];estimate=r['raw_mean'];axes[j,2].text(.01,-.06,f'IoU GT {q[0]:.3f} / GP {estimate[0]:.3f}; near/far recall GT {q[1]:.2f}/{q[2]:.2f}',transform=axes[j,2].transAxes,fontsize=10)
   records.append(dict(id=sid,source_rgb_sha256=sha(rgbfile),source_gt_sha256=sha(gtfile),heads_sha256=sha(a.replay/r['heads_file']),true_targets=q.tolist(),predicted_raw_targets=estimate))
  fig.suptitle(pair['site']+' | nearby-position day/night moving frames',fontsize=20)
  fig.text(.02,.025,f"Recorded camera XY difference: {pair['position_difference_m']:.3f} m. Native components: 1280 x 720. Metrics: 640 x 360 evaluation grid.\nEach row has its own GT; conditions are controlled on a new segment. This pair is illustrative, not a general night-failure definition.",fontsize=10)
  fig.savefig(a.output/(pair['site']+'_day_night.png'),dpi=160);fig.savefig(a.output/(pair['site']+'_day_night.svg'));plt.close(fig)
 (a.output/'case_provenance.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n');print('hash-bound day/night cases',len(records),flush=True)
if __name__=='__main__':main()
