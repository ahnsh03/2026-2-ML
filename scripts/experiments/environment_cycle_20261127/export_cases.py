"""Fixed post-assessment illustrations with verified GT and both perception maps."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import load_target_model,sha256_file
from quality_estimation.native import NativeSamples
from quality_estimation.perception_identity import resolve_quality_perception
from quality_estimation.diagnostic_targets import gt_review_flags
from study_data import read,dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();S=a.study;C=S/'confirmation'
 if not (S/'reports/fresh_confirmation_assessment.json').exists():raise ValueError('illustrations require completed frozen assessment')
 rows=read(C/'registry/index_frozen.json')['fresh_confirmation']['rows'];z0=np.load(C/'packs_f0/fresh_confirmation.npz');z1=np.load(C/'packs_f1/fresh_confirmation.npz');assert z0['ids'].tolist()==z1['ids'].tolist()==[r['sample_id'] for r in rows];q0=z0['y'];q1=z1['y'];gain=q1-q0;gt_support=(z0['counts'][:,:,0]+z0['counts'][:,:,2]).sum(1);valid=np.flatnonzero(np.isfinite(gain)&(gt_support>=50));hours=np.array([r['hour'] for r in rows]);chosen=[]
 for mask,maximize in [(hours>=20,True),(hours==7,True),(np.ones(len(rows),bool),False)]:
  eligible=[i for i in valid if mask[i] and i not in chosen]
  if eligible:chosen.append(max(eligible,key=lambda i:gain[i]) if maximize else min(eligible,key=lambda i:gain[i]))
 while len(chosen)<3:
  eligible=[i for i in valid if i not in chosen];chosen.append(eligible[len(eligible)//2])
 source=NativeSamples(Path('/mnt'),[rows[i] for i in chosen]);items=[source[i] for i in range(len(chosen))];torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;device=torch.device('cuda');inputs=torch.from_numpy(np.stack([r['image'] for r in items])).to(device)
 candidate=read(S/'protocol/provisional_f1_frozen.json')['selected']['run'];updated,identity,_=resolve_quality_perception(Path(candidate),None);pred=[]
 for path,weights in [(a.checkpoint,'ema'),(updated,'model')]:
  model=load_target_model(path,device,weights=weights)
  with torch.inference_mode():_,mark=model.model(inputs);pred.append(mark.argmax(1).cpu().numpy()[:,12:372])
  del model
 out=S/'figures';out.mkdir(exist_ok=True);case_dir=out/'case_components';case_dir.mkdir(exist_ok=True);fig,axes=plt.subplots(3,4,figsize=(16,8.8));records=[]
 palette=np.array([[32,38,45],[245,245,245],[255,210,35],[170,170,170]],np.uint8)
 for rownum,(i,item) in enumerate(zip(chosen,items)):
  rgb=cv2.resize(item['rgb'],(640,360),interpolation=cv2.INTER_AREA);gt=item['target'][12:372];mask=np.isin(gt,[1,2]);native_gt=palette[np.where(mask,gt,0)];images=[rgb,native_gt];actual=[]
  for prediction in pred:
   pm=np.isin(prediction[rownum],[1,2]);union=(pm|mask).sum();iou=float((pm&mask).sum()/union) if union else None;actual.append(iou);overlay=rgb.copy();overlay[pm]=np.rint(.55*overlay[pm]+.45*np.array([255,100,10])).astype(np.uint8);images.append(overlay)
  for original,recomputed in zip([q0[i],q1[i]],actual):
   if recomputed is None or abs(float(original)-recomputed)>.005:raise ValueError('illustration GT/model alignment differs from frozen evidence')
  titles=['RGB','GT white/yellow lanes' if mask.any() else 'GT: no lane pixels',f'v6 overlay / IoU{actual[0]:.3f}',f'Updated overlay / IoU{actual[1]:.3f}'];alias=chr(65+rownum);directory=case_dir/('case_'+alias);directory.mkdir(exist_ok=True)
  for j,(image,title) in enumerate(zip(images,titles)):
   axes[rownum,j].imshow(image);axes[rownum,j].set_title(title,fontsize=10);axes[rownum,j].axis('off');cv2.imwrite(str(directory/(['rgb','gt','v6_overlay','updated_overlay'][j]+'.png')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
  axes[rownum,0].text(0,-.07,f"Case{alias} / weather{rows[i]['weather']} / {rows[i]['hour']:02d}:00",transform=axes[rownum,0].transAxes,fontsize=10)
  records.append(dict(alias=alias,sample_id=rows[i]['sample_id'],native_lane_gt_pixels=int(mask.sum()),rgb_sha256=item['hashes']['rgb'],semantic_sha256=item['hashes']['semantic'],frozen_iou=[float(q0[i]),float(q1[i])],illustration_iou=actual))
 fig.suptitle('Post-assessment illustrations (GT >=50px): largest gains and worst change',fontsize=14);fig.tight_layout();fig.savefig(out/'08_paired_gt_cases.png',dpi=160);plt.close(fig);dump(C/'illustration_manifest_private.json',dict(cases=records,selection='Largest night/dawn gains and worst delta among defined GT>=50pixel rows, post-assessment only; not representative averages.',checkpoint_sha256=sha256_file(a.checkpoint),updated_identity=identity,source_sha256=sha(__file__)))
 print('VERIFIED GT ILLUSTRATIONS',len(records),flush=True)
if __name__=='__main__':main()
