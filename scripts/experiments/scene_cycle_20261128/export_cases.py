"""Export RGB/true-GT/prediction/FP-FN cases at a consistent16:9 aspect ratio."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import load_target_model,sha256_file
from quality_estimation.adaptation_cache import CachedAdaptationSamples
from quality_estimation.perception_identity import resolve_quality_perception
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.review_targets import support_aware_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['pack','manifest','cache','adaptation-run','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--case-index',type=Path,help='Reuse baseline case IDs for a paired before/after illustration')
    p.add_argument('--case-label-prefix',default='baseline_',help='Display label prefix for explicitly fixed cases; use fresh_ for new confirmation illustrations')
    p.add_argument('--case-scope',default='Opened Dev case illustration',help='Describe the cohort used for post-audit illustrations; never changes selection or adoption')
    a=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    checkpoint,identity,_=resolve_quality_perception(a.adaptation_run,None)
    receipt=json.loads(a.pack.with_suffix('.json').read_text())
    validate_pack_perception(receipt,identity,'model_state_dict')
    if receipt['sha256']!=sha256_file(a.pack):raise ValueError('case pack changed')
    with np.load(a.pack)as z:data={k:z[k]for k in z.files}
    rows=[json.loads(line)for line in a.manifest.read_text().splitlines()];lookup={r['sample_id']:r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('duplicate case source IDs')
    for key,pixel in zip(data['ids'],data['pixel_sha256']):
        if str(key)not in lookup or lookup[str(key)]['pixel_sha256']!=str(pixel):raise ValueError('case source RGB differs from feature pack')
    a.output.mkdir(parents=True,exist_ok=False)
    t=support_aware_targets(data['counts']);used=set();chosen=[]
    definitions=[('near_fn',t['near']['false_negative'],data['counts'][:,6:8,2].sum(1)),
        ('near_fp',t['near']['false_positive'],data['counts'][:,6:8,1].sum(1)),
        ('near_retained_far_error',(~t['near']['error'])&t['far']['error']&(t['near']['gt_pixels']>=50)&(t['near']['predicted_pixels']>=50),data['counts'][:,4:6,2].sum(1)),
        ('07h_near_retained',(data['hour']==7)&(~t['near']['error'])&(t['near']['gt_pixels']>=50)&(t['near']['predicted_pixels']>=50),t['whole_frame_iou'])]
    for name,eligible,rank in definitions:
        candidates=[int(i)for i in np.flatnonzero(eligible)if int(i)not in used]
        if not candidates:continue
        i=max(candidates,key=lambda j:(rank[j],str(data['ids'][j])));used.add(i);chosen.append((name,i))
    if a.case_index is not None:
        baseline=json.loads(a.case_index.read_text());indices={str(s):i for i,s in enumerate(data['ids'])};chosen=[]
        for case in baseline['cases']:
            if case['source_manifest_sha256']!=sha256_file(a.manifest):raise ValueError('paired case source membership differs')
            chosen.append((a.case_label_prefix+case['case'],indices[case['sample_id']]))
    selected=[lookup[str(data['ids'][i])]for _,i in chosen];dataset=CachedAdaptationSamples(a.cache,selected,verify=True)
    model=load_target_model(checkpoint,torch.device('cpu'),weights='model')
    plt.rcParams.update({'font.family':'DejaVu Sans'});images=[];records=[]
    with torch.inference_mode():
        for j,(name,i)in enumerate(chosen):
            item=dataset[j];_,output=model.model(item['image'][None]);pred=np.isin(output.argmax(1)[0,12:372].numpy(),[1,2]);gt=np.isin(item['targets']['road_marking'][12:372].numpy(),[1,2]);rgb=np.rint(item['image'][:,12:372].permute(1,2,0).numpy()*255).astype(np.uint8)
            if int(gt.sum())!=int((data['counts'][i,:,0]+data['counts'][i,:,2]).sum()):raise ValueError('true GT mask does not match selected frame counts')
            gt_image=np.zeros_like(rgb);gt_image[gt]=[0,220,255];pred_image=np.zeros_like(rgb);pred_image[pred]=[70,225,120]
            errors=(rgb.astype(float)*.55).astype(np.uint8);errors[pred & gt]=[60,190,90];errors[pred & ~gt]=[255,70,70];errors[~pred & gt]=[255,225,30]
            tp=int((pred & gt).sum());fp=int((pred & ~gt).sum());fn=int((~pred & gt).sum());quality=tp/(tp+fp+fn)if tp+fp+fn else None
            if data['counts'][i].sum(0).tolist()!=[tp,fp,fn]:raise ValueError('case prediction/error raster differs from the versioned feature pack')
            folder=a.output/name;folder.mkdir()
            panels=[rgb,gt_image,pred_image,errors]
            for filename,value in zip(['rgb.png','gt.png','prediction.png','errors.png'],panels):cv2.imwrite(str(folder/filename),cv2.cvtColor(value,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(folder/'gt_mask.png'),gt.astype(np.uint8)*255)
            record=dict(case=name,sample_id=str(data['ids'][i]),site=str(data['site'][i]),weather=int(data['weather'][i]),hour=int(data['hour'][i]),
                display_hw=[360,640],source_camera_hw=[720,1280],display_grid='model-input crop12:372; true GT nearest-neighbor target raster',recommended_aspect_ratio='16:9; preserve proportions',true_gt_pixels=int(gt.sum()),perception_identity=identity,
                recomputed_lane_iou=quality,source_manifest_sha256=sha256_file(a.manifest),source_pack_sha256=sha256_file(a.pack),
                image_sha256={str(path.relative_to(a.output)):sha256_file(path)for path in folder.glob('*.png')},
                scope=a.case_scope+', selected after GT audit. GT panel is decoded source target, never perception output; no safety/normality claim.')
            records.append(record);images.append((record,panels))
    for start,title in [(0,'Supported error examples'),(2,'Distance and illumination context')]:
        group=images[start:start+2]
        if not group:continue
        fig,axes=plt.subplots(len(group),4,figsize=(16,2.6*len(group)),squeeze=False,layout='constrained')
        for row,(record,panels)in enumerate(group):
            for col,value in enumerate(panels):
                axes[row,col].imshow(value,interpolation='nearest');axes[row,col].axis('off')
                axes[row,col].set_title(['RGB input','True GT: white/yellow union','Perception prediction','TP green / FP red / FN yellow'][col],fontsize=10)
            axes[row,0].set_title('RGB input | '+record['case']+'\npreset'+str(record['weather'])+' / '+str(record['hour'])+'h / IoU '+f"{record['recomputed_lane_iou']:.3f}",fontsize=10)
        fig.suptitle(title+' — same frame and preserved16:9 panels',fontsize=13)
        for ext in ['png','svg']:fig.savefig(a.output/(('05_near_errors'if start==0 else'06_distance_dawn')+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
    (a.output/'cases.json').write_text(json.dumps(dict(cases=records,source_sha256=sha256_file(Path(__file__)),baseline_case_index_sha256=sha256_file(a.case_index)if a.case_index is not None else None,case_index_sha256=sha256_file(a.case_index)if a.case_index is not None else None),indent=2,allow_nan=False)+'\n')
    print('EXPORTED VERIFIED TRUE-GT CASES',len(records),flush=True)


if __name__=='__main__':main()
