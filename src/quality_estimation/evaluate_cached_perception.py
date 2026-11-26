"""Paired FP32 evaluation of a verified perception version on lossless cached inputs."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from common.bridge import load_target_model,sha256_file
from .adaptation_cache import CachedAdaptationSamples
from .artifacts import write_json
from .perception_identity import resolve_quality_perception
from .spatial_features import SpatialProfile

def summarize(records):
    result={}
    for head in ['lane','drivable','white','yellow','stopline']:
        counts=np.array([r['counts'][head] for r in records],np.int64);union=counts.sum(1);defined=union>0
        frame=np.divide(counts[:,0],union,out=np.full(len(counts),np.nan),where=defined)
        tp,fp,fn=counts.sum(0)
        result[head]=dict(rows=len(records),defined=int(defined.sum()),mean_frame_iou=float(np.mean(frame[defined])) if defined.any() else None,
                          pooled_pixel_iou=float(tp/(tp+fp+fn)) if tp+fp+fn else None,
                          tp=int(tp),fp=int(fp),fn=int(fn))
        if head=='lane':
            groups=np.array([r.get('site') or r.get('parent_session_id','unknown') for r in records]);valid=(counts[:,0]+counts[:,2]>=50)&defined
            perroad={str(g):float(np.mean(frame[(groups==g)&valid])) for g in np.unique(groups) if ((groups==g)&valid).any()}
            result[head].update(supported_macro_road_iou=float(np.mean(list(perroad.values()))) if perroad else None,supported_by_road=perroad,supported_rows=int(valid.sum()))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    model=p.add_mutually_exclusive_group(required=True);model.add_argument('--checkpoint',type=Path);model.add_argument('--adaptation-run',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--role',choices=['development_retention','heldout_diagnostic'],required=True)
    a=p.parse_args();rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    if a.output.exists():raise ValueError('evaluation output exists')
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('empty or duplicate evaluation rows')
    permitted=['development','retention'] if a.role=='development_retention' else ['heldout_diagnostic']
    if any(r['adaptation_role'] not in permitted for r in rows):raise ValueError('evaluation role differs from frozen manifest')
    checkpoint,identity,membership=resolve_quality_perception(a.adaptation_run,a.checkpoint)
    ids={r['sample_id'] for r in membership};pixels={r['pixel_sha256'] for r in membership}
    if ids.intersection(r['sample_id'] for r in rows) or pixels.intersection(r['pixel_sha256'] for r in rows):raise ValueError('evaluation pixels admitted for training')
    a.output.mkdir(parents=True);cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(a.device);loaded=load_target_model(checkpoint,device,weights='ema' if identity['kind']=='baseline' else 'model')
    loader=DataLoader(CachedAdaptationSamples(a.cache,rows,verify=True),batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',shuffle=False)
    profile=SpatialProfile.load();roi=profile.native[::2,::2];records=[];start=time.monotonic()
    with torch.inference_mode(),(a.output/'per_frame.jsonl').open('x') as stream:
        for batch in loader:
            drive,mark=loaded.model(batch['image'].to(device));drive=drive.argmax(1).cpu().numpy();mark=mark.argmax(1).cpu().numpy()
            for j,i in enumerate(batch['index'].tolist()):
                row=rows[i];truth=batch['targets']['road_marking'][j].numpy();valid=batch['valid_mask'][j].numpy();pred=mark[j]
                def counts(p,t):return [int(np.count_nonzero(p&t&valid)),int(np.count_nonzero(p&~t&valid)),int(np.count_nonzero(~p&t&valid))]
                metrics={'lane':counts(np.isin(pred,[1,2]),np.isin(truth,[1,2])),
                         'drivable':counts(drive[j]==1,batch['targets']['drivable'][j].numpy()==1)}
                for name,c in [('white',1),('yellow',2),('stopline',3)]:metrics[name]=counts(pred==c,truth==c)
                record={k:row.get(k) for k in ['sample_id','parent_session_id','view','site','source_key','weather','hour','xy','adaptation_role','pixel_sha256']};record['counts']=metrics
                if row['view']=='front':
                    p=np.isin(pred[12:372],[1,2]);t=np.isin(truth[12:372],[1,2]);region=np.stack([np.bincount(roi[m],minlength=9) for m in [p&t,p&~t,~p&t]],1)
                    if region.sum(0).tolist()!=metrics['lane']:raise ValueError('image-band counts differ from whole-image target')
                    record['region_counts']=region.tolist()
                stream.write(json.dumps(record,allow_nan=False)+'\n');records.append(record)
            if len(records)%100< a.batch_size:stream.flush();print('evaluated',len(records),'/',len(rows),flush=True)
    grouped={}
    for key in ['adaptation_role','view','source_key']:
        grouped[key]={str(value):summarize([r for r in records if r[key]==value]) for value in sorted({r[key] for r in records},key=str)}
    conditions=sorted({(r['weather'],r['hour']) for r in records if r['weather'] is not None})
    grouped['condition']={str(pair):summarize([r for r in records if (r['weather'],r['hour'])==pair]) for pair in conditions}
    metadata=dict(status='complete',perception_identity=identity,checkpoint_sha256=sha256_file(checkpoint),perception_weights=loaded.weights,
                  manifest_sha256=sha256_file(a.manifest),source_sha256=sha256_file(Path(__file__)),role=a.role,rows=len(records),
                  metrics=summarize(records),grouped=grouped,elapsed_seconds=time.monotonic()-start,device=str(device),precision='FP32; TF32 off; deterministic cuDNN',
                  scope='Previously opened cohorts excluded from new adaptation; paired diagnostic, not independent safety/visibility truth.')
    write_json(a.output/'summary.json',metadata);(a.output/'_SUCCESS').write_text(sha256_file(a.output/'summary.json')+'\n')
if __name__=='__main__':main()
