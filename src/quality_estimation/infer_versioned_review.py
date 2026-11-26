"""Offline RGB-only replay through a verified f0/g0 or f1/g1 review bundle."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from common.bridge import load_target_model,sha256_file
from .perception_identity import resolve_quality_perception
from .versioned_review import VersionedReview
from .infer_images import RGBSamples
from .build_dataset import collate
from .roi10 import reduce_heads
from .selected_region_features import extract_selected_features

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['bundle','manifest','data-root','output']:p.add_argument('--'+key,type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--checkpoint',type=Path);source.add_argument('--adaptation-run',type=Path)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',default='cuda');p.add_argument('--budget',type=int);a=p.parse_args()
    if a.output.exists():raise ValueError('RGB replay output exists')
    path,identity,_=resolve_quality_perception(a.adaptation_run,a.checkpoint);review=VersionedReview(a.bundle,perception_identity=identity)
    columns=list(dict.fromkeys(review.columns+review.environment.columns));lookup={n:i for i,n in enumerate(columns)}
    take=lambda x,names:x[:,[lookup[n] for n in names]]
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('empty/duplicate RGB rows')
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(a.device);perception=load_target_model(path,device,weights='ema' if identity['kind']=='baseline' else 'model')
    loader=DataLoader(RGBSamples(a.data_root,rows),batch_size=a.batch_size,num_workers=2,collate_fn=collate,shuffle=False)
    a.output.mkdir(parents=True);vectors=[];records=[];parts={};started=time.monotonic()
    with torch.inference_mode():
        for images,items in loader:
            if any(item['rgb'].shape!=(720,1280,3) for item in items):raise ValueError('fixed front camera dimensions differ')
            drive,mark=perception.model(images.to(device));ds=drive.argmax(1).cpu().numpy();ps=mark.float().softmax(1).cpu().numpy()
            x=np.stack([extract_selected_features(item['rgb'],reduce_heads(d,p_,item['geometry']),review.profile,columns,marking_probability=p_) for item,d,p_ in zip(items,ds,ps)])
            prediction=review.predict(take(x,review.columns),take(x,review.environment.columns));vectors.append(x)
            for key,value in prediction.items():parts.setdefault(key,[]).append(value)
            for j,item in enumerate(items):
                record={key:value[j].tolist() for key,value in prediction.items()}
                record.update(sample_id=item['sample_id'],rgb_sha256=item['sha256'],pixel_sha256=item['pixel_sha256'],checkpoint_sha256=identity['checkpoint_sha256'],
                              perception_identity=identity,perception_weights=perception.weights,profile=review.profile.version,regions_sha256=review.profile.regions_sha256,
                              model_sha256=sha256_file(a.bundle/'quality_gp.npz'),uses_ground_truth=False,admission_approved=False,
                              band_diagnostic_near_metric='near_f1' if len(review.target_names)==4 else 'near_recall_only_not_precision')
                records.append(record)
    predictions={k:np.concatenate(v) for k,v in parts.items()};ids=np.array([r['sample_id'] for r in records]);budget=a.budget if a.budget is not None else int(np.ceil(len(rows)*.1))
    selected,priority=review.select(ids,predictions,budget)
    rounding=np.ceil if review.policy.get('audit_rounding','floor')=='ceil' else np.floor
    audit=int(rounding(budget*review.policy['audit_fraction']));priority_count=budget-audit
    candidates=[dict(records[i],priority=float(priority[i]),selection_rank=j+1,review_population=len(rows),budget=budget,
                     selection_component='priority' if j<priority_count else 'random_audit',
                     conditional_inclusion_probability=1. if j<priority_count else audit/(len(rows)-priority_count),
                     selection_scope='Whole provided archive buffer; not independently ranked inference batches.') for j,i in enumerate(selected)]
    for name,value in [('predictions.jsonl',records),('candidates.jsonl',candidates)]:
        (a.output/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in value))
    np.savez_compressed(a.output/'features.npz',X=np.concatenate(vectors),ids=ids,ordered_features=columns)
    metadata=dict(status='complete',rows=len(rows),budget=budget,audit_slots=audit,priority_slots=priority_count,uses_ground_truth=False,perception_identity=identity,perception_weights=perception.weights,
                  checkpoint_sha256=identity['checkpoint_sha256'],profile=review.profile.version,regions_sha256=review.profile.regions_sha256,
                  bundle_manifest_sha256=sha256_file(a.bundle/'bundle_manifest.json'),manifest_sha256=sha256_file(a.manifest),source_sha256=sha256_file(Path(__file__)),
                  candidates_sha256=sha256_file(a.output/'candidates.jsonl'),seconds=time.monotonic()-started,scope='Offline RGB replay; no live ROS or automatic training admission.')
    metadata['artifact_sha256']={f.name:sha256_file(f) for f in a.output.iterdir() if f.is_file()}
    (a.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n');(a.output/'_SUCCESS').write_text(sha256_file(a.output/'run.json')+'\n')
if __name__=='__main__':main()
