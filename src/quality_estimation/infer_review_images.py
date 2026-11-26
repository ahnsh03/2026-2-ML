"""Replay RGB only through fixed perception, selected GPs and review calibration.

No Semantic path, GT target, simulator weather/time or road ID is loaded. This
is offline RGB replay, not a live ROS or end-to-end real-time acceptance test.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from common.bridge import load_target_model,sha256_file
from .infer_images import RGBSamples
from .build_dataset import collate,worker_init
from .spatial_features import SpatialProfile
from .selected_region_features import extract_selected_features
from .roi10 import reduce_heads
from .drive_review import DriveReviewPredictor

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','data-root','feature-study','calibration','policy','checkpoint','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    profile=SpatialProfile.load();checkpoint=sha256_file(args.checkpoint)
    predictor=DriveReviewPredictor(args.feature_study,args.calibration,args.policy,checkpoint_sha256=checkpoint,regions_sha256=profile.regions_sha256)
    columns=list(dict.fromkeys(predictor.columns+(predictor.auxiliary['columns'] if predictor.auxiliary else [])+predictor.environment['ordered_features']))
    positions={name:i for i,name in enumerate(columns)}
    def take(vector,names):return vector[:,[positions[n] for n in names]]
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device);model=load_target_model(args.checkpoint,device,weights='ema')
    if model.weights!='ema_state_dict':raise ValueError('expected EMA perception weights')
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or len(set(r['sample_id'] for r in rows))!=len(rows):raise ValueError('empty/duplicate RGB IDs')
    loader=DataLoader(RGBSamples(args.data_root,rows),batch_size=4,num_workers=0,collate_fn=collate,worker_init_fn=worker_init,shuffle=False)
    args.output.mkdir(parents=True);vectors=[];ids=[];results=[];start=time.perf_counter()
    with torch.inference_mode():
        for images,items in loader:
            drive,mark=model.model(images.to(device));drives=drive.argmax(1).cpu().numpy();probabilities=mark.float().softmax(1).cpu().numpy()
            x=np.stack([extract_selected_features(item['rgb'],reduce_heads(d,p,item['geometry']),profile,columns,marking_probability=p) for item,d,p in zip(items,drives,probabilities)])
            prediction=predictor.predict_vectors(take(x,predictor.columns),take(x,predictor.auxiliary['columns']) if predictor.auxiliary else None,take(x,predictor.environment['ordered_features']))
            for i,item in enumerate(items):
                result={key:value[i].tolist() for key,value in prediction.items()}
                result.update(sample_id=item['sample_id'],rgb_sha256=item['sha256'],uses_ground_truth=False)
                results.append(result);ids.append(item['sample_id'])
            vectors.append(x)
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results))
    np.savez_compressed(args.output/'features.npz',X=np.concatenate(vectors),ids=ids,ordered_features=columns)
    metadata=dict(status='complete',rows=len(ids),uses_ground_truth=False,checkpoint_sha256=checkpoint,regions_sha256=profile.regions_sha256,model_sha256=predictor.calibration['model_sha256'],policy_sha256=sha256_file(args.policy),manifest_sha256=sha256_file(args.manifest),source_sha256=sha256_file(Path(__file__)),seconds=time.perf_counter()-start,device=str(device),scope='GT-free RGB replay; no live ROS or FPS claim.')
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)
if __name__=='__main__':main()
