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
from .perception_identity import baseline_identity

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','data-root','feature-study','calibration','policy','checkpoint','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--simulator-release',required=True)
    parser.add_argument('--save-heads',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    profile=SpatialProfile.load();checkpoint=sha256_file(args.checkpoint)
    predictor=DriveReviewPredictor(args.feature_study,args.calibration,args.policy,checkpoint_sha256=checkpoint,regions_sha256=profile.regions_sha256,simulator_release=args.simulator_release)
    columns=list(dict.fromkeys(predictor.columns+(predictor.auxiliary['columns'] if predictor.auxiliary else [])+predictor.environment['ordered_features']))
    positions={name:i for i,name in enumerate(columns)}
    def take(vector,names):return vector[:,[positions[n] for n in names]]
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device);model=load_target_model(args.checkpoint,device,weights='ema')
    if model.weights!='ema_state_dict':raise ValueError('expected EMA perception weights')
    identity=baseline_identity()
    if checkpoint!=identity['checkpoint_sha256']:raise ValueError('this frozen drive study requires the baseline perception')
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or len(set(r['sample_id'] for r in rows))!=len(rows):raise ValueError('empty/duplicate RGB IDs')
    loader=DataLoader(RGBSamples(args.data_root,rows),batch_size=4,num_workers=0,collate_fn=collate,worker_init_fn=worker_init,shuffle=False)
    args.output.mkdir(parents=True);vectors=[];ids=[];results=[];start=time.perf_counter()
    (args.output/'input_manifest.jsonl').write_bytes(args.manifest.read_bytes())
    with torch.inference_mode():
        for images,items in loader:
            drive,mark=model.model(images.to(device));drives=drive.argmax(1).cpu().numpy();probabilities=mark.float().softmax(1).cpu().numpy()
            x=np.stack([extract_selected_features(item['rgb'],reduce_heads(d,p,item['geometry']),profile,columns,marking_probability=p) for item,d,p in zip(items,drives,probabilities)])
            prediction=predictor.predict_vectors(take(x,predictor.columns),take(x,predictor.auxiliary['columns']) if predictor.auxiliary else None,take(x,predictor.environment['ordered_features']))
            for i,item in enumerate(items):
                result={key:value[i].tolist() for key,value in prediction.items()}
                result.update(sample_id=item['sample_id'],rgb_sha256=item['sha256'],uses_ground_truth=False,
                              checkpoint_sha256=checkpoint,profile=profile.version,regions_sha256=profile.regions_sha256,
                              perception_identity=identity,perception_weights=model.weights)
                result['spatial_scope']='Diagnostic probabilities assume sufficient near/far annotation support. Online GT support is unknown; not an automatic failure/admission decision.'
                if args.save_heads:
                    name=f'{len(results):06d}_heads.npz'
                    np.savez_compressed(args.output/name,marking_class=probabilities[i].argmax(0).astype(np.uint8),head_maps=reduce_heads(drives[i],probabilities[i],item['geometry']))
                    result.update(heads_file=name,heads_sha256=sha256_file(args.output/name))
                results.append(result);ids.append(item['sample_id'])
            vectors.append(x)
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results))
    np.savez_compressed(args.output/'features.npz',X=np.concatenate(vectors),ids=ids,ordered_features=columns)
    metadata=dict(status='complete',rows=len(ids),uses_ground_truth=False,checkpoint_sha256=checkpoint,regions_sha256=profile.regions_sha256,model_sha256=predictor.calibration['model_sha256'],policy_sha256=sha256_file(args.policy),manifest_sha256=sha256_file(args.manifest),source_sha256=sha256_file(Path(__file__)),seconds=time.perf_counter()-start,device=str(device),scope='GT-free RGB replay; no live ROS or FPS claim.')
    metadata['artifacts_sha256']={p.name:sha256_file(p) for p in args.output.iterdir() if p.is_file()}
    metadata.update(perception_identity=identity,perception_weights=model.weights,simulator_release=args.simulator_release,
                    calibration_sha256=sha256_file(args.calibration/'calibration_frozen.json'),
                    auxiliary_model_sha256=predictor.auxiliary['model_sha256'] if predictor.auxiliary else None,
                    environment_profile_sha256=sha256_file(args.calibration/'environment_profile.json'))
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'_SUCCESS').write_text(sha256_file(args.output/'run.json')+'\n');print(json.dumps(metadata),flush=True)
if __name__=='__main__':main()
