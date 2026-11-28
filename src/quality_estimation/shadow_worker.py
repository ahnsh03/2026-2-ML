"""Persistent GT-free scorer for an append-only RGB archive; bounded latest queue.

Writes separate inference/selection receipts. Original captured frames remain
available even when a busy live worker skips them. No vehicle command publisher.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import cv2,numpy as np,torch
from common.bridge import load_target_model,sha256_file,git_head
from .camera_udp import atomic_json,utc
from .perception_identity import resolve_public_perception
from .support_aware_review import SupportAwareReview
from .infer_images import load_rgb_row
from .roi10 import reduce_heads
from .selected_region_features import extract_selected_features
from .runtime_catalog import verify_repository_bundle
from .shadow_replay import validate_batch_plan


class PersistentReviewScorer:
    def __init__(self,release,bundles,device='cuda'):
        cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        checkpoint,self.identity,_=resolve_public_perception(release)
        self.device=torch.device(device);self.perception=load_target_model(checkpoint,self.device,weights='model')
        for path in bundles.values():verify_repository_bundle(path)
        self.reviews={name:SupportAwareReview(path,perception_identity=self.identity)for name,path in bundles.items()}
        if 'primary'not in self.reviews:raise ValueError('primary reference bundle required')
        self.columns=list(dict.fromkeys(n for r in self.reviews.values()for n in r.columns+r.environment.columns));self.lookup={n:i for i,n in enumerate(self.columns)}
        self.bundle_hashes={name:sha256_file(Path(path)/'bundle_manifest.json')for name,path in bundles.items()}
    def take(self,x,names):return x[:,[self.lookup[n]for n in names]]
    @torch.inference_mode()
    def score(self,root,rows):
        items=[load_rgb_row(root,row,i)for i,row in enumerate(rows)]
        if any(i['rgb'].shape!=(720,1280,3)for i in items):raise ValueError('fixed front 1280x720 geometry required')
        images=torch.from_numpy(np.stack([i['image']for i in items])).to(self.device);drive,mark=self.perception.model(images)
        ds=drive.argmax(1).cpu().numpy();ps=mark.float().softmax(1).cpu().numpy();profile=self.reviews['primary'].profile
        x=np.stack([extract_selected_features(i['rgb'],reduce_heads(d,p,i['geometry']),profile,self.columns,marking_probability=p)for i,d,p in zip(items,ds,ps)])
        pred={name:r.predict(self.take(x,r.columns),self.take(x,r.environment.columns))for name,r in self.reviews.items()}
        return items,x,pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['archive','output','perception-release','bundle']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--manifest',type=Path,help='Optional immutable RGB subset for exact processed-population replay');p.add_argument('--replay-batches',type=Path,help='Original live batch membership; offline replay only');p.add_argument('--gp-bundle',type=Path);p.add_argument('--rf-bundle',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--window',type=int,default=100);p.add_argument('--fraction',type=float,default=.1);p.add_argument('--watch',action='store_true');p.add_argument('--maximum-seconds',type=float,default=3600);p.add_argument('--input-timeout',type=float,default=60)
    a=p.parse_args()
    if a.batch_size<1 or a.window<1 or not 0<a.fraction<=1 or a.maximum_seconds<=0 or a.input_timeout<=0:raise ValueError('invalid runtime bounds')
    batch_plan=None;replay_sizes=None;replay_batch=0;replay_manifest_sha=None;batch_plan_sha=None
    if a.replay_batches:
        if a.watch or not a.manifest:raise ValueError('replay batch plan requires a fixed offline manifest')
        batch_plan=json.loads(a.replay_batches.read_text());replay_sizes=validate_batch_plan(batch_plan,[json.loads(line)for line in a.manifest.read_text().splitlines()]);replay_manifest_sha=sha256_file(a.manifest);batch_plan_sha=sha256_file(a.replay_batches)
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();atomic_json(a.output/'status.json',dict(status='initializing',updated_utc=utc()))
    bundles={'primary':a.bundle}
    if a.gp_bundle:bundles['GP']=a.gp_bundle
    if a.rf_bundle:bundles['RF']=a.rf_bundle
    scorer=PersistentReviewScorer(a.perception_release,bundles,a.device)
    warmup_started=time.monotonic()
    demo_root=Path(__file__).resolve().parents[2]/'models/examples'
    demo_row=json.loads((demo_root/'rgb.jsonl').read_text().splitlines()[0])
    for batch in [1,a.batch_size]:scorer.score(demo_root,[demo_row]*batch)
    warmup_seconds=time.monotonic()-warmup_started
    source_digest=sha256_file(Path(__file__));identity=scorer.identity
    repo_root=Path(__file__).resolve().parents[2]
    components=['src/quality_estimation/shadow_replay.py','src/quality_estimation/selected_region_features.py','src/quality_estimation/region_candidates.py',
                'src/quality_estimation/support_aware_review.py','src/quality_estimation/support_review_selection.py',
                'src/quality_estimation/scene_environment.py','src/quality_estimation/spatial_gp.py',
                'src/quality_estimation/portable_forest.py','src/quality_estimation/roi10.py',
                'src/quality_estimation/road_structure_features.py','src/camera_semantic_perception/geometry.py','src/quality_estimation/shadow_replay.py']
    implementation_sha256={name:sha256_file(repo_root/name)for name in components}
    context=dict(code_git_head=git_head(repo_root),implementation_sha256=implementation_sha256,schema='persistent-review-shadow-1',started_utc=utc(),perception_identity=identity,bundle_manifest_sha256=scorer.bundle_hashes,
        warmup_seconds=warmup_seconds,feature_columns=scorer.columns,window=a.window,budget_fraction=a.fraction,device=a.device,torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,replay_batch_plan_sha256=batch_plan_sha,
        camera=dict(view='front',width=1280,height=720,profile=scorer.reviews['primary'].profile.version,regions_sha256=scorer.reviews['primary'].profile.regions_sha256),
        runtime_hardware=dict(device=str(scorer.device),cuda_runtime=torch.version.cuda,gpu_name=torch.cuda.get_device_name(scorer.device)if scorer.device.type=='cuda'else None,capability=list(torch.cuda.get_device_capability(scorer.device))if scorer.device.type=='cuda'else None),uses_ground_truth=False,emits_vehicle_commands=False,automatic_training_admission=False,queue='bounded_latest_batch'if a.watch else'complete_replay',source_sha256=source_digest)
    atomic_json(a.output/'context.json',context);atomic_json(a.output/'READY.json',dict(utc=utc(),context_sha256=sha256_file(a.output/'context.json')))
    atomic_json(a.output/'status.json',dict(status='waiting_input',updated_utc=utc(),context_sha256=sha256_file(a.output/'context.json')))
    seen=0;processed=0;skipped=0;windows=0;buffer=[];features=[];all_features=[];ids=[];latencies=[];reason='end_of_archive';error=None
    streams={name:(a.output/(name+'.jsonl')).open('x',buffering=1)for name in ['predictions','candidates','inference_skipped','processed_rgb']}
    def flush():
        nonlocal buffer,features,windows
        if not buffer:return
        x=np.concatenate(features);sample_ids=np.array([r['sample_id']for r in buffer]);budget=int(math.ceil(len(buffer)*a.fraction))
        for name,r in scorer.reviews.items():
            keys=buffer[0]['review'][name];pred={k:np.array([row['review'][name][k]for row in buffer])for k in keys}
            chosen,priority,details=r.select_with_features(sample_ids,pred,budget,scorer.take(x,r.columns))
            for rank,i in enumerate(chosen):
                item=dict(sample_id=str(sample_ids[i]),review_family=name,window=windows,population=len(buffer),budget=budget,rank=rank+1,
                    priority=float(priority[i]),selection_reason=details['reasons'][rank],conditional_inclusion_probability=details['conditional_inclusion_probability'][rank],
                    admission_approved=False,uses_ground_truth=False,perception_identity=identity)
                streams['candidates'].write(json.dumps(item,allow_nan=False)+'\n')
        windows+=1;buffer=[];features=[]
    try:
        while time.monotonic()-started<a.maximum_seconds:
            manifest=a.manifest or a.archive/'rgb.jsonl'
            lines=manifest.read_text().splitlines()if manifest.exists()else[]
            # Writers commit a complete row with newline after the immutable JPEG.
            rows=[]
            for line in lines:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if line!=lines[-1]:raise
            if len(rows)<seen:raise ValueError('archive manifest truncated')
            if len({row['sample_id']for row in rows})!=len(rows):raise ValueError('archive has duplicate frame identifiers')
            pending=rows[seen:]
            done=not a.watch or(a.archive/'receiver_done.json').exists()or(a.archive/'STOP_WORKER').exists()
            if pending:
                if a.watch and len(pending)>a.batch_size:
                    omitted=pending[:-a.batch_size]
                    for row in omitted:streams['inference_skipped'].write(json.dumps(dict(sample_id=row['sample_id'],reason='bounded_latest_queue',raw_preserved=True))+'\n')
                    seen+=len(omitted);skipped+=len(omitted);pending=pending[-a.batch_size:]
                size=replay_sizes[replay_batch]if replay_sizes else a.batch_size
                batch=pending[:size];tick=time.monotonic();items,x,pred=scorer.score(a.archive,batch);duration=time.monotonic()-tick
                if replay_sizes:replay_batch+=1
                for i,(row,item)in enumerate(zip(batch,items)):
                    result=dict(sample_id=item['sample_id'],rgb_sha256=item['sha256'],pixel_sha256=item['pixel_sha256'],processed_utc=utc(),
                        batch_seconds=duration,batch_rows=len(batch),review={name:{key:value[i].tolist()for key,value in pp.items()}for name,pp in pred.items()},
                        uses_ground_truth=False,admission_approved=False,perception_identity=identity)
                    streams['predictions'].write(json.dumps(result,allow_nan=False)+'\n');streams['processed_rgb'].write(json.dumps(row,allow_nan=False)+'\n')
                    buffer.append(result);features.append(x[i:i+1]);all_features.append(x[i:i+1]);ids.append(item['sample_id'])
                    if len(buffer)==a.window:flush()
                seen+=len(batch);processed+=len(batch);latencies.append(duration/len(batch))
                atomic_json(a.output/'status.json',dict(status='running',processed=processed,inference_skipped=skipped,archive_rows_seen=seen,windows=windows,pending_window_rows=len(buffer),updated_utc=utc()))
                continue
            if done:reason='archive_complete';break
            if processed==0 and time.monotonic()-started>warmup_seconds+a.input_timeout:
                reason='no_input_timeout';break
            time.sleep(.1)
        else:reason='worker_time_limit'
        flush()
        if all_features:np.savez_compressed(a.output/'features.npz',X=np.concatenate(all_features),ids=np.array(ids),ordered_features=np.array(scorer.columns))
        for r in scorer.reviews.values():r.verify_unchanged()
        if any(sha256_file(repo_root/name)!=value for name,value in implementation_sha256.items()):raise ValueError('runtime implementation changed during session')
        if sha256_file(Path(__file__))!=source_digest:raise ValueError('worker source changed during execution')
        if replay_sizes and(replay_batch!=len(replay_sizes)or sha256_file(a.manifest)!=replay_manifest_sha or sha256_file(a.replay_batches)!=batch_plan_sha):raise ValueError('replay inputs changed or planned batches were not completed')
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e));reason='worker_error';raise
    finally:
        for s in streams.values():s.close()
        status=dict(status='failed'if error else'complete'if processed else'no_input',reason=reason,processed=processed,inference_skipped=skipped,archive_rows_seen=seen,selection_windows=windows,seconds=time.monotonic()-started,
                    mean_processing_seconds_per_frame=float(np.mean(latencies))if latencies else None,error=error,finished_utc=utc(),context_sha256=sha256_file(a.output/'context.json'),input_manifest_sha256=sha256_file(a.manifest or a.archive/'rgb.jsonl')if(a.manifest or a.archive/'rgb.jsonl').exists()else None)
        atomic_json(a.output/'status.json',status);atomic_json(a.output/'run.json',status)
        if error is None and processed:(a.output/'_SUCCESS').write_text(sha256_file(a.output/'run.json')+'\n')
    print(json.dumps(status))
    if not processed:raise SystemExit(3)
if __name__=='__main__':main()
