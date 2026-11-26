"""Balanced, resumable long adaptation using bounded-memory source-verified caches."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler

from common.bridge import DEFAULT_CHECKPOINT_SHA256,load_target_model,resolve_checkpoint,sha256_file
from twinlite_morai import MaskedTwinLiteLoss,ROAD_MARKING_TASK,adapt_twinlite_outputs
from .adaptation_cache import CachedAdaptationSamples
from .artifacts import write_json
from .distillation import masked_replay_distillation
from .perception_data import POLICY_SHA256


class BalancedBatches(Sampler):
    """Cycling permutations cover pools; the replay RNG is shared across arms."""
    def __init__(self,n_new,n_replay,new_per_batch,replay_per_batch,steps,seed,start=0):
        self.args=n_new,n_replay,new_per_batch,replay_per_batch,steps,seed,start
    def __len__(self):return self.args[4]-self.args[6]
    def __iter__(self):
        nn,nr,bn,br,steps,seed,start=self.args
        rngs=[np.random.default_rng(seed),np.random.default_rng(seed+100000)]
        sizes=[nn,nr];pending=[[],[]]
        for step in range(steps):
            groups=[]
            for j,count in enumerate([bn,br]):
                while len(pending[j])<count:pending[j].extend(rngs[j].permutation(sizes[j]).tolist())
                group=pending[j][:count];del pending[j][:count];groups.extend([i+(nn if j else 0) for i in group])
            if step>=start:yield groups


def atomic_checkpoint(path,state):
    temp=path.with_suffix('.tmp.pt');torch.save(state,temp);temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['new-manifest','replay-manifest','protocol','cache','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint');p.add_argument('--arm',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());out=a.output_dir
    if out.exists() and not a.resume:raise ValueError('training output exists')
    new=[json.loads(s) for s in a.new_manifest.read_text().splitlines() if s.strip()]
    replay=[json.loads(s) for s in a.replay_manifest.read_text().splitlines() if s.strip()];rows=new+replay
    if len({r['pixel_sha256'] for r in rows})!=len(rows):raise ValueError('duplicate training pixels')
    if {r['sample_id'] for r in new}&{r['sample_id'] for r in replay}:raise ValueError('overlapping admissions')
    control=a.arm=='ReplayOnly'
    if control:
        if any(r.get('adaptation_role')!='replay_replacement' or r.get('source_role')!='f0_train' for r in new):raise ValueError('replay replacement must be original f0 train')
    elif any(r.get('adaptation_role')!='candidate_pool' for r in new):raise ValueError('non-candidate admission')
    if any(r.get('source_role')!='f0_train' for r in replay):raise ValueError('unverified replay membership')
    if a.arm not in protocol['selection_arms']:raise ValueError('arm not preregistered')
    checkpoint=resolve_checkpoint(a.checkpoint)
    if sha256_file(checkpoint)!=DEFAULT_CHECKPOINT_SHA256:raise ValueError('paired study must start from f0')
    out.mkdir(parents=True,exist_ok=True)
    if not a.resume:
        for source,name in [(a.protocol,'protocol.json'),(a.new_manifest,'new_manifest.jsonl'),(a.replay_manifest,'replay_manifest.jsonl')]:shutil.copyfile(source,out/name)
        write_json(out/'training_membership.json',[dict(sample_id=r['sample_id'],parent_session_id=r.get('parent_session_id',r.get('group')),
                pixel_sha256=r['pixel_sha256'],input_sha256=r['expected_sha256'],cache_receipt_sha256=r['cache_receipt_sha256']) for r in rows])
    metadata=dict(schema_version='ml-perception-adaptation-1',status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                  arm=a.arm,seed=protocol['seed'],initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,target_policy_sha256=POLICY_SHA256,
                  new_rows=0 if control else len(new),replacement_replay_rows=len(new) if control else 0,replay_rows=len(replay),steps=protocol['steps'],protocol_sha256=sha256_file(a.protocol),
                  source_sha256={str(Path(__file__).name):sha256_file(Path(__file__)), 'adaptation_cache.py':sha256_file(Path(__file__).with_name('adaptation_cache.py'))},
                  train_parent_groups=sorted({r.get('parent_session_id',r.get('group','unknown')) for r in rows}),
                  final_test_accessed=False,quality_regressor_recalibrated_for_f1=False,
                  selection='fixed optimizer step; no test/checkpoint selection',strict_bitwise_determinism=False,
                  precision=protocol['precision'],allowed_nondeterministic_operations=['adaptive_avg_pool2d_backward_cuda'])
    for source in [Path(__file__),Path(__file__).with_name('adaptation_cache.py')]:
        destination=out/'source'/source.name;destination.parent.mkdir(exist_ok=True)
        if not destination.exists():shutil.copyfile(source,destination)
        if sha256_file(destination)!=sha256_file(source):raise ValueError('resume source changed')
    write_json(out/'run.json',metadata)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');loaded=load_target_model(checkpoint,device);model=loaded.model.train()
    teacher=None
    if protocol.get('replay_distillation_weight',0)>0:
        teacher=load_target_model(checkpoint,device).model.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    criterion=MaskedTwinLiteLoss(task_spec=ROAD_MARKING_TASK).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    start=0;prior_seconds=0.
    if a.resume:
        saved=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        if saved['protocol_sha256']!=sha256_file(a.protocol):raise ValueError('resume protocol changed')
        model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];prior_seconds=saved['elapsed_seconds']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    sampler=BalancedBatches(len(new),len(replay),protocol['new_per_batch'],protocol['replay_per_batch'],protocol['steps'],protocol['seed'],start)
    loader=DataLoader(CachedAdaptationSamples(a.cache,rows,verify=True),batch_sampler=sampler,num_workers=protocol.get('workers',4),pin_memory=True,prefetch_factor=2)
    amp=protocol['precision']=='BF16';began=time.monotonic();history=[]
    def save_progress(step):
        state=dict(model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,elapsed_seconds=prior_seconds+time.monotonic()-began,
                   protocol_sha256=sha256_file(a.protocol),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_checkpoint(out/'latest.pt',state)
    try:
        with (out/'training_steps.jsonl').open('a' if a.resume else 'x') as stream:
            for step,batch in enumerate(loader,start+1):
                image=batch['image'].to(device,non_blocking=True);targets={k:v.to(device,non_blocking=True) for k,v in batch['targets'].items()};valid=batch['valid_mask'].to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                progress=(step-1)/max(protocol['steps']-1,1)
                factor=protocol.get('minimum_lr_factor',1.)+(1-protocol.get('minimum_lr_factor',1.))*.5*(1+np.cos(np.pi*progress))
                if step<=protocol.get('warmup_steps',0):factor*=step/protocol['warmup_steps']
                for group in optimizer.param_groups:group['lr']=protocol['learning_rate']*factor
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):outputs=adapt_twinlite_outputs(model(image),'road_marking')
                # Reductions run in FP32, including focal/Tversky sums over many pixels.
                losses=criterion({k:v.float() for k,v in outputs.items()},targets,valid);loss=losses['loss']
                if teacher is not None:
                    n=protocol['new_per_batch']
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                        reference=adapt_twinlite_outputs(teacher(image[n:]),'road_marking')
                    penalty=masked_replay_distillation({k:v[n:].float() for k,v in outputs.items()},
                                                       {k:v.float() for k,v in reference.items()},valid[n:],protocol.get('distillation_temperature',2.))
                    losses['supervised_loss']=loss;losses['replay_distillation']=penalty
                    loss=loss+protocol['replay_distillation_weight']*penalty;losses['loss']=loss
                if not torch.isfinite(loss):raise ValueError('non-finite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip_norm'])
                if not torch.isfinite(norm):raise ValueError('non-finite gradients')
                optimizer.step();history.append(float(loss.detach()))
                elapsed=prior_seconds+time.monotonic()-began
                if step%20==0 or step==1:
                    record=dict(step=step,losses={k:float(v.detach()) for k,v in losses.items()},gradient_norm=float(norm),elapsed_seconds=elapsed,
                                learning_rate=optimizer.param_groups[0]['lr'],
                                memory_allocated=int(torch.cuda.max_memory_allocated()),sample_ids=[rows[i]['sample_id'] for i in batch['index'].tolist()])
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(a.arm,step,'loss',round(history[-1],6),'seconds',round(elapsed,1),flush=True)
                if step%protocol.get('checkpoint_every',500)==0:save_progress(step)
        if not all(torch.equal(v.cpu(),buffers[k]) for k,v in model.named_buffers() if k in buffers):raise ValueError('BatchNorm buffers changed')
        save_progress(protocol['steps'])
        state=dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},task_spec=ROAD_MARKING_TASK.as_dict(),
                   upstream_commit=loaded.upstream_commit,epoch=None,adaptation_steps=protocol['steps'],initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                   source_weights=loaded.weights,adaptation_protocol_sha256=sha256_file(a.protocol))
        atomic_checkpoint(out/'checkpoint.pt',state)
        logged=[json.loads(line) for line in (out/'training_steps.jsonl').read_text().splitlines() if line.strip()]
        write_json(out/'summary.json',dict(metadata,status='complete',elapsed_seconds=prior_seconds+time.monotonic()-began,
                   new_draws=0 if control else protocol['steps']*protocol['new_per_batch'],replay_draws=protocol['steps']*(protocol['replay_per_batch']+(protocol['new_per_batch'] if control else 0)),
                   first_loss=logged[0]['losses']['loss'],last_loss=logged[-1]['losses']['loss'],checkpoint_sha256=sha256_file(out/'checkpoint.pt'),batch_norm_buffers_unchanged=True))
        hashes={name:sha256_file(out/name) for name in ['checkpoint.pt','training_membership.json','protocol.json','summary.json','new_manifest.jsonl','replay_manifest.jsonl','training_steps.jsonl','source/retrain_streaming.py','source/adaptation_cache.py']}
        write_json(out/'run.json',dict(metadata,status='complete',checkpoint_sha256=hashes['checkpoint.pt'],artifact_sha256=hashes))
        (out/'_SUCCESS').write_text(sha256_file(out/'run.json')+'\n')
    except BaseException as exc:
        write_json(out/'run.json',dict(metadata,status='failed',error=str(exc)));raise


if __name__=='__main__':main()
