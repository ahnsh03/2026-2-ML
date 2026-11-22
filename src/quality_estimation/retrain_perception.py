"""Fixed-budget f0-to-f1 pilot on selected candidate labels plus original replay."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import warnings

import cv2
import numpy as np
import torch

from common.bridge import DEFAULT_CHECKPOINT_SHA256, load_target_model, resolve_checkpoint, sha256_file
from twinlite_morai import MaskedTwinLiteLoss, ROAD_MARKING_TASK, twinlite_training_step, adapt_twinlite_outputs
from .artifacts import write_json
from .build_dataset import REPO_ROOT
from .dataset_io import require
from .distillation import distillation_settings, masked_replay_distillation
from .perception_data import (NativePerceptionSamples, POLICY_SHA256, SELECTION_ARMS, perception_batch,
                              validate_adaptation_roles, validate_adaptation_selection)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-manifest", type=Path, required=True)
    p.add_argument("--replay-manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", choices=(*SELECTION_ARMS, "ReplayOnly"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint")
    a = p.parse_args()
    require(not a.output_dir.exists(), "adaptation output already exists")
    protocol = json.loads(a.protocol.read_text())
    distillation = distillation_settings(protocol)
    require(a.seed in protocol["seeds"] and a.arm in protocol["selection_arms"], "arm/seed not preregistered")
    require(protocol["optimizer"] == "AdamW" and protocol["augmentation"].startswith("none"), "unsupported adaptation protocol")
    n_new, n_replay, steps = protocol["new_per_batch"], protocol["replay_per_batch"], protocol["steps"]
    require(all(isinstance(v,int) and v>0 for v in (n_replay,steps)) and isinstance(n_new,int) and n_new>=0, "invalid fixed training budget")
    require((n_new==0)==(a.arm=="ReplayOnly"),"zero new-data draws are only valid for the replay-only control")
    if a.arm=="ReplayOnly":
        require(protocol.get("replacement_replay_per_batch")==4 and protocol.get("common_replay_per_batch")==4 and n_replay==8,
                "replay control must preserve four common replay slots and replace four new-data slots")
    require(protocol["batch_size"] == n_new+n_replay, "batch budget mismatch")
    require(protocol.get("allowed_nondeterministic_operations")==["adaptive_avg_pool2d_backward_cuda"],
            "explicit pooling-backward numerical limitation must be recorded in the protocol")
    new_rows = [json.loads(s) for s in a.new_manifest.read_text().splitlines() if s.strip()]
    replay_rows = [json.loads(s) for s in a.replay_manifest.read_text().splitlines() if s.strip()]
    validate_adaptation_roles(new_rows,replay_rows,replay_only=a.arm=="ReplayOnly")
    validate_adaptation_selection(new_rows,a.arm,a.seed,protocol["new_budget_fraction"])
    checkpoint = resolve_checkpoint(a.checkpoint)
    require(sha256_file(checkpoint) == DEFAULT_CHECKPOINT_SHA256, "adaptation must start from frozen Medium v6 EMA")
    out = a.output_dir; out.mkdir(parents=True,exist_ok=False)
    for source,name in ((a.protocol,"protocol.json"),(a.new_manifest,"new_manifest.jsonl"),(a.replay_manifest,"replay_manifest.jsonl")):
        shutil.copyfile(source,out/name)
    source_hashes = {}
    for folder in (REPO_ROOT/"src",REPO_ROOT/"third_party/TwinLiteNetPlus/model"):
        for source in sorted(folder.rglob("*.py")):
            relative = source.relative_to(REPO_ROOT); destination=out/"source"/relative
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
            source_hashes[str(relative)] = sha256_file(destination)
    source=REPO_ROOT/"ros/src/data_collection/src/data_collection/mask_baker.py"
    destination=out/"source"/source.relative_to(REPO_ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    source_hashes[str(source.relative_to(REPO_ROOT))]=sha256_file(destination)
    metadata = dict(status="running",schema_version="ml-perception-adaptation-1",created_utc=datetime.now(timezone.utc).isoformat(),
                    arm=a.arm,seed=a.seed,initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,target_policy_sha256=POLICY_SHA256,
                    new_rows=len(new_rows),replay_rows=len(replay_rows),steps=steps,new_draws=steps*n_new,replay_draws=steps*n_replay,
                    protocol_sha256=sha256_file(a.protocol),source_sha256=source_hashes,selection="fixed optimizer-step endpoint; no evaluation-based checkpoint selection",
                    train_parent_groups=sorted({r["parent_session_id"] for r in new_rows+replay_rows}),
                    final_test_accessed=False,quality_regressor_recalibrated_for_f1=False,
                    strict_bitwise_determinism=False,allowed_nondeterministic_operations=protocol["allowed_nondeterministic_operations"])
    metadata['replay_distillation']=distillation
    write_json(out/"run.json",metadata)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Small fixed candidate/replay pools are cached to avoid repeated raw decoding.
        dataset=NativePerceptionSamples(a.data_root,new_rows+replay_rows)
        cached=[dataset[i] for i in range(len(dataset))]
        require(len({r["pixel_sha256"] for r in cached}) == len(cached), "duplicate source pixels in adaptation input")
        write_json(out/"training_membership.json",[dict(sample_id=row["sample_id"],parent_session_id=row["parent_session_id"],
                   pixel_sha256=item["pixel_sha256"],input_sha256=item["hashes"]) for row,item in zip(new_rows+replay_rows,cached)])
        teacher_cache={};teacher_digest=hashlib.sha256()
        if distillation:
            # Cache a frozen teacher before loading the student, avoiding two resident GPU models.
            teacher=load_target_model(checkpoint,device).model.eval()
            with torch.inference_mode():
                for start in range(len(new_rows),len(cached),4):
                    items=cached[start:start+4];teacher_batch=perception_batch(items,device)
                    targets=adapt_twinlite_outputs(teacher(teacher_batch['image']),'road_marking')
                    for j in range(len(items)):
                        teacher_cache[start+j]={head:value[j].detach().cpu().clone() for head,value in targets.items()}
                        for head in ('drivable','road_marking'):
                            teacher_digest.update(teacher_cache[start+j][head].numpy().tobytes())
            del teacher,teacher_batch,targets
            if device.type=='cuda':torch.cuda.empty_cache()
            # Teacher initialization must not change the original student's RNG stream.
            torch.manual_seed(a.seed)
            metadata['replay_teacher_logits_sha256']=teacher_digest.hexdigest()
            write_json(out/'run.json',metadata)
        loaded=load_target_model(checkpoint,device);require(loaded.task_spec==ROAD_MARKING_TASK,"adaptation task mismatch")
        model=loaded.model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        frozen_bn={name:value.detach().cpu().clone() for name,value in model.named_buffers()
                   if name.endswith(("running_mean","running_var","num_batches_tracked"))}
        criterion=MaskedTwinLiteLoss(task_spec=ROAD_MARKING_TASK).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["learning_rate"],weight_decay=protocol["weight_decay"])
        # Separate RNGs keep replay draws identical between arms with the same seed.
        new_rng=np.random.default_rng(a.seed);replay_rng=np.random.default_rng(a.seed+100000)
        started=time.perf_counter();loss_history=[];numerical_warnings=set()
        with (out/"training_steps.jsonl").open("x") as stream:
            for step in range(1,steps+1):
                if a.arm=="ReplayOnly":
                    new_indices=new_rng.integers(0,len(cached),size=4)
                    replay_indices=replay_rng.integers(0,len(cached),size=4)
                else:
                    new_indices=new_rng.integers(0,len(new_rows),size=n_new)
                    replay_indices=replay_rng.integers(len(new_rows),len(cached),size=n_replay)
                indices=np.concatenate([new_indices,replay_indices]);batch=perception_batch([cached[i] for i in indices],device)
                optimizer.zero_grad(set_to_none=True)
                outputs,losses=twinlite_training_step(model,batch,criterion)
                loss=losses["loss"]
                if distillation:
                    first=0 if a.arm=='ReplayOnly' else n_new
                    teacher_outputs={head:torch.stack([teacher_cache[int(i)][head] for i in indices[first:]]).to(device)
                                     for head in ('drivable','road_marking')}
                    penalty=masked_replay_distillation({head:value[first:] for head,value in outputs.items()},
                              teacher_outputs,batch['valid_mask'][first:],distillation['temperature'])
                    losses['supervised_loss']=loss
                    losses['replay_distillation']=penalty
                    loss=loss+distillation['weight']*penalty
                    losses['loss']=loss
                require(bool(torch.isfinite(loss)),"non-finite adaptation loss")
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    loss.backward()
                for warning in observed:
                    message=str(warning.message)
                    if "deterministic implementation" in message:
                        require("adaptive_avg_pool2d_backward_cuda" in message,"unexpected nondeterministic training operation: "+message)
                    numerical_warnings.add(message)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["gradient_clip_norm"])
                require(bool(torch.isfinite(norm)),"non-finite adaptation gradients")
                optimizer.step()
                record=dict(step=step,losses={k:float(v.detach().cpu()) for k,v in losses.items()},gradient_norm=float(norm.detach().cpu()),
                            sample_ids=[(new_rows+replay_rows)[i]["sample_id"] for i in indices],elapsed_seconds=time.perf_counter()-started)
                stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush();loss_history.append(float(loss.detach().cpu()))
                if step%20==0 or step==steps:print(f"step {step}/{steps}: loss={loss_history[-1]:.6f}; {time.perf_counter()-started:.1f}s",flush=True)
        require(all(torch.equal(value.detach().cpu(),frozen_bn[name]) for name,value in model.named_buffers() if name in frozen_bn),
                "frozen BatchNorm running statistics changed")
        state=dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},task_spec=ROAD_MARKING_TASK.as_dict(),
                   upstream_commit=loaded.upstream_commit,epoch=None,adaptation_steps=steps,initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                   source_weights=loaded.weights,adaptation_protocol_sha256=sha256_file(a.protocol))
        torch.save(state,out/"checkpoint.pt")
        write_json(out/"numerical_warnings.json",sorted(numerical_warnings))
        write_json(out/"summary.json",dict(metadata,status="complete",elapsed_seconds=time.perf_counter()-started,
                   first_loss=loss_history[0],last_loss=loss_history[-1],batch_norm_buffers_unchanged=True,
                   checkpoint_sha256=sha256_file(out/"checkpoint.pt")))
        hashes={str(f.relative_to(out)):sha256_file(f) for f in sorted(out.rglob("*")) if f.is_file() and f.name!="run.json"}
        write_json(out/"run.json",dict(metadata,status="complete",artifact_sha256=hashes))
        (out/"_SUCCESS").write_text(sha256_file(out/"run.json")+"\n")
        print("Completed fixed-budget adaptation",a.arm,a.seed,flush=True)
    except BaseException as exc:
        write_json(out/"run.json",dict(metadata,status="failed",error=str(exc)));raise


if __name__=="__main__":
    main()
