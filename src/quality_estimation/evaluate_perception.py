"""Compare frozen f0/f1 perception masks on identical native evaluation rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.bridge import DEFAULT_CHECKPOINT_SHA256, load_target_model, resolve_checkpoint, sha256_file
from .artifacts import write_json
from .build_dataset import worker_init
from .dataset_io import require
from .perception_data import NativePerceptionSamples, POLICY_SHA256, perception_batch


def _preserve_items(items):
    """Keep native arrays and per-frame provenance together across worker batches."""
    return items


from .perception_identity import load_adaptation


def counts(prediction,target,valid):
    pred=prediction & valid;truth=target & valid
    tp=int(np.count_nonzero(pred & truth));fp=int(np.count_nonzero(pred & ~truth));fn=int(np.count_nonzero(~pred & truth))
    union=tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/union if union else None)


def summarize(records):
    result={}
    for head in ("lane","drivable","white","yellow","stopline"):
        values=[r["metrics"][head]["iou"] for r in records if r["metrics"][head]["iou"] is not None]
        tp,fp,fn=(sum(r["metrics"][head][k] for r in records) for k in ("tp","fp","fn"))
        result[head]=dict(rows=len(records),defined=len(values),undefined=len(records)-len(values),
                          mean_frame_iou=float(np.mean(values)) if values else None,
                          pooled_pixel_iou=tp/(tp+fp+fn) if tp+fp+fn else None,tp=tp,fp=fp,fn=fn)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--adaptation-run",type=Path,help="Omit for the original frozen Medium v6 EMA")
    p.add_argument("--role",choices=("meta_dev","diagnostic_only","final_test"),required=True)
    p.add_argument("--freeze",type=Path,help="Final registry with perception_checkpoints and evaluation_parent_groups")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--num-workers",type=int,default=4,
                   help="Parallel CPU image/target preparation; 0 keeps serial preparation")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();require(a.batch_size>0 and a.num_workers>=0,"invalid batch/worker settings");require(not a.output_dir.exists(),"evaluation output already exists")
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    require(bool(rows) and len({r["sample_id"] for r in rows})==len(rows),"empty/duplicate evaluation IDs")
    require(all(r.get("source_role")=="after_f0_training" or a.role=="diagnostic_only" for r in rows),"exposed/unknown evaluation source")
    if a.role!="diagnostic_only":
        require(all(r.get("preregistered_quality_role")==a.role for r in rows),"evaluation differs from reserved collection roles")
    else:
        require(all(r.get("preregistered_quality_role")!="final_test" for r in rows),"reserved final cannot become diagnostic data")
    adaptation=None;membership=[]
    if a.adaptation_run:checkpoint,adaptation,membership=load_adaptation(a.adaptation_run)
    else:
        checkpoint=resolve_checkpoint();require(sha256_file(checkpoint)==DEFAULT_CHECKPOINT_SHA256,"f0 checkpoint differs")
    digest=sha256_file(checkpoint);parents={r["parent_session_id"] for r in rows}
    trained_ids={r["sample_id"] for r in membership};trained_pixels={r["pixel_sha256"] for r in membership}
    trained_parents={r["parent_session_id"] for r in membership}
    require(not parents.intersection(trained_parents),"evaluation parent used for adaptation")
    require(not {r["sample_id"] for r in rows}.intersection(trained_ids),"evaluation sample used for adaptation")
    freeze=None
    if a.role=="final_test":
        require(a.freeze is not None,"final perception evaluation requires a freeze")
        freeze=json.loads(a.freeze.read_text())
        require(freeze.get("target_policy_sha256")==POLICY_SHA256,"frozen perception target policy differs")
        require(freeze.get("evaluation_role")=="final_test","frozen perception role differs")
        require(digest in freeze["perception_checkpoints"].values(),"perception checkpoint is not frozen")
        require(parents==set(freeze["evaluation_parent_groups"]),"final parent groups differ from freeze")
        require(freeze["final_test_accessed_before_freeze"] is False,"registry declares prior final access")
        created=datetime.fromisoformat(freeze["created_utc"].replace("Z","+00:00"))
        require(created.tzinfo is not None and created<=datetime.now(timezone.utc),"invalid freeze time")
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False);(out/"audit").mkdir()
    metadata=dict(status="running",checkpoint_sha256=digest,role=a.role,target_policy_sha256=POLICY_SHA256,
                  manifest_sha256=sha256_file(a.manifest),rows=len(rows),freeze_sha256=sha256_file(a.freeze) if freeze else None,
                  batch_size=a.batch_size,num_workers=a.num_workers,
                  source_sha256=sha256_file(Path(__file__)),adaptation_run_sha256=sha256_file(a.adaptation_run/"run.json") if adaptation else None)
    write_json(out/"run.json",metadata);(out/"evaluator_source.py").write_bytes(Path(__file__).read_bytes())
    try:
        cv2.setNumThreads(1);torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu");loaded=load_target_model(checkpoint,device)
        source=NativePerceptionSamples(a.data_root,rows);records=[];audited=set();started=time.perf_counter()
        loader=DataLoader(source,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,
                          collate_fn=_preserve_items,worker_init_fn=worker_init,
                          persistent_workers=a.num_workers>0)
        with (out/"per_frame.jsonl").open("x") as stream,torch.inference_mode():
            for number,items in enumerate(loader):
                start=number*a.batch_size
                require([r["index"] for r in items]==list(range(start,min(start+a.batch_size,len(rows)))),
                        "evaluation worker batch order differs from manifest")
                require(not {r["pixel_sha256"] for r in items}.intersection(trained_pixels),"evaluation pixels used for adaptation")
                batch=perception_batch(items,device);outputs=loaded.model(batch["image"])
                da,mark=(v.argmax(1).cpu().numpy() for v in outputs)
                for offset,item in enumerate(items):
                    row=rows[start+offset];valid=item["valid_mask"];gt=item["targets"]["road_marking"];pred=mark[offset]
                    metrics={"lane":counts(np.isin(pred,[1,2]),np.isin(gt,[1,2]),valid),
                             "drivable":counts(da[offset]==1,item["targets"]["drivable"]==1,valid)}
                    for cls,head in ((1,"white"),(2,"yellow"),(3,"stopline")):metrics[head]=counts(pred==cls,gt==cls,valid)
                    record=dict(sample_id=row["sample_id"],parent_session_id=row["parent_session_id"],condition=row.get("condition"),
                                cohort=row.get("cohort"),capture_mode=row.get("capture_mode"),metrics=metrics,
                                input_sha256=item["hashes"],pixel_sha256=item["pixel_sha256"])
                    records.append(record);stream.write(json.dumps(record,allow_nan=False)+"\n")
                    if row["parent_session_id"] not in audited:
                        np.savez_compressed(out/"audit"/f"{start+offset:06d}.npz",prediction=pred,gt=gt,drivable_prediction=da[offset],
                                            drivable_gt=item["targets"]["drivable"],valid=valid)
                        audited.add(row["parent_session_id"])
                if (start//a.batch_size)%20==0:print(len(records),"/",len(rows),flush=True)
        strata={}
        for field in ("parent_session_id","condition","cohort","capture_mode"):
            groups=defaultdict(list)
            for record in records:groups[json.dumps(record[field],sort_keys=True)].append(record)
            strata[field]={key:summarize(values) for key,values in groups.items()}
        report=dict(metadata,status="complete",metrics=summarize(records),grouped_metrics=strata,elapsed_seconds=time.perf_counter()-started,
                    final_no_retuning=a.role=="final_test")
        write_json(out/"evaluation.json",report)
        hashes={str(f.relative_to(out)):sha256_file(f) for f in out.rglob("*") if f.is_file() and f.name!="run.json"}
        write_json(out/"run.json",dict(metadata,status="complete",artifact_sha256=hashes));(out/"_SUCCESS").write_text(sha256_file(out/"run.json")+"\n")
        print(report["metrics"],flush=True)
    except BaseException as exc:
        write_json(out/"run.json",dict(metadata,status="failed",error=str(exc)));raise


if __name__=="__main__":main()
