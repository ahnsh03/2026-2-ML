#!/usr/bin/env python3
"""After RGB selection freeze, independently rebuild ROI9 counts from raw GT."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.bridge import load_target_model,sha256_file
from quality_estimation.build_dataset import collate,native_marking
from quality_estimation.infer_images import RGBSamples
from quality_estimation.image_io import color_channels
from quality_estimation.perception_identity import resolve_quality_perception
from quality_estimation.review_targets import support_aware_targets
from quality_estimation.spatial_features import SpatialProfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("release","rgb-root","gt-root","source-index","queue","freeze","reference-pack","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args()
    frozen=json.loads(a.freeze.read_text())
    run=json.loads((a.queue/"run.json").read_text())
    if (sha256_file(a.queue/"candidates.jsonl")!=frozen["primary_candidates_sha256"]
            or sha256_file(a.queue/"run.json")!=frozen["primary_run_sha256"]
            or sha256_file(a.source_index)!=frozen["source_index_sha256"]):
        raise ValueError("selection or source index changed after freeze")
    if (a.queue/"_SUCCESS").read_text().strip()!=sha256_file(a.queue/"run.json"):
        raise ValueError("RGB run incomplete")
    rows=json.loads(a.source_index.read_text())["confirmation"]["rows"]
    predictions=[json.loads(x)for x in (a.queue/"predictions.jsonl").read_text().splitlines()]
    if [r["sample_id"]for r in rows]!=[r["sample_id"]for r in predictions]:
        raise ValueError("source and frozen prediction order differ")
    checkpoint,identity,_=resolve_quality_perception(a.release/"models/perception",None)
    if identity!=run["perception_identity"]:
        raise ValueError("GT recomputation belongs to a different perception version")
    with np.load(a.reference_pack,allow_pickle=False) as ref:
        reference_counts=ref["counts"].copy()
        if ref["ids"].tolist()!=[r["sample_id"]for r in rows]:
            raise ValueError("reference count order differs")
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(a.device)
    model=load_target_model(checkpoint,device,weights="model")
    profile=SpatialProfile.load()
    if profile.regions_sha256!=run["regions_sha256"]:
        raise ValueError("GT ROI profile differs from the frozen replay")
    regions=profile.native[::2,::2]
    loader=DataLoader(RGBSamples(a.rgb_root,rows),batch_size=a.batch_size,num_workers=2,collate_fn=collate,shuffle=False)
    counts=[]
    with torch.inference_mode():
        for images,items in loader:
            _,mark=model.model(images.to(device))
            masks=np.isin(mark.float().softmax(1).argmax(1)[:,12:372].cpu().numpy(),[1,2])
            for item,lane in zip(items,masks):
                index=item["index"];row=rows[index]
                if item["pixel_sha256"]!=row["pixel_sha256"]:
                    raise ValueError("source RGB pixel hash differs")
                file=(a.gt_root/row["semantic"]).resolve();file.relative_to(a.gt_root.resolve())
                if sha256_file(file)!=row["expected_sha256"]["semantic"]:
                    raise ValueError("raw Semantic checksum differs")
                sem=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
                if sem is not None:
                    sem=color_channels(sem,"Semantic")
                if sem is None or sem.shape!=(720,1280,3):
                    raise ValueError("Semantic shape differs")
                truth=np.isin(native_marking(sem)[::2,::2],[1,2])
                events=(lane&truth,lane&~truth,~lane&truth)
                counts.append(np.stack([np.bincount(regions[e],minlength=9)for e in events],axis=1))
            if len(counts)%100==0:
                print("Raw GT counts rebuilt",len(counts),flush=True)
    counts=np.asarray(counts,dtype=np.int64)
    targets=support_aware_targets(counts)
    gt_pixels=counts[:,:,0]+counts[:,:,2]
    reference_gt=reference_counts[:,:,0]+reference_counts[:,:,2]
    if not np.array_equal(gt_pixels,reference_gt):
        raise ValueError("raw Semantic ROI support differs from official targets")
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/"recomputed_counts.npz",ids=np.array([r["sample_id"]for r in rows]),
        counts=counts,y=targets["whole_frame_iou"],near_errors=targets["near"]["error"])
    mismatch=np.flatnonzero(np.any(counts!=reference_counts,axis=(1,2)))
    selected={json.loads(x)["sample_id"]for x in (a.queue/"candidates.jsonl").read_text().splitlines()}
    chosen=np.array([r["sample_id"]in selected for r in rows])
    report=dict(rows=len(rows),perception_identity=identity,batch_size=a.batch_size,device=str(device),
        gt_support_exact_all_rows=True,count_exact_rows=len(rows)-len(mismatch),
        count_mismatched_ids=[rows[i]["sample_id"]for i in mismatch],
        max_count_difference=int(np.max(abs(counts-reference_counts))),
        whole_iou_reference_max_difference=float(np.nanmax(abs(targets["whole_frame_iou"]-support_aware_targets(reference_counts)["whole_frame_iou"]))),
        near_errors=int(targets["near"]["error"].sum()),selected=int(chosen.sum()),
        selected_near_errors=int((chosen&targets["near"]["error"]).sum()),
        source_sha256=sha256_file(Path(__file__)),raw_gt_opened_only_after_selection_freeze=True,
        reference_pack_sha256=sha256_file(a.reference_pack),freeze_sha256=sha256_file(a.freeze))
    (a.output/"run.json").write_text(json.dumps(report,indent=2)+"\n")
    (a.output/"_SUCCESS").write_text(sha256_file(a.output/"run.json")+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
