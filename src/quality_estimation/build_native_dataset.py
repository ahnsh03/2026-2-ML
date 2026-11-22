"""Generate X/y from an explicitly curated native front RGB/Semantic manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.bridge import DEFAULT_CHECKPOINT_SHA256, git_head, load_target_model, resolve_checkpoint, sha256_file
from .artifacts import FeatureDatasetWriter
from .build_dataset import REPO_ROOT, collate, worker_init
from .features import extract_features, geometric_valid
from .labels import build_label
from .native import NativeSamples
from .perception_identity import resolve_quality_perception


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--adaptation-run", type=Path, help="Explicit hash-checked updated perception run")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--audit-per-run", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.audit_per_run < 1:
        raise ValueError("invalid batch/worker/audit settings")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(20261116)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("empty manifest or duplicate sample IDs")
    required = ("sample_id", "run_id", "parent_session_id", "view", "image", "semantic", "source_role")
    for row in rows:
        if any(not row.get(k) for k in required):
            raise ValueError("missing required lineage: " + str(row.get("sample_id")))
    output_parent = args.output_dir.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_parent).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB available on output volume")
    checkpoint, identity, membership = resolve_quality_perception(args.adaptation_run, args.checkpoint)
    used_ids = {r["sample_id"] for r in membership}
    used_parents = {r["parent_session_id"] for r in membership}
    used_pixels = {r["pixel_sha256"] for r in membership}
    if used_ids.intersection(r["sample_id"] for r in rows) or used_parents.intersection(r["parent_session_id"] for r in rows):
        raise ValueError("quality data overlaps perception adaptation training")
    model = load_target_model(checkpoint, torch.device(args.device))
    metadata = {"input_kind": "native_semantic", "input_manifest": str(args.manifest),
                "input_manifest_sha256": sha256_file(args.manifest), "input_count": len(rows),
                "data_root": str(args.data_root), "checkpoint_sha256": model.checkpoint_sha256,
                "perception_identity": identity,
                "weights": model.weights, "epoch": model.epoch, "upstream_commit": model.upstream_commit,
                "ml_repo_commit": git_head(REPO_ROOT), "precision": "fp32; tf32 off; deterministic cudnn",
                "source_sha256": {str(f.relative_to(REPO_ROOT)): sha256_file(f)
                    for folder in (REPO_ROOT / "src", REPO_ROOT / "third_party/TwinLiteNetPlus/model")
                    for f in sorted(folder.rglob("*.py"))},
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "device": args.device},
                "target_policy": "visible white/yellow/stopline; actor-negative; geometry valid; blue or unknown rejected",
                "quality_split": "unassigned; split manifest is a separate immutable artifact", "args": vars(args)}
    metadata["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata["source_sha256"]["ros/src/data_collection/src/data_collection/mask_baker.py"] = sha256_file(REPO_ROOT / "ros/src/data_collection/src/data_collection/mask_baker.py")
    loader = DataLoader(NativeSamples(args.data_root, rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, worker_init_fn=worker_init)
    started = time.perf_counter()
    with FeatureDatasetWriter(args.output_dir, metadata, args.audit_per_run) as writer, torch.inference_mode():
        for relative, digest in metadata["source_sha256"].items():
            destination = args.output_dir / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / relative, destination)
            if sha256_file(destination) != digest:
                raise ValueError("dataset source changed while being archived")
        count = 0
        for images, items in loader:
            logits = model.model(images.to(args.device))[1]
            if logits.shape[1:] != (4, 384, 640):
                raise ValueError("unexpected road marking output shape")
            probabilities = logits.float().softmax(1).cpu().numpy()
            for item, prob in zip(items, probabilities):
                row = rows[item["index"]]
                if item["pixel_sha256"] in used_pixels:
                    raise ValueError("quality data pixels used for perception adaptation")
                prediction = prob.argmax(0).astype(np.uint8)
                x = extract_features(item["rgb"], prob, item["geometry"])
                label = build_label(prediction, item["target"], geometric_valid(item["geometry"]), item["gt_valid"])
                sample = dict(row, geometry=asdict(item["geometry"]), input_sha256=item["hashes"],
                              pixel_sha256=item["pixel_sha256"], dhash64=item["dhash64"],
                              quality_split=None, semantic_palette_verified=True)
                fixture = dict(probabilities=prob, prediction=prediction, gt=item["target"], gt_valid=item["gt_valid"])
                writer.add(x, label, sample, fixture)
                count += 1
            if count % (args.batch_size * 10) == 0 or count == len(rows):
                print(f"{count}/{len(rows)}; {time.perf_counter()-started:.1f}s", flush=True)
        print(json.dumps(writer.finalize({"native_gt_frames": count})), flush=True)


if __name__ == "__main__":
    main()
