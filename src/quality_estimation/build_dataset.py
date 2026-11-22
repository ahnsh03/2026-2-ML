"""Build aligned X/y from a versioned front-camera dataset, without fitting a regressor.

Run: python3 -m quality_estimation.build_dataset --output-dir /outputs/<new-run>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from camera_semantic_perception.geometry import letterbox_bgr
from common.bridge import (DEFAULT_CHECKPOINT_SHA256, DEFAULT_DATASET_VERSION,
                           git_head, load_target_model, parse_condition,
                           resolve_checkpoint, sha256_file)
from twinlite_morai import MoraiTwinLiteDataset, ROAD_MARKING_TASK
from .features import FEATURE_NAMES, extract_features, feature_schema, geometric_valid
from .labels import build_label

REPO_ROOT = Path(__file__).resolve().parents[2]
# Reuse the ML repository's pure-Python capture palette helper, without ROS runtime.
sys.path.insert(0, str(REPO_ROOT / "ros/src/data_collection/src"))
from data_collection.mask_baker import semantic_unknown_colors  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_input(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("input manifest paths must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("input path escapes data root: " + relative) from exc
    return path


def read_image(root, relative, flags):
    path = resolve_input(root, relative)
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("image cannot be decoded: " + relative)
    return image, hashlib.sha256(data).hexdigest()


def color_channels(image, label):
    """Match the original training loader's BGR/BGRA policy; alpha is not an input."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(label + " must be a BGR/BGRA uint8 image")
    return image[:, :, :3]


def native_marking(semantic_bgr):
    if semantic_bgr.dtype != np.uint8 or semantic_bgr.ndim != 3 or semantic_bgr.shape[2] != 3:
        raise ValueError("Semantic image must be HxWx3 uint8")
    unknown = semantic_unknown_colors(semantic_bgr)
    if unknown:
        raise ValueError("unknown Semantic colors: " + str(unknown))
    if np.any(np.all(semantic_bgr == (255, 178, 0), axis=2)):
        raise ValueError("blue lane requires an explicit new target policy")
    target = np.zeros(semantic_bgr.shape[:2], dtype=np.uint8)
    for value, bgr in ((1, (255, 255, 255)), (2, (0, 255, 255)), (3, (0, 0, 255))):
        target[np.all(semantic_bgr == bgr, axis=2)] = value
    return target


def prepare_rows(root, version, splits, max_per_run=None, group_map=None):
    """Keep original exposure/split roles; do not invent a quality train/test split."""
    group_map = group_map or {}
    rows, source_hashes = [], {}
    datasets = []
    for split in splits:
        ds = MoraiTwinLiteDataset(str(root), version, split, verify_files=False)
        if ds.task_spec != ROAD_MARKING_TASK:
            raise ValueError("quality dataset requires the v6 four-class task")
        datasets.append(ds)
        manifest = root / "dataset_versions" / version / (split + ".jsonl")
        source_hashes[str(manifest.relative_to(root))] = sha256_file(manifest)
        candidates = [r for r in ds.rows if r["view"] == "front"]
        counts = Counter()
        for r in candidates:
            if max_per_run is not None and counts[r["run_id"]] >= max_per_run:
                continue
            counts[r["run_id"]] += 1
            rows.append(dict(r))
    if not rows:
        raise ValueError("no front-camera samples selected")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("sample IDs overlap across source splits")
    metadata = datasets[0].dataset_meta
    source_hashes[f"dataset_versions/{version}/dataset.json"] = sha256_file(
        root / "dataset_versions" / version / "dataset.json")
    policies = {r["run_id"]: {k: r.get(k) for k in ("policy_id", "policy_sha256", "cache_geometry_sha256", "cache_transform_sha256")}
                for r in metadata["runs"]}
    lineage = {}
    for run_id in sorted({r["run_id"] for r in rows}):
        run = resolve_input(root, "datasets/" + run_id)
        path = run / "manifest.jsonl"
        source_hashes[str(path.relative_to(root))] = sha256_file(path)
        frames = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                fid = int(item["frame_id"])
                if fid in frames:
                    raise ValueError("duplicate frame in capture lineage")
                if item.get("run_id", run_id) != run_id:
                    raise ValueError("capture lineage run mismatch")
                frames[fid] = item
        lineage[run_id] = frames
    for row in rows:
        run_id = row["run_id"]
        item = lineage[run_id][int(row["frame_id"])]
        image = str(Path("datasets") / run_id / item["paths"]["intensity"]["front"])
        if resolve_input(root, image) != resolve_input(root, row["image"]):
            raise ValueError("RGB and capture lineage do not match")
        row["semantic"] = str(Path("datasets") / run_id / item["paths"]["semantic"]["front"])
        row["capture_name"] = item.get("custom_name")
        row["capture_requested_at_utc"] = item.get("requested_at_utc")
        row["parent_session_id"] = group_map.get(run_id, row.get("parent_session_id", run_id))
        row["group_source"] = "explicit_group_map" if run_id in group_map else "source_parent_or_capture_run"
        row["target_policy"] = policies[run_id]
    return rows, source_hashes, metadata


class QualitySamples(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        bgr, rgb_sha = read_image(self.root, row["image"], cv2.IMREAD_UNCHANGED)
        bgr = color_channels(bgr, "RGB")
        model_input, geometry = letterbox_bgr(bgr, tuple(row["geometry"]["target_hw"]))
        for field in ("original_hw", "target_hw", "resized_hw", "pad_ltrb"):
            if tuple(row["geometry"][field]) != getattr(geometry, field):
                raise ValueError("manifest/decoded geometry mismatch: " + row["sample_id"])
        valid_geometry = geometric_valid(geometry)
        target_path = row["targets"]["road_marking"]
        valid_path = row.get("valid_masks", {}).get("road_marking", row["valid_mask"])
        target, target_sha = read_image(self.root, target_path, cv2.IMREAD_UNCHANGED)
        valid, valid_sha = read_image(self.root, valid_path, cv2.IMREAD_UNCHANGED)
        if (target.shape != geometry.target_hw or target.dtype != np.uint8
                or not np.isin(target, [0, 1, 2, 3]).all()):
            raise ValueError("invalid cached road-marking target")
        if valid.shape != target.shape or valid.dtype != np.uint8 or not np.isin(valid, [0, 1]).all():
            raise ValueError("invalid annotation validity mask")
        if np.any(valid.astype(bool) & ~valid_geometry):
            raise ValueError("annotation validity includes padding")
        semantic, semantic_sha = read_image(self.root, row["semantic"], cv2.IMREAD_UNCHANGED)
        semantic = color_channels(semantic, "Semantic")
        if semantic.shape != bgr.shape:
            raise ValueError("RGB/Semantic native shape mismatch")
        native = native_marking(semantic)
        h, w = geometry.resized_hw
        x, y, _, _ = geometry.pad_ltrb
        reconstructed = np.zeros(geometry.target_hw, dtype=np.uint8)
        reconstructed[y:y+h, x:x+w] = cv2.resize(native, (w, h), interpolation=cv2.INTER_NEAREST)
        if not np.array_equal(reconstructed[valid_geometry], target[valid_geometry]):
            raise ValueError("native Semantic and cached marking target differ: " + row["sample_id"])
        return {"index": index, "image": model_input, "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                "geometry": geometry, "target": target, "gt_valid": valid.astype(bool),
                "hashes": {"rgb": rgb_sha, "semantic": semantic_sha, "target": target_sha, "gt_valid": valid_sha}}


def collate(items):
    return torch.from_numpy(np.stack([item["image"] for item in items])), items


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("PERCEPTION_DATA", "/data"))
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--split", action="append", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--checkpoint")
    parser.add_argument("--twinlite-root")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-per-run", type=int, help="Deterministic first N front frames per run, for smoke checks")
    parser.add_argument("--group-map", help="Optional JSON mapping capture run IDs to parent-session IDs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audit-per-run", type=int, default=1, help="Store float32 probability fixtures for first N frames per run")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.audit_per_run < 0 or (args.max_per_run is not None and args.max_per_run <= 0):
        raise ValueError("invalid batch/worker/sample settings")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    splits = args.split or ["train", "val"]
    if len(set(splits)) != len(splits):
        raise ValueError("duplicate source split")
    root, out = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 10 * 1024**3:
        raise RuntimeError("output volume has less than 10 GiB available")
    started = time.perf_counter()
    write_json(out / "run.json", {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args)})
    group_map = json.loads(Path(args.group_map).read_text()) if args.group_map else {}
    if not isinstance(group_map, dict) or any(not isinstance(v, str) or not v for v in group_map.values()):
        raise ValueError("group-map must map run IDs to nonempty strings")
    rows, sources, original_meta = prepare_rows(root, args.dataset_version, splits, args.max_per_run, group_map)
    checkpoint = resolve_checkpoint(args.checkpoint)
    if sha256_file(checkpoint) != DEFAULT_CHECKPOINT_SHA256:
        raise ValueError("this schema is frozen to the verified Medium v6 checkpoint")
    loaded = load_target_model(checkpoint, torch.device(args.device), twinlite_root=args.twinlite_root)
    if loaded.task_spec != ROAD_MARKING_TASK:
        raise ValueError("checkpoint task is not road_marking")
    schema = feature_schema()
    write_json(out / "schema.json", schema)
    write_json(out / "sample_ids.json", [r["sample_id"] for r in rows])
    write_json(out / "feature_names.json", list(FEATURE_NAMES))
    code = {str(p.relative_to(REPO_ROOT)): sha256_file(p)
            for folder in (REPO_ROOT / "src", REPO_ROOT / "third_party/TwinLiteNetPlus/model")
            for p in sorted(folder.rglob("*.py"))}
    code["ros/src/data_collection/src/data_collection/mask_baker.py"] = sha256_file(
        REPO_ROOT / "ros/src/data_collection/src/data_collection/mask_baker.py")
    metadata = {"schema_version": schema["schema_version"], "status": "running", "sample_count": len(rows),
                "feature_count": len(FEATURE_NAMES), "source_dataset": args.dataset_version,
                "source_manifest_sha256": sources, "source_split_policy": original_meta.get("split_policy"),
                "checkpoint_sha256": loaded.checkpoint_sha256, "weights": loaded.weights, "epoch": loaded.epoch,
                "upstream_commit": loaded.upstream_commit, "feature_schema_sha256": sha256_file(out / "schema.json"),
                "ml_repo_commit": git_head(REPO_ROOT), "source_sha256": code, "precision": "fp32",
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "device": args.device},
                "input_color_policy": "BGR/BGRA uint8 decoded; alpha discarded, matching the perception training loader",
                "quality_split": "unassigned; original perception train/val are exposure metadata, not quality splits",
                "grouping": "capture run IDs unless explicit parent group-map supplied; related-run grouping still requires review",
                "args": vars(args), "regression_training_performed": False}
    if args.group_map:
        metadata["group_map_sha256"] = sha256_file(Path(args.group_map))
    write_json(out / "dataset.json", metadata)
    loader = DataLoader(QualitySamples(root, rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, worker_init_fn=worker_init)
    X = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.full(len(rows), np.nan, dtype=np.float32)
    defined = np.zeros(len(rows), dtype=bool)
    (out / "audit").mkdir()
    audited, summaries = Counter(), {}
    label_fields = ["sample_id", "lane_iou", "label_defined", "undefined_reason", "tp", "fp", "fn", "union", "gt_pixels", "pred_pixels", "valid_pixels"]
    with (out / "features.csv").open("w", newline="", encoding="utf-8") as fx, \
            (out / "labels.csv").open("w", newline="", encoding="utf-8") as fy, \
            (out / "samples.jsonl").open("w", encoding="utf-8") as fm, torch.inference_mode():
        wx = csv.writer(fx); wx.writerow(["sample_id", *FEATURE_NAMES])
        wy = csv.DictWriter(fy, fieldnames=label_fields); wy.writeheader()
        complete = 0
        for images, items in loader:
            logits = loaded.model(images.to(args.device))[1]
            if logits.shape[1:] != (4, 384, 640):
                raise ValueError("unexpected model output dimensions")
            probabilities = logits.float().softmax(dim=1).cpu().numpy()
            for item, prob in zip(items, probabilities):
                index = item["index"]; row = rows[index]
                prediction = prob.argmax(axis=0).astype(np.uint8)
                # X is computed before labels, with no GT or metadata arguments.
                X[index] = extract_features(item["rgb"], prob, item["geometry"])
                label = build_label(prediction, item["target"], geometric_valid(item["geometry"]), item["gt_valid"])
                defined[index] = label["label_defined"]
                if label["label_defined"]:
                    y[index] = label["lane_iou"]
                wx.writerow([row["sample_id"], *X[index].tolist()])
                wy.writerow({"sample_id": row["sample_id"], **label})
                meta = {k: row[k] for k in ("sample_id", "run_id", "frame_id", "view", "split", "geometry", "image", "semantic", "capture_name", "capture_requested_at_utc", "parent_session_id", "group_source", "target_policy")}
                meta.update(row_index=index, input_sha256=item["hashes"], condition=parse_condition(row["run_id"]),
                            source_target=row["targets"]["road_marking"],
                            source_gt_valid=row.get("valid_masks", {}).get("road_marking", row["valid_mask"]),
                            quality_split=None, source_role={"train": "f0_train", "val": "historical_dev", "test": "source_test_unverified"}[row["split"]],
                            semantic_cache_parity=True)
                fm.write(json.dumps(meta, ensure_ascii=False, allow_nan=False) + "\n")
                stat = summaries.setdefault(row["run_id"], {"n": 0, "defined": 0, "iou_sum": 0.0, "source_split": row["split"]})
                stat["n"] += 1
                stat["defined"] += int(label["label_defined"])
                stat["iou_sum"] += label["lane_iou"] or 0.0
                if audited[row["run_id"]] < args.audit_per_run:
                    np.savez_compressed(out / "audit" / f"{index:06d}.npz", probabilities=prob,
                                        prediction=prediction, gt=item["target"], gt_valid=item["gt_valid"])
                    audited[row["run_id"]] += 1
                complete += 1
            if complete % (args.batch_size * 10) == 0 or complete == len(rows):
                fx.flush(); fy.flush(); fm.flush()
                print(f"{complete}/{len(rows)} frames; {time.perf_counter()-started:.1f}s", flush=True)
    if not np.isfinite(X).all() or np.any((y[defined] < 0) | (y[defined] > 1)):
        raise ValueError("invalid final arrays")
    np.save(out / "X.npy", X, allow_pickle=False)
    np.save(out / "y.npy", y, allow_pickle=False)
    np.save(out / "label_defined.npy", defined, allow_pickle=False)
    for stat in summaries.values():
        stat["mean_iou"] = stat.pop("iou_sum") / stat["defined"] if stat["defined"] else None
    write_json(out / "summary.json", {"X_shape": list(X.shape), "y_shape": list(y.shape), "defined_labels": int(defined.sum()),
                                      "undefined_labels": int((~defined).sum()), "semantic_cache_parity_frames": len(rows),
                                      "finite_features": True, "runs": summaries, "elapsed_seconds": time.perf_counter()-started})
    metadata["status"] = "complete"
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["artifact_sha256"] = {p.name: sha256_file(p) for p in out.iterdir()
                                   if p.is_file() and p.name not in ("dataset.json", "run.json")}
    write_json(out / "dataset.json", metadata)
    write_json(out / "run.json", {"status": "complete", "dataset_metadata_sha256": sha256_file(out / "dataset.json")})
    (out / "_SUCCESS").write_text(sha256_file(out / "dataset.json") + "\n")
    print(f"Complete: X={X.shape}, y={y.shape}, defined={defined.sum()} -> {out}", flush=True)


if __name__ == "__main__":
    main()
