"""Recompute fixed-v6 nine-region candidates and offline spatial labels from registered RGB/GT."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json, time
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from common.bridge import DEFAULT_CHECKPOINT_SHA256, load_target_model, sha256_file
from quality_estimation.build_dataset import collate, worker_init
from quality_estimation.labels import build_label
from quality_estimation.native import NativeSamples
from quality_estimation.region_candidates import candidate_schema, extract_candidates
from quality_estimation.road_structure_features import (
    extract_road_structure,
    ordered_features,
)
from quality_estimation.roi10 import reduce_heads
from quality_estimation.spatial_features import (
    SpatialProfile,
    UNCERTAINTY_FEATURES,
    extract_global_uncertainty,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--keys", nargs="+")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert sha256_file(a.checkpoint) == DEFAULT_CHECKPOINT_SHA256
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    loaded = load_target_model(a.checkpoint, torch.device("cuda"), weights="ema")
    profile = SpatialProfile.load()
    native = profile.native[::2, ::2]
    schema = candidate_schema(profile)
    schema.pop("histograms")
    schema.update(
        version="roi9-gp-candidates-20261125-v1",
        uncertainty=list(UNCERTAINTY_FEATURES),
        morphology=ordered_features(),
    )
    schema_text = json.dumps(schema, indent=2) + "\n"
    schema_path = a.output / "schema.json"
    if schema_path.exists():
        assert schema_path.read_text() == schema_text
    else:
        schema_path.write_text(schema_text)
    index = json.loads(a.index.read_text())
    keys = a.keys or list(index)
    for key in keys:
        dest = a.output / (key + ".npz")
        if dest.exists():
            assert (
                sha256_file(dest)
                == json.loads(dest.with_suffix(".json").read_text())["sha256"]
            )
            print("verified", key, flush=True)
            continue
        start = time.monotonic()
        records = index[key]["rows"]
        dataset = NativeSamples(a.data_root, records)
        loader = DataLoader(
            dataset,
            batch_size=a.batch_size,
            num_workers=4,
            collate_fn=collate,
            worker_init_fn=worker_init,
            pin_memory=True,
        )
        results = []

        def one(item):
            d, p, record = item
            heads = reduce_heads(d, p, record["geometry"])
            features = extract_candidates(
                record["rgb"], heads, profile, include_histograms=False
            )
            features["uncertainty"] = extract_global_uncertainty(p[:, 12:372])
            features["morphology"] = extract_road_structure(p[:, 12:372], profile)
            # GT is attached only after the RGB/output-only feature vector is complete.
            truth = build_label(
                p.argmax(0).astype(np.uint8),
                record["target"],
                record["gt_valid"],
                record["gt_valid"],
            )
            pred = np.isin(p[:, 12:372].argmax(0), (1, 2))
            gt = np.isin(record["target"][12:372], (1, 2))
            counts = np.stack(
                [
                    np.bincount(native[m], minlength=9)
                    for m in [pred & gt, pred & ~gt, ~pred & gt]
                ],
                1,
            )
            assert counts.sum(0).tolist() == [truth["tp"], truth["fp"], truth["fn"]]
            return (
                features,
                counts,
                truth["lane_iou"] if truth["label_defined"] else np.nan,
                record["pixel_sha256"],
            )

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool:
            for images, items in loader:
                drive, mark = loaded.model(images.cuda(non_blocking=True))
                ds = drive.argmax(1).cpu().numpy()
                ps = mark.float().softmax(1).cpu().numpy()
                results.extend(pool.map(one, zip(ds, ps, items)))
        assert len(results) == len(records)
        arrays = {
            name: np.stack([r[0][name] for r in results]) for name in results[0][0]
        }
        arrays.update(
            counts=np.stack([r[1] for r in results]),
            y=np.asarray([r[2] for r in results], np.float32),
            ids=np.asarray([r["sample_id"] for r in records]),
            pixel_sha256=np.asarray([r[3] for r in results]),
        )
        for name in ["group", "site", "cohort", "weather", "hour", "xy"]:
            arrays[name] = np.asarray([r[name] for r in records])
        for name in ["rgb", "output", "relations", "uncertainty", "morphology"]:
            assert np.isfinite(arrays[name]).all()
        temp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(dest)
        receipt = dict(
            n=len(records),
            sha256=sha256_file(dest),
            schema_sha256=sha256_file(schema_path),
            checkpoint_sha256=loaded.checkpoint_sha256,
            index_sha256=sha256_file(a.index),
            source_sha256=sha256_file(Path(__file__)),
            torch=torch.__version__,
            numpy=np.__version__,
            opencv=cv2.__version__,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            simulator_release=index[key]["simulator_release"],
            seconds=time.monotonic() - start,
        )
        dest.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(key, len(records), round(receipt["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
