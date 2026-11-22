"""Check an X/y artifact, its one-to-one joins, source fixtures and independent counts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label as connected_labels
from scipy.special import entr

from camera_semantic_perception.geometry import compute_letterbox_geometry
from common.bridge import sha256_file
from .features import FEATURE_NAMES, extract_features, geometric_valid
from .build_dataset import QualitySamples
from .dataset_io import load_dataset, require
from .native import NativeSamples
from .perception_identity import identity_from_adaptation


def verify(dataset_dir, data_root, perception_identity=None):
    out, root = Path(dataset_dir), Path(data_root)
    # Every audit remains active under python -O.
    load_dataset(out, perception_identity=perception_identity)
    meta = json.loads((out / "dataset.json").read_text())
    require(meta["status"] == "complete", 'verification failed: meta["status"] == "complete"')
    require((out / "_SUCCESS").read_text().strip() == sha256_file(out / "dataset.json"), 'verification failed: (out / "_SUCCESS").read_text().strip() == sha256_file(out / "dataset.json")')
    for name, digest in meta["artifact_sha256"].items():
        require(sha256_file(out / name) == digest, name)
    for name, digest in meta.get("source_manifest_sha256", {}).items():
        require(sha256_file(root / name) == digest, name)
    if meta.get("input_kind") == "native_semantic":
        require(sha256_file(Path(meta["input_manifest"])) == meta["input_manifest_sha256"], 'verification failed: sha256_file(Path(meta["input_manifest"])) == meta["input_manifest_sha256"]')
    ids = json.loads((out / "sample_ids.json").read_text())
    names = json.loads((out / "feature_names.json").read_text())
    require(names == list(FEATURE_NAMES), 'verification failed: names == list(FEATURE_NAMES)')
    require(len(ids) == len(set(ids)), 'verification failed: len(ids) == len(set(ids))')
    X = np.load(out / "X.npy", allow_pickle=False)
    y = np.load(out / "y.npy", allow_pickle=False)
    defined = np.load(out / "label_defined.npy", allow_pickle=False)
    require(X.shape == (len(ids), 56) and X.dtype == np.float32 and np.isfinite(X).all(), 'verification failed: X.shape == (len(ids), 56) and X.dtype == np.float32 and np.isfinite(X).all()')
    require(y.shape == defined.shape == (len(ids),) and y.dtype == np.float32 and defined.dtype == bool, 'verification failed: y.shape == defined.shape == (len(ids),) and y.dtype == np.float32 and defined.dtype == bool')
    require(np.isfinite(y[defined]).all() and np.isnan(y[~defined]).all(), 'verification failed: np.isfinite(y[defined]).all() and np.isnan(y[~defined]).all()')
    with (out / "features.csv").open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["sample_id", *names], 'verification failed: reader.fieldnames == ["sample_id", *names]')
        feature_rows = list(reader)
    with (out / "labels.csv").open() as stream:
        label_rows = list(csv.DictReader(stream))
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    for rows in (feature_rows, label_rows, samples):
        require([r["sample_id"] for r in rows] == ids, 'verification failed: [r["sample_id"] for r in rows] == ids')
    X_csv = np.asarray([[float(r[k]) for k in names] for r in feature_rows], dtype=np.float32)
    np.testing.assert_array_equal(X_csv, X)
    for i, label in enumerate(label_rows):
        tp, fp, fn, union = (int(label[k]) for k in ("tp", "fp", "fn", "union"))
        require(tp >= 0 and fp >= 0 and fn >= 0 and union == tp + fp + fn, 'verification failed: tp >= 0 and fp >= 0 and fn >= 0 and union == tp + fp + fn')
        require(defined[i] == (label["label_defined"] == "True") == (union > 0), 'verification failed: defined[i] == (label["label_defined"] == "True") == (union > 0)')
        require(int(label["gt_pixels"]) == tp + fn and int(label["pred_pixels"]) == tp + fp, 'verification failed: int(label["gt_pixels"]) == tp + fn and int(label["pred_pixels"]) == tp + fp')
        if defined[i]:
            require(abs(float(y[i]) - tp / union) <= 3e-8, 'verification failed: abs(float(y[i]) - tp / union) <= 3e-8')
            require(abs(float(label["lane_iou"]) - tp / union) < 1e-14, 'verification failed: abs(float(label["lane_iou"]) - tp / union) < 1e-14')
        else:
            require(label["lane_iou"] == "", 'verification failed: label["lane_iou"] == ""')
        require(samples[i]["row_index"] == i and samples[i]["quality_split"] is None, 'verification failed: samples[i]["row_index"] == i and samples[i]["quality_split"] is None')
        if meta.get("input_kind") == "native_semantic":
            require(samples[i]["semantic_palette_verified"] is True, 'verification failed: samples[i]["semantic_palette_verified"] is True')
        else:
            require(samples[i]["semantic_cache_parity"] is True, 'verification failed: samples[i]["semantic_cache_parity"] is True')
    constants = [names[i] for i in range(len(names)) if np.ptp(X[:, i]) == 0]
    duplicate_groups = defaultdict(list)
    for sample in samples:
        duplicate_groups[sample["input_sha256"]["rgb"]].append(sample)
    duplicates = [rs for rs in duplicate_groups.values() if len(rs) > 1]
    crossing = [rs for rs in duplicates if len({s["source_role"] for s in rs}) > 1]
    fixture_results = []
    for fixture in sorted((out / "audit").glob("*.npz")):
        index = int(fixture.stem); sample = samples[index]
        if meta.get("input_kind") == "native_semantic":
            item = NativeSamples(root, [sample])[0]
        else:
            row = {**sample, "targets": {"road_marking": sample["source_target"]},
                   "valid_mask": sample["source_gt_valid"], "valid_masks": {"road_marking": sample["source_gt_valid"]}}
            item = QualitySamples(root, [row])[0]
        require(item["hashes"] == sample["input_sha256"], 'verification failed: item["hashes"] == sample["input_sha256"]')
        with np.load(fixture, allow_pickle=False) as f:
            probabilities, pred, gt, valid = (f[k] for k in ("probabilities", "prediction", "gt", "gt_valid"))
        np.testing.assert_array_equal(gt, item["target"])
        np.testing.assert_array_equal(valid, item["gt_valid"])
        np.testing.assert_array_equal(pred, probabilities.argmax(0))
        geometry = item["geometry"]
        mask = geometric_valid(geometry) & valid
        p = (np.isin(pred, [1, 2]) & mask).astype(np.uint8)
        g = (np.isin(gt, [1, 2]) & mask).astype(np.uint8)
        intersection = cv2.countNonZero(cv2.bitwise_and(p, g))
        union = cv2.countNonZero(cv2.bitwise_or(p, g))
        require(intersection == int(label_rows[index]["tp"]), 'verification failed: intersection == int(label_rows[index]["tp"])')
        require(union == int(label_rows[index]["union"]), 'verification failed: union == int(label_rows[index]["union"])')
        np.testing.assert_allclose(extract_features(item["rgb"], probabilities, geometry), X[index], rtol=0, atol=1e-7)
        changed = probabilities.copy(); changed[:, ~geometric_valid(geometry)] = .25
        np.testing.assert_array_equal(extract_features(item["rgb"], changed, geometry), X[index])
        # Independent all-region formulas, directly from source RGB/probabilities.
        # SciPy connectivity and entropy provide separate implementations.
        x0, y0, _, _ = geometry.pad_ltrb; hh, ww = geometry.resized_hw
        q = probabilities[:, y0:y0+hh, x0:x0+ww].astype(np.float64)
        fg = np.isin(q.argmax(0), [1, 2]); gray = cv2.cvtColor(item["rgb"], cv2.COLOR_RGB2GRAY)
        entropy = entr(q).sum(0) / np.log(4)
        edges = cv2.Canny(gray,100,200,apertureSize=3,L2gradient=True) > 0
        regions = [(slice(None),slice(None),slice(None),slice(None))]
        for rr in range(3):
            for cc in range(2):
                regions.append((slice(rr*hh//3,(rr+1)*hh//3),slice(cc*ww//2,(cc+1)*ww//2),
                                slice(rr*gray.shape[0]//3,(rr+1)*gray.shape[0]//3),
                                slice(cc*gray.shape[1]//2,(cc+1)*gray.shape[1]//2)))
        expected = []
        for yy,xx,aa,bb in regions:
            selected = fg[yy,xx]; image = gray[aa,bb].astype(np.float64)
            expected.extend([float(np.mean((q[1]+q[2])[yy,xx][selected])) if selected.any() else 0,
                             float(entropy[yy,xx].mean()), float(selected.mean()),
                             connected_labels(selected,np.ones((3,3)))[1],
                             float(image.mean()/255),float(np.mean(image<40)),float(image.std()/255),
                             float(edges[aa,bb].mean())])
        np.testing.assert_allclose(X[index], expected, rtol=0, atol=2e-6)
        fixture_results.append({"sample_id": ids[index], "source_hashes_match": True,
                                "native_semantic_parity": True, "opencv_iou_counts_match": True,
                                "global_feature_formula_check": True, "all_56_feature_formula_check": True,
                                "feature_max_abs_error": float(np.max(np.abs(X[index].astype(np.float64)-expected))),
                                "padding_invariance": True})
    require(fixture_results, "at least one probability fixture is required for this verifier")
    return {"status": "passed", "X_shape": list(X.shape), "y_shape": list(y.shape),
            "defined_labels": int(defined.sum()), "undefined_labels": int((~defined).sum()),
            "unique_sample_ids": True, "csv_npy_join_and_values": "exact", "finite_features": True,
            "all_label_count_identities": True, "artifact_and_source_manifest_hashes": "match",
            "semantic_cache_parity_frames_from_builder": len(samples) if meta.get("input_kind") != "native_semantic" else None,
            "native_semantic_frames": len(samples) if meta.get("input_kind") == "native_semantic" else None,
            "fixtures_rechecked": len(fixture_results), "fixture_results": fixture_results,
            "constant_columns": constants, "duplicate_rgb_groups": len(duplicates),
            "rows_in_duplicate_rgb_groups": sum(map(len, duplicates)),
            "duplicate_rgb_groups_crossing_source_roles": len(crossing),
            "source_roles": dict(Counter(r["source_role"] for r in samples)),
            "quality_split_assigned": False, "regressor_trained": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--data-root", default="/data")
    parser.add_argument("--report", required=True, help="New report file outside the immutable dataset")
    parser.add_argument("--adaptation-run", type=Path)
    args = parser.parse_args()
    report = Path(args.report)
    if report.exists():
        raise FileExistsError(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    identity = identity_from_adaptation(args.adaptation_run) if args.adaptation_run else None
    result = verify(args.dataset_dir, args.data_root, identity)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "fixture_results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
