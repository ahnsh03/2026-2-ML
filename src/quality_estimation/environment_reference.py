"""Fit training-only IoU references for ranking unusual quality within a frozen RGB cluster."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .artifacts import write_json
from .cycle_gate import completed_run
from .dataset_io import load_dataset, require
from .environment_clusters import assign_environment_clusters
from .perception_identity import baseline_identity, identity_from_adaptation


def fit_reference(y, groups, cluster_count):
    """Estimate a lower quintile with a global fallback for groups with fewer than 20 training rows."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    require(y.ndim == 1 and groups.shape == y.shape and groups.dtype.kind in "iu",
            "expected one integer cluster per training IoU")
    require(len(y) >= 20 and np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all(), "invalid training IoU")
    require(cluster_count >= 2 and ((groups >= 0) & (groups < cluster_count)).all(), "invalid training cluster")
    global_q20 = float(np.quantile(y, .2))
    records = []
    for group in range(cluster_count):
        values = y[groups == group]
        records.append(dict(rows=len(values), q20=float(np.quantile(values, .2)) if len(values) >= 20 else global_q20,
                            source="cluster" if len(values) >= 20 else "global"))
    return dict(schema="environment-iou-reference-v1", quantile=.2, minimum_cluster_rows=20,
                training_rows=len(y), global_q20=global_q20, clusters=records)


def load_reference(directory, inference):
    run = completed_run(directory)
    for key in ("model_run_sha256", "checkpoint_sha256", "environment_centers_sha256"):
        require(bool(run.get(key)) and run[key] == inference.get(key), "environment reference mismatch: " + key)
    require("reference.json" in run["artifact_sha256"], "unlisted environment reference")
    reference = json.loads((directory / "reference.json").read_text())
    require(reference.get("schema") == "environment-iou-reference-v1" and reference.get("quantile") == .2,
            "unknown environment reference")
    require(len(reference["clusters"]) >= 2 and all(
        type(group["rows"]) is int and group["rows"] >= 0 and math.isfinite(group["q20"]) and 0 <= group["q20"] <= 1
        for group in reference["clusters"]), "invalid environment reference groups")
    return reference


def with_shortfall(row, reference):
    group = row.get("environment_cluster")
    require(type(group) is int and 0 <= group < len(reference["clusters"]), "missing or invalid environment cluster")
    q20 = reference["clusters"][group]["q20"]
    return dict(row, environment_reference_iou=q20, estimated_iou_shortfall=q20 - row["estimated_iou"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--environment-fit", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Original dataset used to fit this quality model")
    parser.add_argument("--adaptation-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    model = completed_run(args.model_run)
    fit = completed_run(args.environment_fit)
    digest = sha256_file(args.model_run / "run.json")
    require(fit.get("fit_role") == "meta_train" and fit["model_run_sha256"] == digest and
            fit["membership_sha256"] == model["artifact_sha256"]["train_membership.json"], "center training membership differs")
    comparison = json.loads((args.model_run / "comparison.json").read_text())
    require(comparison["dataset_metadata_sha256"] == sha256_file(args.dataset_dir / "dataset.json"),
            "dataset differs from model training")
    identity = identity_from_adaptation(args.adaptation_run) if args.adaptation_run else baseline_identity()
    data = load_dataset(args.dataset_dir, perception_identity=identity)
    require(comparison["checkpoint_sha256"] == identity["checkpoint_sha256"], "quality training perception differs")
    centers_path = args.environment_fit / "environment_centers.json"
    centers = json.loads(centers_path.read_text())
    members = json.loads((args.model_run / "train_membership.json").read_text())
    kept = json.loads((args.environment_fit / "training_rows.json").read_text())
    by_id = {member["sample_id"]: member for member in kept}
    require(len(kept) == len(by_id) == len({m["pixel_sha256"] for m in kept}) == fit["unique_training_pixels"],
            "duplicate or missing centroid training rows")
    features = np.load(args.environment_fit / "environment_features.npy", allow_pickle=False)
    require(len(features) == len(kept), "centroid descriptor count differs")
    training = [m for m in members if m["sample_id"] in by_id]
    require([m["sample_id"] for m in training] == [m["sample_id"] for m in kept], "unexpected centroid training IDs/order")
    indices = {sid: i for i, sid in enumerate(data.ids)}
    labels = []
    for member, center_row in zip(training, kept):
        i = indices[member["sample_id"]]
        sample = data.samples[i]
        require(data.defined[i] and member["rgb_sha256"] == center_row["rgb_sha256"] == sample["input_sha256"]["rgb"],
                "centroid row differs from the training label")
        require(member.get("pixel_sha256") in (None, center_row["pixel_sha256"]) and
                sample.get("pixel_sha256") in (None, center_row["pixel_sha256"]), "training pixels differ")
        require(sample.get("preregistered_quality_role") not in ("final_test", "candidate_pool", "meta_calibration"),
                "reserved row in reference training")
        labels.append(data.y[i])
    reference = fit_reference(labels, assign_environment_clusters(centers, features), centers["k"])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "reference.json", reference)
    write_json(out / "run.json", dict(status="complete", fit_role="meta_train", model_run_sha256=digest,
               checkpoint_sha256=identity["checkpoint_sha256"], dataset_metadata_sha256=comparison["dataset_metadata_sha256"],
               environment_fit_sha256=sha256_file(args.environment_fit / "run.json"), environment_centers_sha256=sha256_file(centers_path),
               artifact_sha256={"reference.json": sha256_file(out / "reference.json")}))
    (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")


if __name__ == "__main__":
    main()
