"""Fit a deterministic alarm cutoff on development normals, then bind its model."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .artifacts import write_json
from .calibration import fit_alarm_cutoff, fit_group_cutoffs, group_alarm_thresholds
from .cycle_gate import completed_run
from .dataset_io import load_dataset, require
from .evaluate_models import reject_training_overlap
from .features import feature_schema
from .metrics import failure_detection_metrics
from .model_io import load_model_run
from .models import predict_array
from .perception_identity import baseline_identity, identity_from_adaptation
from .splits import read_split


def load_alarm_calibration(directory, model_run, bundle):
    directory = Path(directory).resolve()
    run = json.loads((directory/"run.json").read_text())
    require(run.get("status") == "complete", "alarm calibration incomplete")
    require((directory/"_SUCCESS").read_text().strip() == sha256_file(directory/"run.json"),
            "alarm calibration completion differs")
    require("calibration.json" in run["artifact_sha256"], "alarm calibration missing")
    for relative, digest in run["artifact_sha256"].items():
        path = (directory/relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("alarm artifact escapes run") from exc
        require(sha256_file(path) == digest, "alarm calibration artifact changed")
    value = json.loads((directory/"calibration.json").read_text())
    require(value["schema_version"] in ("ml-quality-alarm-1", "ml-quality-alarm-2-grouped") and value["fit_role"] == "meta_dev",
            "alarm calibration role/schema differs")
    require(value["model_run_sha256"] == sha256_file(Path(model_run)/"run.json")
            and value["model"] == bundle["model"]
            and value["checkpoint_sha256"] == bundle["checkpoint_sha256"]
            and value["perception_identity"] == bundle.get("perception_identity", baseline_identity())
            and value["feature_schema"] == feature_schema(), "alarm calibration model identity mismatch")
    policy = value["policy"]
    require(policy["operator"] == "<" and 0 <= policy["alarm_threshold"] <= 1
            and 0 < policy["target_iou"] <= 1, "invalid saved alarm policy")
    if value["schema_version"] == "ml-quality-alarm-2-grouped":
        grouped = value["group_policy"]
        require("environment_centers.json" in run["artifact_sha256"]
                and run["artifact_sha256"]["environment_centers.json"] == value["centers_sha256"]
                and grouped["global"] == policy, "environment alarm identity differs")
        require(all(k.isdecimal() and np.isfinite(v) and 0 <= v <= 1 for k,v in grouped["group_thresholds"].items()),
                "invalid environment cutoff")
    return value


def development_environment_rows(inference_dir, selected, estimate, model_run, bundle):
    """Join a locked RGB inference artifact to eligible calibration IDs and decoded pixels."""
    record = completed_run(inference_dir)
    require(record.get("uses_ground_truth") is False and record["model_run_sha256"] == sha256_file(model_run / "run.json")
            and record["checkpoint_sha256"] == bundle["checkpoint_sha256"] and record["model"] == bundle["model"],
            "RGB predictions differ from the calibrated quality model")
    artifact = record["artifact_sha256"]
    require("environment_centers.json" in artifact and "predictions.jsonl" in artifact
            and artifact["environment_centers.json"] == record.get("environment_centers_sha256"),
            "RGB inference did not preserve its environment centers")
    rows = [json.loads(line) for line in (inference_dir / "predictions.jsonl").read_text().splitlines()]
    by_id = {r["sample_id"]: r for r in rows}
    require(len(by_id) == len(rows) and set(by_id) == {r["sample_id"] for r in selected},
            "environment calibration requires exactly the eligible development IDs")
    require(len(estimate) == len(selected), "development scores differ in length")
    groups = []
    for src, score in zip(selected, estimate):
        row = by_id[src["sample_id"]]
        require(row["pixel_sha256"] == src["pixel_sha256"] and abs(row["estimated_iou"] - score) <= 1e-7,
                "development RGB pixels or quality prediction differ")
        cluster = row.get("environment_cluster")
        require(type(cluster) is int and cluster >= 0, "invalid development environment assignment")
        groups.append(cluster)
    center_bytes = (inference_dir / "environment_centers.json").read_bytes()
    require(hashlib.sha256(center_bytes).hexdigest() == record["environment_centers_sha256"], "centers changed during load")
    return np.asarray(groups), center_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--model", choices=("selected", "OLS", "Ridge", "RandomForest"), default="selected")
    p.add_argument("--adaptation-run", type=Path)
    p.add_argument("--environment-inference", type=Path, help="RGB-only centroid assignments for precisely the eligible development rows")
    p.add_argument("--maximum-fpr", type=float, default=.1)
    p.add_argument("--target-iou", type=float, default=.7)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    require(not a.output_dir.exists(), "alarm output already exists")
    identity = identity_from_adaptation(a.adaptation_run) if a.adaptation_run else baseline_identity()
    data = load_dataset(a.dataset_dir, perception_identity=identity)
    split = read_split(a.split, data)
    bundle, _ = load_model_run(a.model_run, a.model, perception_identity=identity)
    ix = np.array([r["role"] == "meta_dev" and r["eligible"] for r in split["assignments"]])
    require(ix.any(), "no development rows for alarm calibration")
    selected = [r for i,r in enumerate(data.samples) if ix[i]]
    membership = json.loads((a.model_run/"train_membership.json").read_text())
    reject_training_overlap(bundle, membership, selected)
    estimate = predict_array(bundle, data.X[ix])["estimated_iou"]
    policy = fit_alarm_cutoff(data.y[ix], estimate, a.maximum_fpr, a.target_iou)
    value = dict(schema_version="ml-quality-alarm-1", created_utc=datetime.now(timezone.utc).isoformat(),
                 model_run_sha256=sha256_file(a.model_run/"run.json"), model=bundle["model"],
                 checkpoint_sha256=bundle["checkpoint_sha256"], perception_identity=identity,
                 feature_schema=feature_schema(), fit_role="meta_dev", policy=policy,
                 dataset_metadata_sha256=sha256_file(data.path/"dataset.json"), split_sha256=sha256_file(a.split),
                 development_metrics=failure_detection_metrics(data.y[ix], estimate, a.target_iou, policy["alarm_threshold"]),
                 final_data_read=False)
    group_values = None
    if a.environment_inference:
        groups, raw_centers = development_environment_rows(a.environment_inference, selected, estimate, a.model_run, bundle)
        grouped = fit_group_cutoffs(data.y[ix], estimate, groups, maximum_fpr=a.maximum_fpr, target_iou=a.target_iou)
        thresholds = group_alarm_thresholds(grouped, groups)
        group_values = list(zip(groups.tolist(), thresholds.tolist()))
        truth = data.y[ix] < np.asarray(a.target_iou, dtype=data.y.dtype)
        valid = np.isfinite(data.y[ix])
        flagged = estimate < thresholds
        value["global_development_metrics"] = value.pop("development_metrics")
        value.update(schema_version="ml-quality-alarm-2-grouped", group_policy=grouped,
                     centers_sha256=hashlib.sha256(raw_centers).hexdigest(),
                     environment_inference_run_sha256=sha256_file(a.environment_inference / "run.json"),
                     group_development_confusion={k: int((valid & mask).sum()) for k, mask in (
                         ("tp", truth & flagged), ("fp", ~truth & flagged), ("fn", truth & ~flagged), ("tn", ~truth & ~flagged))})
    a.output_dir.mkdir(parents=True)
    if group_values is not None:
        (a.output_dir / "environment_centers.json").write_bytes(raw_centers)
    write_json(a.output_dir/"calibration.json", value)
    (a.output_dir/"source.py").write_bytes(Path(__file__).read_bytes())
    with (a.output_dir/"development_predictions.csv").open("x", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "actual_iou", "estimated_iou", "alarm"] + (["environment_cluster", "alarm_threshold"] if group_values is not None else []))
        for i, (sid, y, pred) in enumerate(zip(np.asarray(data.ids)[ix], data.y[ix], estimate)):
            cutoff = group_values[i][1] if group_values is not None else policy["alarm_threshold"]
            w.writerow([sid, float(y) if np.isfinite(y) else "", float(pred), bool(pred < cutoff)]
                       + (list(group_values[i]) if group_values is not None else []))
    artifacts = {p.name:sha256_file(p) for p in a.output_dir.iterdir() if p.is_file()}
    write_json(a.output_dir/"run.json", dict(status="complete", artifact_sha256=artifacts))
    (a.output_dir/"_SUCCESS").write_text(sha256_file(a.output_dir/"run.json")+"\n")
    print(json.dumps(value.get("group_policy", policy)))


if __name__ == "__main__":
    main()
