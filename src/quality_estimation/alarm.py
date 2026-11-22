"""Fit a deterministic alarm cutoff on development normals, then bind its model."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .artifacts import write_json
from .dataset_io import load_dataset, require
from .evaluate_models import reject_training_overlap
from .features import feature_schema
from .metrics import failure_detection_metrics
from .model_io import load_model_run
from .models import predict_array
from .perception_identity import baseline_identity, identity_from_adaptation
from .splits import read_split


def fit_alarm_cutoff(y, estimated_iou, max_false_positive_rate=.1, target_iou=.7, min_normals=20):
    """Largest strict-< cutoff retaining the requested empirical normal FPR.

    This is an empirical development constraint, not a population guarantee.
    Ties remain together; no evaluation labels or weather metadata are used.
    """
    y, estimate = np.asarray(y), np.asarray(estimated_iou, np.float64)
    require(y.ndim == 1 and y.shape == estimate.shape and y.dtype.kind == "f", "invalid calibration arrays")
    require(0 <= max_false_positive_rate < 1 and 0 < target_iou <= 1 and min_normals >= 1,
            "invalid calibration policy")
    require(np.isfinite(estimate).all() and not np.isinf(y).any(), "nonfinite calibration data")
    valid = np.isfinite(y)
    require(((y[valid] >= 0) & (y[valid] <= 1)).all(), "target IoU outside range")
    normal = valid & (y >= np.asarray(target_iou, dtype=y.dtype))
    require(normal.sum() >= min_normals, "insufficient development normal frames")
    values = np.sort(np.clip(estimate[normal], 0, 1))
    budget = math.floor(len(values)*max_false_positive_rate + 1e-12)
    cutoff = float(values[budget])
    observed = int((values < cutoff).sum())
    require(observed <= budget, "calibration false-positive budget exceeded")
    return dict(target_iou=float(target_iou), alarm_threshold=cutoff, operator="<",
                max_development_false_positive_rate=float(max_false_positive_rate),
                normal_rows=len(values), allowed_false_positives=budget,
                actual_false_positives=observed, actual_development_false_positive_rate=observed/len(values),
                undefined_targets=int((~valid).sum()), score_dtype="float64",
                rule="normal score order statistic at floor(n*FPR), strict <; ties stay together",
                guarantee_scope="empirical development constraint only; held-out FPR can increase")


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
    require(value["schema_version"] == "ml-quality-alarm-1" and value["fit_role"] == "meta_dev",
            "alarm calibration role/schema differs")
    require(value["model_run_sha256"] == sha256_file(Path(model_run)/"run.json")
            and value["model"] == bundle["model"]
            and value["checkpoint_sha256"] == bundle["checkpoint_sha256"]
            and value["perception_identity"] == bundle.get("perception_identity", baseline_identity())
            and value["feature_schema"] == feature_schema(), "alarm calibration model identity mismatch")
    policy = value["policy"]
    require(policy["operator"] == "<" and 0 <= policy["alarm_threshold"] <= 1
            and 0 < policy["target_iou"] <= 1, "invalid saved alarm policy")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--model", choices=("selected", "OLS", "Ridge", "RandomForest"), default="selected")
    p.add_argument("--adaptation-run", type=Path)
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
    a.output_dir.mkdir(parents=True)
    write_json(a.output_dir/"calibration.json", value)
    (a.output_dir/"source.py").write_bytes(Path(__file__).read_bytes())
    with (a.output_dir/"development_predictions.csv").open("x", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "actual_iou", "estimated_iou", "alarm"])
        for sid,y,pred in zip(np.asarray(data.ids)[ix],data.y[ix],estimate):
            w.writerow([sid, float(y) if np.isfinite(y) else "", float(pred), bool(pred<policy["alarm_threshold"])])
    artifacts = {p.name:sha256_file(p) for p in a.output_dir.iterdir() if p.is_file()}
    write_json(a.output_dir/"run.json", dict(status="complete", artifact_sha256=artifacts))
    (a.output_dir/"_SUCCESS").write_text(sha256_file(a.output_dir/"run.json")+"\n")
    print(json.dumps(policy))


if __name__ == "__main__":
    main()
