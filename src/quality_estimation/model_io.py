"""Load locally trained, completed model bundles with their artifact checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from .dataset_io import require
from .features import FEATURE_NAMES, feature_schema
from .models import QUALITY_MODELS, feature_indices
from .perception_identity import require_matching_identity


def load_model_run(directory, model="selected", inference_jobs=1, perception_identity=None):
    """Only load trusted local joblib files; hashes protect against accidental edits."""
    directory = Path(directory).resolve()
    require(isinstance(inference_jobs, int) and inference_jobs >= 1, "inference_jobs must be positive")
    run = json.loads((directory / "run.json").read_text())
    require(run.get("status") == "complete", "model run is not complete")
    require((directory / "_SUCCESS").read_text().strip() == sha256_file(directory / "run.json"),
            "model completion hash mismatch")
    if model == "selected":
        model = run["selected_model"]
    require(model in QUALITY_MODELS, "unknown model")
    required = {"models/" + model + ".joblib", "comparison.json", "protocol.json", "train_membership.json"}
    require(required.issubset(run["artifact_sha256"]), "model artifact inventory is incomplete")
    for name, digest in run["artifact_sha256"].items():
        path = (directory / name).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("model artifact path escapes run") from exc
        require(sha256_file(path) == digest, "model artifact hash mismatch: " + name)
    bundle = joblib.load(directory / "models" / (model + ".joblib"))
    require(bundle["model"] == model, "model identity mismatch")
    require_matching_identity(bundle["checkpoint_sha256"], bundle.get("perception_identity"), perception_identity)
    require(bundle["feature_schema"] == feature_schema(), "model feature definitions differ")
    require(bundle["ordered_features"] == list(FEATURE_NAMES), "model feature order differs")
    require(bundle["column_indices"] == feature_indices(bundle["feature_set"]), "model feature subset differs")
    require(bundle["protocol_sha256"] == sha256_file(directory / "protocol.json"), "model protocol differs")
    # Joblib thread dispatch dominates a single small forest prediction.
    # This changes execution parallelism only; learned trees remain untouched.
    if model == "RandomForest":
        bundle["estimator"].set_params(regressor__n_jobs=inference_jobs)
    bundle["runtime_inference_jobs"] = inference_jobs
    return bundle, run


def write_prediction_csv(path, ids, labels, predictions):
    """Preserve existing point-estimate columns and append IoU standard deviations when available."""
    columns = [(name + suffix, values[key]) for name, values in predictions.items()
               for suffix, key in (("_raw", "raw_estimate"), ("_clipped", "estimated_iou"))]
    columns.extend((name + "_std", values["prediction_std"]) for name, values in predictions.items()
                   if "prediction_std" in values)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "actual_iou", *[header for header, _ in columns]])
        for index, sid in enumerate(ids):
            writer.writerow([sid, float(labels[index]) if np.isfinite(labels[index]) else "",
                             *[float(values[index]) for _, values in columns]])
