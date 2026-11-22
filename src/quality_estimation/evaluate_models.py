"""Evaluate frozen quality models on an explicit role without refitting."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .artifacts import write_json
from .dataset_io import load_dataset, require
from .metrics import regression_metrics, selection_metrics
from .model_io import load_model_run
from .models import predict_array
from .splits import read_split
from .perception_identity import identity_from_adaptation, require_matching_identity


def reject_training_overlap(bundle, training_membership, samples, final_test=False, selection_membership=None):
    """All requested rows are checked, including undefined labels and thinned rows."""
    forbidden_membership = list(training_membership)
    if final_test:
        require(selection_membership is not None, "final_test requires model-selection membership")
        forbidden_membership += selection_membership
    trained_ids = {r["sample_id"] for r in forbidden_membership}
    trained_pixels = {r.get("pixel_sha256") or r["rgb_sha256"] for r in forbidden_membership}
    forbidden_parents = set(bundle["train_parent_groups"])
    if final_test:
        forbidden_parents.update(bundle["selection_parent_groups"])
    for row in samples:
        require(row["sample_id"] not in trained_ids, "evaluation sample was used for training or selection")
        require(row["parent_session_id"] not in forbidden_parents,
                "evaluation parent was used for training or model selection")
        require((row.get("pixel_sha256") or row["input_sha256"]["rgb"]) not in trained_pixels,
                "evaluation image pixels were used for training or selection")


def validate_freeze(path, model_run, role, dataset, requested_samples):
    freeze = json.loads(Path(path).read_text())
    require(freeze["model_run_sha256"] == sha256_file(Path(model_run) / "run.json"), "model changed since freeze")
    require(freeze["feature_schema_sha256"] == dataset.metadata["feature_schema_sha256"], "frozen feature schema differs")
    if "checkpoint_sha256" in freeze:
        require(freeze["checkpoint_sha256"] == dataset.metadata["checkpoint_sha256"], "frozen checkpoint differs")
        require_matching_identity(dataset.metadata["checkpoint_sha256"], dataset.metadata.get("perception_identity"),
                                  freeze.get("perception_identity"))
    require(freeze["evaluation_role"] == role, "evaluation role differs from freeze")
    expected = set(freeze["evaluation_parent_groups"])
    require({r["parent_session_id"] for r in requested_samples} == expected, "frozen evaluation groups differ")
    require(freeze["final_test_accessed_before_freeze"] is False, "freeze declares prior final-test access")
    created = datetime.fromisoformat(freeze["created_utc"].replace("Z", "+00:00"))
    require(created.tzinfo is not None and created <= datetime.now(timezone.utc), "invalid freeze timestamp")
    return freeze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--role", choices=("meta_dev", "candidate_pool", "final_test", "diagnostic_only"), required=True)
    p.add_argument("--freeze", type=Path, help="Required for final_test; freeze before examining its labels/predictions")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--adaptation-run", type=Path)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    require(a.role != "final_test" or a.freeze is not None, "final_test requires a model freeze")
    identity = identity_from_adaptation(a.adaptation_run) if a.adaptation_run else None
    data = load_dataset(a.dataset_dir, perception_identity=identity)
    split = read_split(a.split, data)
    requested = np.array([r["role"] == a.role for r in split["assignments"]])
    eligible = np.array([r["eligible"] for r in split["assignments"]], bool)
    pool = requested & eligible
    require(pool.any(), "requested role has no eligible rows")
    requested_samples = [r for i, r in enumerate(data.samples) if requested[i]]
    freeze = validate_freeze(a.freeze, a.model_run, a.role, data, requested_samples) if a.freeze else None
    bundles = {kind: load_model_run(a.model_run, kind, perception_identity=identity)[0] for kind in ("OLS", "Ridge", "RandomForest")}
    membership = json.loads((a.model_run / "train_membership.json").read_text())
    selection_path = a.model_run / "selection_membership.json"
    selection_membership = None
    if a.role == "final_test":
        inventory = json.loads((a.model_run / "run.json").read_text())["artifact_sha256"]
        require("selection_membership.json" in inventory, "final_test requires hashed selection membership")
        selection_membership = json.loads(selection_path.read_text())
    for bundle in bundles.values():
        reject_training_overlap(bundle, membership, requested_samples, a.role == "final_test", selection_membership)
    protocol = json.loads((a.model_run / "protocol.json").read_text())
    out = a.output_dir; out.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", role=a.role, created_utc=datetime.now(timezone.utc).isoformat(),
                    model_run_sha256=sha256_file(a.model_run / "run.json"),
                    dataset_metadata_sha256=sha256_file(data.path / "dataset.json"),
                    split_sha256=sha256_file(a.split), freeze_sha256=sha256_file(a.freeze) if a.freeze else None,
                    refitting_performed=False)
    write_json(out / "run.json", metadata)
    try:
        X, labels, ids = data.X[pool], data.y[pool], np.asarray(data.ids)[pool]
        predictions = {kind: predict_array(bundle, X) for kind, bundle in bundles.items()}
        scores = {kind: values["risk_score"] for kind, values in predictions.items()}
        scores["Confidence"] = -X[:, data.names.index("global.lane_confidence")]
        scores["Entropy"] = X[:, data.names.index("global.marking_entropy")]
        selection = selection_metrics(labels, ids, scores, protocol["budget_fractions"], protocol["failure_threshold"],
                                      protocol["random_repeats"], protocol["seed"])
        metrics = {kind: regression_metrics(labels, values["raw_estimate"]) for kind, values in predictions.items()}
        samples = [r for i,r in enumerate(data.samples) if pool[i]]
        strata = {}
        for field in ("parent_session_id", "condition", "cohort", "capture_mode"):
            values = [json.dumps(r.get(field, "unknown"), sort_keys=True) for r in samples]
            strata[field] = {}
            for value in sorted(set(values)):
                mask = np.array([v == value for v in values])
                strata[field][value] = dict(rows=int(mask.sum()),
                    failures=int((np.isfinite(labels[mask]) & (labels[mask] < protocol["failure_threshold"])).sum()),
                    models={kind: regression_metrics(labels[mask], pred["raw_estimate"][mask])
                            for kind,pred in predictions.items()})
        report = dict(metadata, status="complete", rows=int(pool.sum()), model_metrics=metrics,
                      grouped_metrics=strata, frozen_selected_model=json.loads((a.model_run / "run.json").read_text())["selected_model"],
                      model_ranking_is_evaluation_only=True, no_post_evaluation_retuning_required=a.role == "final_test")
        write_json(out / "evaluation.json", report)
        write_json(out / "selection.json", selection)
        with (out / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "actual_iou", *[kind+suffix for kind in predictions for suffix in ("_raw", "_clipped")]])
            for i, sid in enumerate(ids):
                writer.writerow([sid, float(labels[i]) if np.isfinite(labels[i]) else "",
                                 *[float(v[key][i]) for v in predictions.values() for key in ("raw_estimate", "estimated_iou")]])
        hashes = {str(path.relative_to(out)): sha256_file(path) for path in out.iterdir()
                  if path.is_file() and path.name != "run.json"}
        write_json(out / "run.json", dict(metadata, status="complete", artifact_sha256=hashes))
        (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")
        print(json.dumps(dict(role=a.role,rows=int(pool.sum()),metrics=metrics)), flush=True)
    except BaseException as exc:
        write_json(out / "run.json", dict(metadata, status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
