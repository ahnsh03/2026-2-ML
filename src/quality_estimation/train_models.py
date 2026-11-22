"""Fit OLS/Ridge/RF on explicit groups; choose by held-out development MAE only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from common.bridge import sha256_file
from .artifacts import write_json
from .build_dataset import REPO_ROOT
from .dataset_io import load_dataset, require
from .features import FEATURE_NAMES, feature_schema
from .metrics import regression_metrics, selection_metrics
from .models import feature_indices, make_estimator, predict_array
from .splits import read_split
from .perception_identity import baseline_identity, identity_from_adaptation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--feature-set", choices=("full", "global_only", "output_only", "rgb_only"), default="full")
    p.add_argument("--adaptation-run", type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    identity = identity_from_adaptation(args.adaptation_run) if args.adaptation_run else baseline_identity()
    data = load_dataset(args.dataset_dir, perception_identity=identity)
    split = read_split(args.split, data)
    protocol = json.loads(args.protocol.read_text())
    require(protocol["selection_metric"] == "clipped_MAE", "unsupported model-selection metric")
    require(protocol["tie_order"] == ["OLS", "Ridge", "RandomForest"], "unsupported tie order")
    assignments = split["assignments"]
    roles = np.array([a["role"] for a in assignments])
    eligible = np.array([a["eligible"] for a in assignments], bool)
    train = (roles == "meta_train") & eligible & data.defined
    dev = (roles == "meta_dev") & eligible & data.defined
    dev_pool = (roles == "meta_dev") & eligible
    require(train.sum() >= 2 and dev.sum() >= 2, "insufficient eligible train/dev labels")
    indices = feature_indices(args.feature_set)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    shutil.copyfile(args.protocol, out / "protocol.json")
    source = {str(f.relative_to(REPO_ROOT)): sha256_file(f) for f in sorted((REPO_ROOT / "src/quality_estimation").glob("*.py"))}
    for relative, digest in source.items():
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
        require(sha256_file(destination) == digest, "training source changed while being archived")
    train_indices = np.flatnonzero(train)
    train_ids = [data.ids[i] for i in train_indices]
    selection_indices = np.flatnonzero(dev_pool)
    selection_ids = [data.ids[i] for i in selection_indices]
    dev_parents = sorted({r["parent_session_id"] for i,r in enumerate(data.samples) if dev_pool[i]})
    train_parents = sorted({r["parent_session_id"] for i,r in enumerate(data.samples) if train[i]})
    common = {"schema_version": "ml-quality-model-1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_metadata_sha256": sha256_file(data.path / "dataset.json"), "split_sha256": sha256_file(args.split),
              "protocol_sha256": sha256_file(args.protocol), "source_sha256": source,
              "checkpoint_sha256": data.metadata["checkpoint_sha256"], "feature_schema": feature_schema(),
              "perception_identity": identity,
              "ordered_features": list(FEATURE_NAMES), "column_indices": indices, "feature_set": args.feature_set,
              "train_sample_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
              "selection_sample_ids_sha256": hashlib.sha256("\n".join(selection_ids).encode()).hexdigest(),
              "train_parent_groups": train_parents, "selection_parent_groups": dev_parents,
              "train_rows": int(train.sum()), "dev_rows": int(dev.sum()), "dev_pool_rows": int(dev_pool.sum()),
              "sklearn_version": sklearn.__version__, "seed": protocol["seed"],
              "fit_n_jobs": protocol["n_jobs"], "inference_n_jobs": 1,
              "criterion": "same development rows; clipped MAE; no final test access"}
    write_json(out / "run.json", dict(common, status="running"))
    write_json(out / "train_membership.json", [{"sample_id": data.ids[i],
        "pixel_sha256": data.samples[i].get("pixel_sha256"), "rgb_sha256": data.samples[i]["input_sha256"]["rgb"],
        "parent_session_id": data.samples[i]["parent_session_id"]} for i in train_indices])
    write_json(out / "selection_membership.json", [{"sample_id": data.ids[i],
        "pixel_sha256": data.samples[i].get("pixel_sha256"), "rgb_sha256": data.samples[i]["input_sha256"]["rgb"],
        "parent_session_id": data.samples[i]["parent_session_id"]} for i in selection_indices])
    Xtrain, ytrain = data.X[train][:, indices].astype(np.float64), data.y[train].astype(np.float64)
    Xdev = data.X[dev][:, indices].astype(np.float64)
    candidates, best, bundles = [], {}, {}
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=protocol["n_jobs"]):
            for kind in protocol["tie_order"]:
                for number, params in enumerate(protocol["models"][kind]):
                    clock = time.perf_counter()
                    estimator = make_estimator(kind, params, protocol["seed"], protocol["n_jobs"])
                    estimator.fit(Xtrain, ytrain)
                    raw = estimator.predict(Xdev)
                    metrics = regression_metrics(data.y[dev], raw)
                    candidate = {"model": kind, "grid_index": number, "parameters": params,
                                 "metrics": metrics, "seconds": time.perf_counter()-clock}
                    candidates.append(candidate)
                    with (out / "candidate_progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(candidate, allow_nan=False) + "\n")
                    if kind not in best or metrics["clipped_MAE"] < best[kind]["metrics"]["clipped_MAE"] - protocol["tie_tolerance"]:
                        best[kind] = candidate
                        bundles[kind] = dict(common, model=kind, parameters=params, estimator=estimator,
                                             validation_metrics=metrics, grid_index=number)
                    print(f"{kind} {number+1}/{len(protocol['models'][kind])}: dev MAE={metrics['clipped_MAE']:.6f}; {time.perf_counter()-clock:.1f}s", flush=True)
            winner = protocol["tie_order"][0]
            for kind in protocol["tie_order"][1:]:
                if best[kind]["metrics"]["clipped_MAE"] < best[winner]["metrics"]["clipped_MAE"] - protocol["tie_tolerance"]:
                    winner = kind
            bundles["RandomForest"]["estimator"].set_params(regressor__n_jobs=1)
            predictions = {name: predict_array(bundle, data.X[dev_pool]) for name,bundle in bundles.items()}
            for name, bundle in bundles.items():
                joblib.dump(bundle, out / "models" / (name + ".joblib"), compress=3)
            labels = data.y[dev_pool]
            dev_ids = np.asarray(data.ids)[dev_pool]
            scores = {name: v["risk_score"] for name, v in predictions.items()}
            scores["Confidence"] = -data.X[dev_pool, data.names.index("global.lane_confidence")]
            scores["Entropy"] = data.X[dev_pool, data.names.index("global.marking_entropy")]
            selection = selection_metrics(labels, dev_ids, scores, protocol["budget_fractions"],
                                           protocol["failure_threshold"], protocol["random_repeats"], protocol["seed"])
            write_json(out / "development_selection.json", selection)
            per_group = {}
            for group in dev_parents:
                rows = np.array([s["parent_session_id"] == group for i,s in enumerate(data.samples) if dev_pool[i]])
                per_group[group] = {kind: regression_metrics(labels[rows], pred["raw_estimate"][rows]) for kind,pred in predictions.items()}
            mean_reference = regression_metrics(labels, np.full(len(labels), ytrain.mean()))
            report = dict(common, status="complete", selected_model=winner, best_by_model=best,
                          training_mean_reference=mean_reference, per_development_parent=per_group,
                          uncertainty=f"only {len(dev_parents)} development parent groups; no independent-session CI" if len(dev_parents)<3 else "parent resampling is a separate analysis",
                          elapsed_seconds=time.perf_counter()-started, final_test_accessed=False)
            write_json(out / "comparison.json", report)
            write_json(out / "candidates.json", candidates)
            with (out / "development_predictions.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "actual_iou", *[k+s for k in predictions for s in ("_raw", "_clipped")]])
                for i, sid in enumerate(dev_ids):
                    writer.writerow([sid, float(labels[i]) if np.isfinite(labels[i]) else "",
                        *[float(v[key][i]) for v in predictions.values() for key in ("raw_estimate", "estimated_iou")]])
            artifacts = {str(f.relative_to(out)): sha256_file(f) for f in out.rglob("*") if f.is_file() and f.name != "run.json"}
            write_json(out / "run.json", dict(status="complete", selected_model=winner, artifact_sha256=artifacts))
            (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")
            print("Selected on development MAE: " + winner, flush=True)
    except BaseException as exc:
        write_json(out / "run.json", dict(common, status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
