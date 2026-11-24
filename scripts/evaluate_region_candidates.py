#!/usr/bin/env python3
"""Score frozen feature/model candidates on named recorded cohorts without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from compare_region_candidates import load, save
from quality_estimation.metrics import failure_detection_metrics, regression_metrics, selection_metrics
from quality_estimation.region_feature_views import feature_matrix


def quality_metrics(data, raw, std=None):
    truth = data["y"]
    predicted = np.clip(raw, 0, 1)
    valid = np.isfinite(truth)
    result = dict(regression=regression_metrics(truth, raw),
                  detection=failure_detection_metrics(truth, predicted),
                  review=selection_metrics(truth, data["ids"], {"model": -predicted},
                                           budgets=(.05, .1, .2, .4), repeats=1))
    if valid.any():
        order = np.lexsort((data["ids"][valid], -predicted[valid]))
        risk = np.cumsum(1-truth[valid][order])/np.arange(1, valid.sum()+1)
        result["aurc"] = float(risk.mean())
        result["aurc_definition"] = "mean cumulative 1-IoU at retained fractions k/N; defined targets only"
    night = valid & np.isin(data["hour"], [5, 20, 23])
    result["night"] = failure_detection_metrics(truth[night], predicted[night])
    if std is not None and valid.any():
        result["gp_observation90"] = dict(
            coverage=float((np.abs(raw[valid]-truth[valid]) <= 1.645*std[valid]).mean()),
            mean_width=float((2*1.645*std[valid]).mean()),
            note="raw mean and model observation standard deviation; no calibration on this cohort",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--frozen-models", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keys", nargs="+")
    args = parser.parse_args()
    freeze = json.loads(args.frozen_models.read_text())
    schema = json.loads((args.packs/"schema.json").read_text())
    keys = args.keys or sorted(p.stem for p in args.packs.glob("*.npz"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"predictions").mkdir(exist_ok=True)
    cohorts = {}
    for key in keys:
        receipt = json.loads((args.packs/(key+".json")).read_text())
        cohorts.setdefault(receipt["cohort"], []).append(key)
    summary = []
    for cohort, cohort_keys in cohorts.items():
        data = load(args.packs, cohort_keys)
        for recipe in freeze["diagnostic_models"]:
            path = args.model_root/recipe["model_file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != recipe["model_sha256"]:
                raise ValueError("frozen model checksum differs")
            model = joblib.load(path)
            x = feature_matrix(data, schema, recipe["ordered_features"])
            std = None
            if recipe["kind"] == "GP":
                raw, std = model.predict(x, return_std=True)
            else:
                raw = model.predict(x)
            row = dict(cohort=cohort, name=recipe["name"], kind=recipe["kind"],
                       target=recipe["target"], n=len(x), dimensions=x.shape[1])
            if recipe["target"] == "y":
                row["metrics"] = quality_metrics(data, raw, std)
            else:
                truth = data[recipe["target"]]
                valid = np.isin(truth, model.classes_)
                labels = np.unique(truth[valid])
                row["metrics"] = dict(
                    n_evaluable=int(valid.sum()), n_unknown_target=int((~valid).sum()),
                    balanced_accuracy=float(np.mean([np.mean(raw[truth == c] == c) for c in labels])) if len(labels) else None,
                    accuracy=float((raw[valid] == truth[valid]).mean()) if valid.any() else None,
                    class_order=model.classes_.tolist(),
                    confusion=(confusion_matrix(truth[valid], raw[valid], labels=model.classes_).tolist()
                               if valid.any() else np.zeros((len(model.classes_), len(model.classes_)), int).tolist()),
                )
            output = args.output_dir/"predictions"/(cohort+"__"+recipe["name"]+".npz")
            arrays = dict(ids=data["ids"], raw=raw)
            if std is not None:
                arrays["std"] = std
            np.savez_compressed(output, **arrays)
            summary.append(row)
            save(args.output_dir/"summary.json", summary)
            print(cohort, recipe["name"], row["metrics"].get("regression", {}).get("clipped_MAE",
                       row["metrics"].get("balanced_accuracy")), flush=True)
    save(args.output_dir/"receipt.json", dict(
        frozen_models_sha256=hashlib.sha256(args.frozen_models.read_bytes()).hexdigest(),
        schema_sha256=hashlib.sha256((args.packs/"schema.json").read_bytes()).hexdigest(),
        cohorts=cohorts, scoring="frozen predictions; no model or feature fitting",
    ))


if __name__ == "__main__":
    main()
