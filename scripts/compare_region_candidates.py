#!/usr/bin/env python3
"""Grouped RF/GP quality and RGB scene comparisons using fixed candidate packs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from quality_estimation.gaussian_process import fit_quality_gp
from quality_estimation.region_feature_views import environment_views, feature_matrix, quality_views


def save(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def load(directory, keys):
    pieces = {}
    schema_digest = hashlib.sha256((directory/"schema.json").read_bytes()).hexdigest()
    for key in keys:
        path = directory/(key+".npz")
        receipt = json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("candidate pack checksum differs")
        if receipt["schema_sha256"] != schema_digest:
            raise ValueError("candidate feature schema differs")
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                pieces.setdefault(name, []).append(data[name])
    result = {name: np.concatenate(values) for name, values in pieces.items()}
    if len(np.unique(result["ids"])) != len(result["ids"]):
        raise ValueError("duplicate sample IDs")
    return result


def fit(kind, x, y, settings, seed, jobs):
    if kind == "RF":
        return RandomForestRegressor(**settings, random_state=seed, n_jobs=jobs).fit(x, y)
    if kind == "GP":
        return fit_quality_gp(x, y, **settings)
    return RandomForestClassifier(**settings, random_state=seed, n_jobs=jobs).fit(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", choices=("quality", "environment"), required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    schema = json.loads((args.packs/"schema.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"models").mkdir(exist_ok=True)
    (args.output_dir/"predictions").mkdir(exist_ok=True)
    train = load(args.packs, protocol["training_keys"])
    dev = load(args.packs, protocol["development_keys"])
    if set(train["ids"]) & set(dev["ids"]) or set(train["pixel_sha256"]) & set(dev["pixel_sha256"]):
        raise ValueError("train/development overlap")
    folds = [np.isin(train["group"], np.array(protocol["training_keys"])[i::3]) for i in range(3)]
    views = quality_views(schema) if args.task == "quality" else environment_views(schema)
    views.update(protocol.get("extra_views", {}))
    names = protocol.get(args.task+"_views", list(views))
    targets = ["y"] if args.task == "quality" else ["weather", "hour"]
    kinds = ["RF", "GP"] if args.task == "quality" else ["SceneRF"]
    progress_path = args.output_dir/"cv_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
    done = {(r["view"], r["target"], r["model"], r["candidate"]): r for r in progress}
    summary = []
    for view in names:
        columns = views[view]
        x = feature_matrix(train, schema, columns)
        dx = feature_matrix(dev, schema, columns)
        for target in targets:
            y = train[target]
            if not np.isfinite(y).all():
                raise ValueError("undefined training target")
            for kind in kinds:
                rows = []
                for candidate, settings in enumerate(protocol["parameters"][kind]):
                    key = (view, target, kind, candidate)
                    if key not in done:
                        start = time.perf_counter()
                        scores = []
                        for hold in folds:
                            model = fit(kind, x[~hold], y[~hold], settings, protocol["seed"], args.jobs)
                            prediction = model.predict(x[hold])
                            score = (float(np.mean(abs(np.clip(prediction, 0, 1)-y[hold])))
                                     if target == "y" else 1-float(balanced_accuracy_score(y[hold], prediction)))
                            scores.append(score)
                        row = dict(view=view, target=target, model=kind, candidate=candidate,
                                   dimensions=len(columns), parameters=settings, fold_loss=scores,
                                   cv_loss=float(np.mean(scores)), seconds=time.perf_counter()-start)
                        progress.append(row)
                        done[key] = row
                        save(progress_path, progress)
                        print(view, target, kind, candidate, round(row["cv_loss"], 6), flush=True)
                    rows.append(done[key])
                best = min(rows, key=lambda r: (r["cv_loss"], r["candidate"]))
                model = fit(kind, x, y, best["parameters"], protocol["seed"], args.jobs)
                predicted = model.predict(dx)
                path = args.output_dir/"models"/(view+"_"+target+"_"+kind+".joblib")
                joblib.dump(model, path, compress=2)
                row = dict(best, ordered_features=columns,
                           model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if target == "y":
                    predicted = np.clip(predicted, 0, 1)
                    row["development_mae"] = float(np.mean(abs(predicted-dev[target])))
                else:
                    row["development_balanced_accuracy"] = float(balanced_accuracy_score(dev[target], predicted))
                    row["development_macro_f1"] = float(f1_score(dev[target], predicted, average="macro"))
                    row["development_confusion"] = confusion_matrix(dev[target], predicted).tolist()
                np.savez_compressed(args.output_dir/"predictions"/(path.stem+".npz"),
                                    ids=dev["ids"], truth=dev[target], pred=predicted)
                summary.append(row)
                save(args.output_dir/"summary.json", summary)
    save(args.output_dir/"receipt.json", dict(
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        schema_sha256=hashlib.sha256((args.packs/"schema.json").read_bytes()).hexdigest(),
        n_train=len(train["ids"]), n_development=len(dev["ids"]),
        cv="three grouped training folds; all preprocessing fitted inside fold",
    ))


if __name__ == "__main__":
    main()
