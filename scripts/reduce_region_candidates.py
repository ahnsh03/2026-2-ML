#!/usr/bin/env python3
"""Evaluate compact feature subsets with ranking refitted inside each training fold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from compare_region_candidates import fit, load, save
from quality_estimation.region_feature_views import feature_matrix, quality_views


def ranking(x, y, method, seed, jobs):
    if method == "RF":
        ranker = RandomForestRegressor(n_estimators=192, min_samples_leaf=8,
                                       max_features=.5, random_state=seed, n_jobs=jobs).fit(x, y)
        score = ranker.feature_importances_
    else:
        score = mutual_info_regression(x, y, random_state=seed, n_neighbors=5)
    return np.argsort(-score, kind="stable"), score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    schema = json.loads((args.packs/"schema.json").read_text())
    train = load(args.packs, protocol["training_keys"])
    dev = load(args.packs, protocol["development_keys"])
    if set(train["ids"]) & set(dev["ids"]) or set(train["pixel_sha256"]) & set(dev["pixel_sha256"]):
        raise ValueError("training and development overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"models").mkdir(exist_ok=True)
    (args.output_dir/"predictions").mkdir(exist_ok=True)
    views = quality_views(schema)
    views.update(protocol.get("extra_views", {}))
    folds = [np.isin(train["group"], np.array(protocol["training_keys"])[i::3]) for i in range(3)]
    progress_path = args.output_dir/"cv_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
    done = {(r["view"], r["ranking"], r["k"], r["model"], r["candidate"]): r for r in progress}
    summary = []
    for view in protocol["reduction_views"]:
        columns = views[view]
        x = feature_matrix(train, schema, columns)
        dx = feature_matrix(dev, schema, columns)
        y = train["y"]
        for method in protocol["ranking_methods"]:
            # Ranks are fitted exclusively on each inner training partition.
            folds_ranked = [ranking(x[~hold], y[~hold], method, protocol["seed"], args.jobs)[0]
                            for hold in folds]
            full_rank, full_scores = ranking(x, y, method, protocol["seed"], args.jobs)
            for k in protocol["subset_sizes"]:
                if k > len(columns):
                    continue
                selected = np.sort(full_rank[:k])
                name = view+"_"+method+"_k"+str(k)
                for kind in ("RF", "GP"):
                    choices = []
                    for candidate, parameters in enumerate(protocol["parameters"][kind]):
                        key = (view, method, k, kind, candidate)
                        if key not in done:
                            errors = []
                            for hold, rank in zip(folds, folds_ranked):
                                subset = np.sort(rank[:k])
                                model = fit(kind, x[~hold][:, subset], y[~hold], parameters,
                                            protocol["seed"], args.jobs)
                                prediction = np.clip(model.predict(x[hold][:, subset]), 0, 1)
                                errors.append(float(np.mean(abs(prediction-y[hold]))))
                            row = dict(view=view, ranking=method, k=k, model=kind,
                                       candidate=candidate, parameters=parameters, fold_loss=errors,
                                       cv_loss=float(np.mean(errors)), name=name)
                            progress.append(row)
                            done[key] = row
                            save(progress_path, progress)
                            print(name, kind, candidate, round(row["cv_loss"], 6), flush=True)
                        choices.append(done[key])
                    best = min(choices, key=lambda r: (r["cv_loss"], r["candidate"]))
                    model = fit(kind, x[:, selected], y, best["parameters"], protocol["seed"], args.jobs)
                    path = args.output_dir/"models"/(name+"_y_"+kind+".joblib")
                    joblib.dump(model, path, compress=2)
                    pred = np.clip(model.predict(dx[:, selected]), 0, 1)
                    row = dict(best, ordered_features=[columns[i] for i in selected], dimensions=k,
                               development_mae=float(np.mean(abs(pred-dev["y"]))),
                               ranked_features=[columns[i] for i in full_rank],
                               ranking_scores=full_scores.tolist(),
                               model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    summary.append(row)
                    save(args.output_dir/"summary.json", summary)
                    np.savez_compressed(args.output_dir/"predictions"/(path.stem+".npz"),
                                        ids=dev["ids"], truth=dev["y"], pred=pred)
    save(args.output_dir/"receipt.json", dict(
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        selection="ranking and model preprocessing refitted on each inner training fold",
    ))


if __name__ == "__main__":
    main()
