#!/usr/bin/env python3
"""Select compact RGB scene descriptors with classifier ranking nested in spatial CV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from compare_region_candidates import load, save
from quality_estimation.region_feature_views import environment_views, feature_matrix


def rank_features(x, y, seed, jobs):
    model = RandomForestClassifier(n_estimators=192, min_samples_leaf=2, max_features=.5,
                                    random_state=seed, n_jobs=jobs).fit(x, y)
    return np.argsort(-model.feature_importances_, kind="stable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    schema = json.loads((args.packs/"schema.json").read_text())
    train = load(args.packs, protocol["training_keys"])
    dev = load(args.packs, protocol["development_keys"])
    if set(train["pixel_sha256"]) & set(dev["pixel_sha256"]):
        raise ValueError("training and development overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"models").mkdir(exist_ok=True)
    (args.output_dir/"predictions").mkdir(exist_ok=True)
    folds = [np.isin(train["group"], np.array(protocol["training_keys"])[i::3]) for i in range(3)]
    views = environment_views(schema)
    records, summaries = [], []
    for view in protocol["environment_reduction_views"]:
        columns = views[view]
        x = feature_matrix(train, schema, columns)
        dx = feature_matrix(dev, schema, columns)
        for target in ("weather", "hour"):
            y = train[target]
            ranks = [rank_features(x[~h], y[~h], protocol["seed"], args.jobs) for h in folds]
            full_rank = rank_features(x, y, protocol["seed"], args.jobs)
            for k in protocol["environment_subset_sizes"]:
                if k >= len(columns):
                    continue
                name = view+"_k"+str(k)
                selected = np.sort(full_rank[:k])
                choices = []
                for candidate, settings in enumerate(protocol["parameters"]["SceneRF"]):
                    scores = []
                    for hold, rank in zip(folds, ranks):
                        subset = np.sort(rank[:k])
                        model = RandomForestClassifier(**settings, random_state=protocol["seed"],
                                                       n_jobs=args.jobs).fit(x[~hold][:, subset], y[~hold])
                        pred = model.predict(x[hold][:, subset])
                        scores.append(1-float(balanced_accuracy_score(y[hold], pred)))
                    row = dict(name=name, view=view, target=target, model="SceneRF", candidate=candidate,
                               dimensions=k, parameters=settings, fold_loss=scores, cv_loss=float(np.mean(scores)))
                    records.append(row)
                    choices.append(row)
                    save(args.output_dir/"cv_progress.json", records)
                    print(name, target, candidate, round(row["cv_loss"], 6), flush=True)
                best = min(choices, key=lambda r: (r["cv_loss"], r["candidate"]))
                model = RandomForestClassifier(**best["parameters"], random_state=protocol["seed"],
                                               n_jobs=args.jobs).fit(x[:, selected], y)
                path = args.output_dir/"models"/(name+"_"+target+"_SceneRF.joblib")
                joblib.dump(model, path, compress=2)
                pred = model.predict(dx[:, selected])
                row = dict(best, ordered_features=[columns[i] for i in selected],
                           development_balanced_accuracy=float(balanced_accuracy_score(dev[target], pred)),
                           model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                summaries.append(row)
                save(args.output_dir/"summary.json", summaries)
                np.savez_compressed(args.output_dir/"predictions"/(path.stem+".npz"),
                                    ids=dev["ids"], truth=dev[target], pred=pred)
    save(args.output_dir/"receipt.json", dict(
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        ranking="classifier importance fitted only inside each training fold; separate weather/hour subsets",
    ))


if __name__ == "__main__":
    main()
