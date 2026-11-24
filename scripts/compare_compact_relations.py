#!/usr/bin/env python3
"""Add physical region relations after nested feature reduction, with paired RF/GP fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from compare_region_candidates import fit, load, save
from reduce_region_candidates import ranking
from quality_estimation.region_feature_views import feature_matrix, quality_views


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
    if set(train["pixel_sha256"]) & set(dev["pixel_sha256"]):
        raise ValueError("training and development overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"models").mkdir(exist_ok=True)
    (args.output_dir/"predictions").mkdir(exist_ok=True)
    views = quality_views(schema)
    folds = [np.isin(train["group"], np.array(protocol["training_keys"])[i::3]) for i in range(3)]
    progress_file = args.output_dir/"cv_progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else []
    done = {(r["name"], r["model"], r["candidate"]): r for r in progress}
    summary = []
    for reduction in protocol["reductions"]:
        columns = views[reduction["view"]]
        x = feature_matrix(train, schema, columns)
        dx = feature_matrix(dev, schema, columns)
        y = train["y"]
        k, method = reduction["k"], reduction["ranking"]
        fold_ranks = [ranking(x[~h], y[~h], method, protocol["seed"], args.jobs)[0] for h in folds]
        full_rank, _ = ranking(x, y, method, protocol["seed"], args.jobs)
        selected = np.sort(full_rank[:k])
        for arm, relation_columns in protocol["relation_arms"].items():
            relation_x = feature_matrix(train, schema, relation_columns) if relation_columns else np.empty((len(y), 0))
            relation_dx = feature_matrix(dev, schema, relation_columns) if relation_columns else np.empty((len(dx), 0))
            name = reduction["name"]+"_"+arm
            final_x = np.column_stack([x[:, selected], relation_x])
            final_dx = np.column_stack([dx[:, selected], relation_dx])
            for kind in ("RF", "GP"):
                choices = []
                for candidate, parameters in enumerate(protocol["parameters"][kind]):
                    key = (name, kind, candidate)
                    if key not in done:
                        errors = []
                        for hold, ranks in zip(folds, fold_ranks):
                            subset = np.sort(ranks[:k])
                            fold_x = np.column_stack([x[:, subset], relation_x])
                            model = fit(kind, fold_x[~hold], y[~hold], parameters, protocol["seed"], args.jobs)
                            predicted = np.clip(model.predict(fold_x[hold]), 0, 1)
                            errors.append(float(np.mean(abs(predicted-y[hold]))))
                        row = dict(name=name, view=reduction["view"], ranking=method, k=k,
                                   relation_arm=arm, model=kind, candidate=candidate,
                                   parameters=parameters, fold_loss=errors, cv_loss=float(np.mean(errors)))
                        progress.append(row)
                        done[key] = row
                        save(progress_file, progress)
                        print(name, kind, candidate, round(row["cv_loss"], 6), flush=True)
                    choices.append(done[key])
                best = min(choices, key=lambda r: (r["cv_loss"], r["candidate"]))
                model = fit(kind, final_x, y, best["parameters"], protocol["seed"], args.jobs)
                path = args.output_dir/"models"/(name+"_y_"+kind+".joblib")
                joblib.dump(model, path, compress=2)
                predicted = np.clip(model.predict(final_dx), 0, 1)
                row = dict(best, ordered_features=[columns[i] for i in selected]+relation_columns,
                           dimensions=final_x.shape[1], development_mae=float(np.mean(abs(predicted-dev["y"]))),
                           model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                summary.append(row)
                save(args.output_dir/"summary.json", summary)
                np.savez_compressed(args.output_dir/"predictions"/(path.stem+".npz"),
                                    ids=dev["ids"], truth=dev["y"], pred=predicted)
    save(args.output_dir/"receipt.json", dict(
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        selection="same within-fold feature ranking as reduction stage; relations appended after reduction",
    ))


if __name__ == "__main__":
    main()
