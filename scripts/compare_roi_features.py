#!/usr/bin/env python3
"""Reproduce the fixed-budget RF/GP comparison on verified seven/ten-region feature packs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quality_estimation.gaussian_process import fit_quality_gp


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_packs(directory, keys):
    columns = {}
    receipts = []
    for key in keys:
        path = directory / (key + ".npz")
        receipt = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != receipt["sha256"]:
            raise ValueError("feature pack checksum differs")
        receipts.append(receipt)
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                columns.setdefault(name, []).append(data[name])
    combined = {name: np.concatenate(values) for name, values in columns.items()}
    if (len(np.unique(combined["ids"])) != len(combined["ids"])
            or len(np.unique(combined["pixel_sha256"])) != len(combined["ids"])):
        raise ValueError("duplicate training/development frames")
    if not np.isfinite(combined["y"]).all():
        raise ValueError("this comparison protocol requires defined training/development targets")
    return combined, receipts


def fit_model(kind, x, y, parameters, seed, jobs):
    if kind == "RF":
        return RandomForestRegressor(**parameters, random_state=seed, n_jobs=jobs).fit(x, y)
    return fit_quality_gp(x, y, **parameters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    train_keys = protocol["training_keys"]
    train, training_receipts = load_packs(args.packs, train_keys)
    development, development_receipts = load_packs(args.packs, protocol["development_keys"])
    if (set(train["ids"]) & set(development["ids"])
            or set(train["pixel_sha256"]) & set(development["pixel_sha256"])):
        raise ValueError("training and development overlap")
    if len(train_keys) < 3 or args.jobs < 1:
        raise ValueError("expected at least three training groups and positive job count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models = args.output_dir / "models"
    models.mkdir()
    seed = protocol["seed"]
    folds = [np.isin(train["group"], np.asarray(train_keys)[i::3]) for i in range(3)]
    if any(not hold.any() or hold.all() for hold in folds):
        raise ValueError("empty grouped fold")
    progress = []
    variants = []
    for roi in (7, 10):
        for arm in protocol["feature_arms"]:
            for kind in ("RF", "GP"):
                x = train[f"x{roi}_{arm}"]
                choices = []
                for candidate, parameters in enumerate(protocol["parameters"][kind]):
                    errors = []
                    for hold in folds:
                        model = fit_model(kind, x[~hold], train["y"][~hold], parameters, seed, args.jobs)
                        estimated = np.clip(model.predict(x[hold]), 0., 1.)
                        errors.append(float(np.mean(abs(estimated - train["y"][hold]))))
                    row = dict(roi=roi, arm=arm, model=kind, candidate=candidate,
                               parameters=parameters, fold_mae=errors, cv_mae=float(np.mean(errors)))
                    choices.append(row)
                    progress.append(row)
                    write_json(args.output_dir / "cv_progress.json", progress)
                best = min(choices, key=lambda row: (row["cv_mae"], row["candidate"]))
                model = fit_model(kind, x, train["y"], best["parameters"], seed, args.jobs)
                output = models / f"{roi}_{arm}_{kind}.joblib"
                joblib.dump(model, output, compress=2)
                predicted = np.clip(model.predict(development[f"x{roi}_{arm}"]), 0., 1.)
                variants.append(dict(
                    roi=roi, arm=arm, model=kind, parameters=best["parameters"],
                    dimension=x.shape[1], cv_mae=best["cv_mae"],
                    development_mae=float(abs(predicted - development["y"]).mean()),
                    model_sha256=sha256(output),
                ))
                write_json(args.output_dir / "development_variants.json", variants)
                print(roi, arm, kind, variants[-1]["development_mae"], flush=True)
    winners = []
    for roi in (7, 10):
        for kind in ("RF", "GP"):
            candidates = [v for v in variants if v["roi"] == roi and v["model"] == kind]
            winners.append(min(candidates, key=lambda v: (v["development_mae"], v["dimension"], v["arm"])))
    write_json(args.output_dir / "frozen_models.json", dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol_sha256=sha256(args.protocol), source_sha256=sha256(Path(__file__)),
        training=training_receipts, development=development_receipts,
        winners=winners, all_variants=variants,
    ))


if __name__ == "__main__":
    main()
