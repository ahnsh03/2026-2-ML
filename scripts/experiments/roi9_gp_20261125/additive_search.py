"""Bounded BO/random comparison for appearance-plus-prediction additive GP kernels."""

from __future__ import annotations
import argparse, itertools, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.bounded_bo import next_expected_improvement
from study import load, read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("packs", "coverage", "protocol", "features", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=True)
    (a.output / "models").mkdir(exist_ok=True)
    p = read(a.protocol)
    columns = read(a.features)["ordered_features"]
    selection = read(a.coverage / "selection.json")["ids"]
    data, s = load(
        a.packs, p["training_keys"] + p["augmentation_pool"] + p["motion_training_keys"]
    )
    lookup = {sid: i for i, sid in enumerate(data["ids"])}
    ix = np.array([lookup[sid] for sid in selection])
    complete = np.isfinite(data["Y"][ix]).all(1)
    ix = ix[complete]
    x = feature_matrix(data, s, columns)[ix]
    y = data["Y"][ix]
    xy = data["xy"][ix]
    dev, _ = load(a.packs, p["development_keys"])
    dx = feature_matrix(dev, s, columns)
    groups = ["appearance" if name in s["rgb"] else "prediction" for name in columns]
    locations, inv = np.unique(xy, axis=0, return_inverse=True)
    clusters = KMeans(n_clusters=6, n_init=10, random_state=p["seed"]).fit_predict(
        locations
    )[inv]
    folds = []
    for fold in range(3):
        held = clusters % 3 == fold
        fit = (~held) & (cKDTree(xy[held]).query(xy)[0] >= 25)
        folds.append((fit, held))
    grid = list(
        itertools.product(
            [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0],
            [0.01, 0.02, 0.05, 0.1, 0.15, 0.3],
            [0.05, 0.15, 0.3, 0.5, 0.7],
            [0.0, 0.4, 0.7],
        )
    )
    coords = np.asarray([[np.log(l), np.log(n), w, r] for l, n, w, r in grid])
    coords = (coords - coords.min(0)) / np.ptp(coords, axis=0)
    initial = [
        grid.index(v)
        for v in [
            (2.8, 0.05, 0.3, 0.4),
            (1.4, 0.05, 0.3, 0.0),
            (5.6, 0.05, 0.3, 0.4),
            (2.8, 0.15, 0.5, 0.4),
            (2.8, 0.02, 0.15, 0.4),
            (4.0, 0.1, 0.7, 0.7),
        ]
    ]
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        grid=grid,
        initial=initial,
        budget_per_method=18,
        objective="mean geographic-fold MAE(Q)+0.15*(MAE(near recall)+MAE(far recall))",
        fold_protocol="Same six training-coordinate clusters and25m embargo as coverage experiment.",
        selection_sha256=sha(a.coverage / "selection.json"),
        columns=columns,
        source_sha256=sha(Path(__file__)),
        model_source_sha256=sha(
            Path(__file__).parents[3] / "src/quality_estimation/spatial_gp.py"
        ),
        scope="Follow-up motivated by already-open archival road diagnostics. No 26R1 test scores accessed. BO and random have the same initial six points and each eighteen completed trials.",
    )
    manifest_path = a.output / "protocol.json"
    if not manifest_path.exists():
        save(manifest_path, manifest)
    else:
        original = read(manifest_path)
        for key in [
            "grid",
            "initial",
            "selection_sha256",
            "columns",
            "source_sha256",
            "model_source_sha256",
        ]:
            assert original[key] == manifest[key]
    progress_path = a.output / "progress.json"
    progress = read(progress_path) if progress_path.exists() else []
    cache = {r["index"]: r for r in progress}
    summaries = []
    random_order = np.random.default_rng(p["seed"]).permutation(len(grid)).tolist()
    for method in ["bo", "random"]:
        indices = []
        losses = []
        trace = []
        for step in range(18):
            index = (
                next_expected_improvement(coords, indices, losses, initial)
                if method == "bo"
                else (
                    initial[step]
                    if step < len(initial)
                    else next(i for i in random_order if i not in indices)
                )
            )
            if index not in cache:
                length, noise, weight, rho = grid[index]
                params = dict(
                    length_multiplier=length,
                    noise_variance=noise,
                    task_correlation=rho,
                    feature_groups=groups,
                    kernel_mode="additive",
                    kernel_group_weights=[weight, 1 - weight],
                )
                start = time.monotonic()
                fold_losses = []
                for fit, held in folds:
                    model = SpatialQualityGP(**params).fit(x[fit], y[fit])
                    mean, _ = model.predict(x[held])
                    fold_losses.append(
                        abs(np.clip(mean, 0, 1) - y[held]).mean(0).tolist()
                    )
                cv = np.mean(fold_losses, 0)
                model = SpatialQualityGP(**params).fit(x, y)
                mean, _ = model.predict(dx)
                row = dict(
                    index=index,
                    parameters=params,
                    cv=cv.tolist(),
                    folds=fold_losses,
                    objective=float(cv[0] + 0.15 * (cv[1] + cv[2])),
                    development=abs(np.clip(mean, 0, 1) - dev["Y"]).mean(0).tolist(),
                    seconds=time.monotonic() - start,
                )
                cache[index] = row
                progress.append(row)
                save(progress_path, progress)
            row = cache[index]
            indices.append(index)
            losses.append(row["objective"])
            trace.append(
                dict(
                    step=step,
                    index=index,
                    objective=row["objective"],
                    best_objective=min(losses),
                )
            )
            print(method, step, index, row["cv"], row["development"], flush=True)
        rows = [cache[i] for i in indices]
        qmin = min(r["cv"][0] for r in rows)
        dmin = min(r["development"][0] for r in rows)
        eligible = [
            r
            for r in rows
            if r["cv"][0] <= qmin + 0.002 and r["development"][0] <= dmin + 0.003
        ]
        if not eligible:
            eligible = [min(rows, key=lambda r: r["development"][0])]
        best = min(eligible, key=lambda r: r["objective"])
        model = SpatialQualityGP(**best["parameters"]).fit(x, y)
        path = a.output / "models" / ("additive_" + method + ".npz")
        if not path.exists():
            model.save(path)
        summaries.append(
            dict(
                best,
                name="additive_" + method,
                kind="SpatialGP",
                rows=len(y),
                dimensions=len(columns),
                ordered_features=columns,
                model_file="models/" + path.name,
                model_sha256=sha(path),
                trace=trace,
            )
        )
        save(a.output / "summary.json", summaries)
    selected = min(summaries, key=lambda r: r["development"][0])
    save(
        a.output / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            selected=selected["name"],
            models=summaries,
            columns=columns,
            dimensions=len(columns),
            protocol_sha256=sha(manifest_path),
            scope="Hyperparameters selected using training geographic CV and original development only. New 26R1 scores still closed. Compare BO/random traces without claiming a search win from a single seed.",
        ),
    )


if __name__ == "__main__":
    main()
