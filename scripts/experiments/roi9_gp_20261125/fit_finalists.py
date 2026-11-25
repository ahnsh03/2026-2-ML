"""Tune spatial task coupling and replay input-only GP label acquisition before external scoring."""

from __future__ import annotations
import argparse, itertools, json, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import conditional_batch
from study import load, read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packs", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    models = a.output / "models"
    models.mkdir(exist_ok=True)
    p = read(a.protocol)
    spec = read(a.features)
    columns = spec["ordered_features"]
    train, s = load(a.packs, p["training_keys"])
    dev, _ = load(a.packs, p["development_keys"])
    x = feature_matrix(train, s, columns)
    dx = feature_matrix(dev, s, columns)
    y = train["Y"]
    if not np.isfinite(y).all():
        raise ValueError("this complete-task protocol requires defined targets")
    folds = [
        np.isin(train["group"], np.asarray(p["training_keys"])[i::3]) for i in range(3)
    ]
    records = []
    length = spec["parameters"]["length_multiplier"]
    for noise, rho in itertools.product([0.01, 0.05, 0.15], [0.0, 0.3, 0.6, 0.9]):
        params = dict(
            length_multiplier=length, noise_variance=noise, task_correlation=rho
        )
        losses = []
        for held in folds:
            model = SpatialQualityGP(**params).fit(x[~held], y[~held])
            mean, _ = model.predict(x[held])
            losses.append(abs(np.clip(mean, 0, 1) - y[held]).mean(0))
        model = SpatialQualityGP(**params).fit(x, y)
        mean, _ = model.predict(dx)
        name = f"coupling_n{noise}_r{rho}"
        path = models / (name + ".npz")
        if not path.exists():
            model.save(path)
        record = dict(
            name=name,
            parameters=params,
            cv=np.mean(losses, 0).tolist(),
            development=abs(np.clip(mean, 0, 1) - dev["Y"]).mean(0).tolist(),
            model_file=str(path.relative_to(a.output)),
            model_sha256=sha(path),
        )
        records.append(record)
        save(a.output / "coupling.json", records)
        print(record, flush=True)
    cv_best = min(r["cv"][0] for r in records)
    dev_best = min(r["development"][0] for r in records)
    eligible = [
        r
        for r in records
        if r["cv"][0] <= cv_best + 0.002 and r["development"][0] <= dev_best + 0.002
    ]
    selected = min(
        eligible, key=lambda r: r["cv"][0] + 0.25 * (r["cv"][1] + r["cv"][2])
    )
    params = selected["parameters"]
    base = SpatialQualityGP.load(a.output / selected["model_file"])
    poolkeys = p["augmentation_pool"]
    parts = []
    poolids = []
    source_hashes = {}
    # Only X/IDs are read for acquisition. Pool GT is opened below after selection is sealed.
    for key in poolkeys:
        path = a.packs / (key + ".npz")
        source_hashes[key] = sha(path)
        with np.load(path) as z:
            parts.append(feature_matrix(z, s, columns))
            poolids.extend(z["ids"].tolist())
    px = np.concatenate(parts)
    poolids = np.asarray(poolids)
    budget = p["posterior_label_budget"]
    covariance = base.posterior_covariance(px)
    noise = params["noise_variance"] * float(base.y_scale_[0]) ** 2
    chosen = conditional_batch(
        np.ones(len(px)), covariance, budget, poolids, noise_variance=noise
    )
    indices = {
        "base": np.array([], int),
        "conditional192": chosen,
        "all624": np.arange(len(px)),
    }
    for seed in [23411, 23412, 23413]:
        indices["random192_s" + str(seed)] = np.sort(
            np.random.default_rng(seed).choice(len(px), budget, replace=False)
        )
    acquisition = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        pool_keys=poolkeys,
        pool_sources=source_hashes,
        base_model_sha256=selected["model_sha256"],
        columns=columns,
        scope="Input-only posterior covariance acquisition replay on previously opened diagnostic frames; pool labels enter only after IDs are recorded",
        selections={name: poolids[ix].tolist() for name, ix in indices.items()},
        model_parameters=params,
        noise_variance_iou=noise,
        protocol_sha256=sha(a.protocol),
    )
    save(a.output / "acquisition.json", acquisition)
    pool, _ = load(a.packs, poolkeys)
    assert np.array_equal(poolids, pool["ids"])
    assert np.isfinite(pool["Y"]).all() and not set(poolids) & set(train["ids"])
    # RF receives the same X and primary labels; choose its settings on the same base folds.
    rf_records = []
    for leaf, ratio, depth in itertools.product([2, 4, 8], [0.5, 1.0], [12, None]):
        setting = dict(
            n_estimators=128, min_samples_leaf=leaf, max_features=ratio, max_depth=depth
        )
        losses = []
        for held in folds:
            m = RandomForestRegressor(**setting, random_state=p["seed"], n_jobs=4).fit(
                x[~held], y[~held, 0]
            )
            losses.append(float(abs(m.predict(x[held]) - y[held, 0]).mean()))
        rf_records.append(
            dict(parameters=setting, cv=float(np.mean(losses)), folds=losses)
        )
    rf_parameters = min(rf_records, key=lambda r: r["cv"])["parameters"]
    save(a.output / "rf_tuning.json", rf_records)
    results = []
    for name, ix in indices.items():
        xx = np.concatenate([x, px[ix]])
        yy = np.concatenate([y, pool["Y"][ix]])
        ids = np.concatenate([train["ids"], poolids[ix]])
        gp = SpatialQualityGP(**params).fit(xx, yy)
        gp_path = models / (name + "_GP.npz")
        gp.save(gp_path)
        mean, cov = gp.predict(dx)
        gp_record = dict(
            name=name + "_GP",
            kind="SpatialGP",
            variant=name,
            rows=len(yy),
            parameters=params,
            development_mae=abs(np.clip(mean, 0, 1) - dev["Y"]).mean(0).tolist(),
            model_file=str(gp_path.relative_to(a.output)),
            model_sha256=sha(gp_path),
            ordered_features=columns,
            dimensions=len(columns),
            training_ids_sha256=__import__("hashlib")
            .sha256("\n".join(ids).encode())
            .hexdigest(),
        )
        results.append(gp_record)
        rf = RandomForestRegressor(
            **rf_parameters, random_state=p["seed"], n_jobs=4
        ).fit(xx, yy[:, 0])
        rf_path = models / (name + "_RF.joblib")
        joblib.dump(rf, rf_path, compress=2)
        results.append(
            dict(
                gp_record,
                name=name + "_RF",
                kind="RF",
                parameters=rf_parameters,
                development_mae=[float(abs(rf.predict(dx) - dev["y"]).mean())],
                model_file=str(rf_path.relative_to(a.output)),
                model_sha256=sha(rf_path),
            )
        )
        auxiliary = RandomForestRegressor(
            **rf_parameters, random_state=p["seed"], n_jobs=4
        ).fit(xx, yy)
        path = models / (name + "_RFaux.joblib")
        joblib.dump(auxiliary, path, compress=2)
        results.append(
            dict(
                gp_record,
                name=name + "_RFaux",
                kind="RFaux",
                parameters=rf_parameters,
                development_mae=abs(auxiliary.predict(dx) - dev["Y"]).mean(0).tolist(),
                model_file=str(path.relative_to(a.output)),
                model_sha256=sha(path),
            )
        )
        save(a.output / "summary.json", results)
        print("augmentation", name, len(yy), gp_record["development_mae"], flush=True)
    best_dev = min(r["development_mae"][0] for r in results if r["kind"] == "SpatialGP")
    full = next(r for r in results if r["name"] == "all624_GP")
    # More diagnosed conditions are preferred when the development difference is small.
    chosen_model = (
        full
        if full["development_mae"][0] <= best_dev + 0.001
        else min(
            [r for r in results if r["kind"] == "SpatialGP"],
            key=lambda r: r["development_mae"][0],
        )
    )
    freeze = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        selected=chosen_model["name"],
        selection_scope="Feature and model selection use historical training/development only; external archive scores not inspected",
        selected_coupling=selected,
        models=results,
        columns=columns,
        dimensions=len(columns),
        source_sha256=sha(Path(__file__)),
        model_source_sha256=sha(
            Path(__file__).parents[3] / "src/quality_estimation/spatial_gp.py"
        ),
        acquisition_sha256=sha(a.output / "acquisition.json"),
        protocol_sha256=sha(a.protocol),
    )
    save(a.output / "frozen.json", freeze)
    print("FROZEN", freeze["selected"], flush=True)


if __name__ == "__main__":
    main()
