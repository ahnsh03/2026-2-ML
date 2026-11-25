"""Bounded target-specific far-recall GP/RF study; the primary IoU model stays frozen."""

from __future__ import annotations
import argparse, itertools, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import joblib
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from study import read, save, sha, load


def view_columns(columns):
    head = {
        "drivable_fraction",
        "lane_fraction",
        "lane_probability",
        "lane_weighted_entropy",
        "lane_weighted_probability",
    }
    far = [
        n.replace("near_left.", "far_left.").replace("near_right.", "far_right.")
        if n.split(".")[-1] in head and n.startswith(("near_left.", "near_right."))
        else n
        for n in columns
    ]
    return dict(
        quality28=columns,
        quality28_plus_pooled_morph6=columns
        + [
            f"{band}.{name}"
            for band in ["far", "near"]
            for name in ["row_coverage", "internal_gap", "soft_outside"]
        ],
        far_output28_plus_far_morph3=far
        + [f"far.{name}" for name in ["row_coverage", "internal_gap", "soft_outside"]],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["study", "bundle", "output"]:
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    out = a.output
    out.mkdir(exist_ok=False)
    (out / "models").mkdir()
    protocol = read(r / "far_followup/protocol.json")
    membership = read(a.bundle / "training_membership.json")
    p = read(r / "coverage_protocol.json")
    keys = p["training_keys"] + p["augmentation_pool"] + p["motion_training_keys"]
    data, s = load(r / "packs", keys)
    lookup = {sid: i for i, sid in enumerate(data["ids"])}
    indices = np.asarray([lookup[v["sample_id"]] for v in membership["rows"]])
    y = data["Y"][indices, 2]
    xy = data["xy"][indices]
    dev, _ = load(r / "packs", p["development_keys"])
    locations, inverse = np.unique(xy, axis=0, return_inverse=True)
    groups = KMeans(n_clusters=6, n_init=10, random_state=23411).fit_predict(locations)[
        inverse
    ]
    folds = []
    audit = []
    for fold in range(3):
        held = groups % 3 == fold
        distance = cKDTree(xy[held]).query(xy)[0]
        fit = ~held & (distance >= 25)
        folds.append((fit, held))
        audit.append(
            dict(
                fit=int(fit.sum()),
                held=int(held.sum()),
                minimum_distance=float(distance[fit].min()),
            )
        )
    save(
        out / "folds.json",
        dict(folds=audit, membership_sha256=sha(a.bundle / "training_membership.json")),
    )
    recipe = read(a.bundle / "feature_recipe.json")
    views = view_columns(recipe["tasks"]["quality_spatial_gp"]["ordered_features"])
    save(out / "views.json", views)
    progress = []
    summaries = []
    for name, columns in views.items():
        x = feature_matrix(data, s, columns)[indices]
        dx = feature_matrix(dev, s, columns)
        for family in ["GP", "RF"]:
            settings = (
                [
                    dict(length_multiplier=l, noise_variance=n, task_correlation=0.0)
                    for l, n in itertools.product(
                        [1.4, 2.8, 5.6], [0.02, 0.05, 0.15, 0.3]
                    )
                ]
                if family == "GP"
                else [
                    dict(
                        n_estimators=128,
                        min_samples_leaf=leaf,
                        max_features=f,
                        max_depth=d,
                        random_state=23411,
                        n_jobs=2,
                    )
                    for leaf, f, d in itertools.product(
                        [2, 5, 10], [0.5, 1.0], [12, None]
                    )
                ]
            )
            trials = []
            for j, parameters in enumerate(settings):
                if datetime.now(timezone.utc) >= datetime.fromisoformat(
                    protocol["deadline_utc"]
                ):
                    raise TimeoutError("follow-up deadline")
                started = time.monotonic()
                scores = []
                for fit, held in folds:
                    model = (
                        SpatialQualityGP(**parameters)
                        if family == "GP"
                        else RandomForestRegressor(**parameters)
                    )
                    model.fit(x[fit], y[fit])
                    prediction = model.predict(x[held])
                    mean = prediction[0][:, 0] if family == "GP" else prediction
                    scores.append(float(abs(np.clip(mean, 0, 1) - y[held]).mean()))
                item = dict(
                    view=name,
                    kind=family,
                    parameters=parameters,
                    cv=float(np.mean(scores)),
                    folds=scores,
                    seconds=time.monotonic() - started,
                )
                trials.append(item)
                progress.append(item)
                save(out / "progress.json", progress)
            best = min(trials, key=lambda v: v["cv"])
            model = (
                SpatialQualityGP(**best["parameters"])
                if family == "GP"
                else RandomForestRegressor(**best["parameters"])
            )
            model.fit(x, y)
            filename = f"{name}_{family}." + ("npz" if family == "GP" else "joblib")
            path = out / "models" / filename
            if family == "GP":
                for key, value in dict(
                    feature_names_=columns,
                    profile_version_=recipe["profile"],
                    regions_sha256_=recipe["regions_sha256"],
                    checkpoint_sha256_=recipe["checkpoint_sha256"],
                    target_names_=["far_lane_recall"],
                ).items():
                    setattr(model, key, np.asarray(value))
                model.save(path)
                mean, cov = model.predict(dx)
                prediction = mean[:, 0]
            else:
                joblib.dump(model, path)
                prediction = model.predict(dx)
            valid = np.isfinite(dev["Y"][:, 2])
            result = dict(
                best,
                ordered_features=columns,
                dimensions=len(columns),
                training_rows=len(y),
                development_mae=float(
                    abs(np.clip(prediction[valid], 0, 1) - dev["Y"][valid, 2]).mean()
                ),
                model_file=str(path.relative_to(out)),
                model_sha256=sha(path),
            )
            summaries.append(result)
            save(out / "summary.json", summaries)
            print(name, family, result["cv"], result["development_mae"], flush=True)
    selected = {
        family: min(
            [v for v in summaries if v["kind"] == family],
            key=lambda v: (v["cv"], v["dimensions"], v["view"]),
        )
        for family in ["GP", "RF"]
    }
    save(
        out / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            selected=selected,
            models=summaries,
            protocol_sha256=sha(r / "far_followup/protocol.json"),
            source_sha256=sha(Path(__file__)),
            primary_recipe_sha256=sha(a.bundle / "feature_recipe.json"),
            scope="Target-specific diagnostic candidates selected by training CV only; development reported after refit. No new26R1 calibration or prospective test labels were read. Primary28-featureQ model remains unchanged.",
        ),
    )


if __name__ == "__main__":
    main()
