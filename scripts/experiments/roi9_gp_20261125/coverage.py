"""Input-stratified quality-training coverage study, with buffered geographic CV.

Historical diagnostic images may be promoted to training, never counted as test
afterward. Perception weights, the 28-feature recipe, and GT definitions stay fixed.
"""

from __future__ import annotations
import argparse, itertools, json, hashlib, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from study import load, read, save, sha, feature_groups


def choose_rows(packs, protocol):
    selected = []
    reasons = {}
    protected = []
    for key in protocol["development_keys"] + protocol["calibration_keys"]:
        with np.load(packs / (key + ".npz")) as z:
            protected.extend(z["xy"])
    tree = cKDTree(np.unique(protected, axis=0))
    # 32 of the 48 scheduled condition images at each original training pose.
    for key in protocol["training_keys"]:
        with np.load(packs / (key + ".npz")) as z:
            for site in np.unique(z["site"]):
                indices = np.flatnonzero(z["site"] == site)
                order = sorted(
                    indices,
                    key=lambda i: hashlib.sha256(
                        ("coverage23411:" + str(z["ids"][i])).encode()
                    ).hexdigest(),
                )
                assert len(order) == 48
                for i in order[:32]:
                    selected.append(str(z["ids"][i]))
                    reasons[str(z["ids"][i])] = "base_pose_condition32"
    for key in protocol["augmentation_pool"]:
        with np.load(packs / (key + ".npz")) as z:
            assert tree.query(z["xy"])[0].min() >= 25
            for sid in z["ids"]:
                selected.append(str(sid))
                reasons[str(sid)] = "diagnostic_pose_augmentation"
    for key in protocol["motion_training_keys"]:
        with np.load(packs / (key + ".npz")) as z:
            eligible = np.flatnonzero(tree.query(z["xy"])[0] >= 25)
            # Uniformly spaced in recorded driving order, after protected-location exclusion.
            assert len(eligible) >= 30
            indices = eligible[
                np.linspace(0, len(eligible) - 1, 30).round().astype(int)
            ]
            for i in indices:
                selected.append(str(z["ids"][i]))
                reasons[str(z["ids"][i])] = "driving_coverage30"
    assert len(selected) == len(set(selected)) == 2496
    return selected, reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("packs", "protocol", "features", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=True)
    (a.output / "models").mkdir(exist_ok=True)
    p = read(a.protocol)
    columns = read(a.features)["ordered_features"]
    selected, reasons = choose_rows(a.packs, p)
    selection = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        ids=selected,
        reasons=reasons,
        protocol_sha256=sha(a.protocol),
        source_sha256=sha(Path(__file__)),
        role="Previously opened diagnostics promoted to quality training; RGB/metadata-only sample selection before this fit stage opens Y. New simulator-release test scores remain closed.",
    )
    selection_path = a.output / "selection.json"
    if selection_path.exists():
        assert read(selection_path)["ids"] == selected
    else:
        save(selection_path, selection)
    data, s = load(
        a.packs, p["training_keys"] + p["augmentation_pool"] + p["motion_training_keys"]
    )
    lookup = {sid: i for i, sid in enumerate(data["ids"])}
    ix = np.array([lookup[sid] for sid in selected])
    x = feature_matrix(data, s, columns)[ix]
    y = data["Y"][ix]
    xy = data["xy"][ix]
    complete = np.isfinite(y).all(1)
    save(
        a.output / "label_eligibility.json",
        dict(
            selected_rows=len(y),
            fit_rows=int(complete.sum()),
            excluded_ids=np.asarray(selected)[~complete].tolist(),
            missing_by_task=(~np.isfinite(y)).sum(0).tolist(),
            reason="Complete three-task GP needs defined near/far recall. Empty GT bands are undefined, never filled with zero. All compared quality models use the same eligible rows.",
        ),
    )
    x = x[complete]
    y = y[complete]
    xy = xy[complete]
    dev, _ = load(a.packs, p["development_keys"])
    dx = feature_matrix(dev, s, columns)
    locations, inv = np.unique(xy, axis=0, return_inverse=True)
    clusters = KMeans(n_clusters=6, n_init=10, random_state=p["seed"]).fit_predict(
        locations
    )[inv]
    fold_pairs = []
    fold_audit = []
    for fold in range(3):
        held = clusters % 3 == fold
        distance = cKDTree(xy[held]).query(xy)[0]
        fit = (~held) & (distance >= 25)
        assert fit.sum() > 500 and held.sum() > 100
        fold_pairs.append((fit, held))
        fold_audit.append(
            dict(
                fold=fold,
                fit=int(fit.sum()),
                held=int(held.sum()),
                embargoed=int(((~held) & (~fit)).sum()),
                minimum_distance=float(distance[fit].min()),
            )
        )
    save(
        a.output / "folds.json",
        dict(folds=fold_audit, selection_sha256=sha(selection_path)),
    )
    summary = []
    progress_path = a.output / "progress.json"
    progress = read(progress_path) if progress_path.exists() else []
    done = {r["id"]: r for r in progress}
    for balanced in (False, True):
        groups = feature_groups(columns, s) if balanced else None
        rows = []
        for j, (length, noise, rho) in enumerate(
            itertools.product([1.4, 2.8, 5.6], [0.02, 0.05, 0.15], [0.0, 0.4])
        ):
            identity = f"GP_b{int(balanced)}_{j}"
            params = dict(
                length_multiplier=length, noise_variance=noise, task_correlation=rho
            )
            if identity not in done:
                start = time.monotonic()
                losses = []
                for fit, held in fold_pairs:
                    model = SpatialQualityGP(**params, feature_groups=groups).fit(
                        x[fit], y[fit]
                    )
                    mean, _ = model.predict(x[held])
                    losses.append(abs(np.clip(mean, 0, 1) - y[held]).mean(0).tolist())
                m = SpatialQualityGP(**params, feature_groups=groups).fit(x, y)
                dm, _ = m.predict(dx)
                row = dict(
                    id=identity,
                    parameters=params,
                    balanced=balanced,
                    cv=np.mean(losses, 0).tolist(),
                    folds=losses,
                    development=abs(np.clip(dm, 0, 1) - dev["Y"]).mean(0).tolist(),
                    seconds=time.monotonic() - start,
                )
                progress.append(row)
                done[identity] = row
                save(progress_path, progress)
                print(identity, row["cv"], row["development"], flush=True)
            rows.append(done[identity])
        minimum_cv = min(r["cv"][0] for r in rows)
        minimum_dev = min(r["development"][0] for r in rows)
        eligible = [
            r
            for r in rows
            if r["cv"][0] <= minimum_cv + 0.002
            and r["development"][0] <= minimum_dev + 0.003
        ]
        if not eligible:
            eligible = [min(rows, key=lambda r: r["development"][0])]
        best = min(
            eligible, key=lambda r: r["cv"][0] + 0.25 * (r["cv"][1] + r["cv"][2])
        )
        model = SpatialQualityGP(**best["parameters"], feature_groups=groups).fit(x, y)
        path = a.output / "models" / (best["id"] + ".npz")
        if not path.exists():
            model.save(path)
        summary.append(
            dict(
                best,
                name="coverage_" + best["id"],
                kind="SpatialGP",
                rows=len(y),
                dimensions=len(columns),
                ordered_features=columns,
                model_file="models/" + path.name,
                model_sha256=sha(path),
            )
        )
        save(a.output / "summary.json", summary)
    rows = []
    for leaf, ratio, depth in itertools.product([2, 4, 8], [0.5, 1.0], [12, None]):
        params = dict(
            n_estimators=128, min_samples_leaf=leaf, max_features=ratio, max_depth=depth
        )
        losses = []
        for fit, held in fold_pairs:
            m = RandomForestRegressor(**params, random_state=p["seed"], n_jobs=4).fit(
                x[fit], y[fit, 0]
            )
            losses.append(float(abs(m.predict(x[held]) - y[held, 0]).mean()))
        model = RandomForestRegressor(**params, random_state=p["seed"], n_jobs=4).fit(
            x, y[:, 0]
        )
        rows.append(
            dict(
                parameters=params,
                cv=float(np.mean(losses)),
                folds=losses,
                development=float(abs(model.predict(dx) - dev["y"]).mean()),
            )
        )
    best = min(rows, key=lambda r: r["cv"])
    save(a.output / "rf_tuning.json", rows)
    for auxiliary in (False, True):
        model = RandomForestRegressor(
            **best["parameters"], random_state=p["seed"], n_jobs=4
        ).fit(x, y if auxiliary else y[:, 0])
        name = "coverage_RFaux" if auxiliary else "coverage_RF"
        path = a.output / "models" / (name + ".joblib")
        joblib.dump(model, path, compress=2)
        summary.append(
            dict(
                best,
                name=name,
                kind="RFaux" if auxiliary else "RF",
                rows=len(y),
                dimensions=len(columns),
                ordered_features=columns,
                model_file="models/" + path.name,
                model_sha256=sha(path),
            )
        )
    save(a.output / "summary.json", summary)
    selected_model = min(
        [r for r in summary if r["kind"] == "SpatialGP"],
        key=lambda r: r["development"][0],
    )
    save(
        a.output / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            selected=selected_model["name"],
            models=summary,
            columns=columns,
            dimensions=len(columns),
            selection_sha256=sha(selection_path),
            protocol_sha256=sha(a.protocol),
            source_sha256=sha(Path(__file__)),
            scope="Buffered geographic CV and original development groups only. Earlier archive metrics already opened and are diagnostic. New 26R1 scores remain closed.",
        ),
    )
    print("FROZEN", selected_model["name"], flush=True)


if __name__ == "__main__":
    main()
