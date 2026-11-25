"""Bounded, grouped feature comparisons and spatial GP fitting on nine-region packs."""

from __future__ import annotations
import argparse, hashlib, itertools, json, time
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from quality_estimation.region_feature_views import feature_matrix, quality_views
from quality_estimation.spatial_gp import SpatialQualityGP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load(root, keys):
    schema = read(root / "schema.json")
    parts = {}
    for key in keys:
        path = root / (key + ".npz")
        receipt = read(path.with_suffix(".json"))
        assert (
            sha(path) == receipt["sha256"]
            and sha(root / "schema.json") == receipt["schema_sha256"]
        )
        with np.load(path) as z:
            for name in z.files:
                parts.setdefault(name, []).append(z[name])
    pack = {k: np.concatenate(v) for k, v in parts.items()}
    assert len(set(pack["ids"])) == len(pack["ids"])
    c = pack["counts"]
    far = c[:, 4:6].sum(1)
    near = c[:, 6:8].sum(1)
    fs = far[:, 0] + far[:, 2]
    ns = near[:, 0] + near[:, 2]
    fr = np.divide(far[:, 0], fs, out=np.full(len(fs), np.nan), where=fs > 0)
    nr = np.divide(near[:, 0], ns, out=np.full(len(ns), np.nan), where=ns > 0)
    pack["Y"] = np.column_stack([pack["y"], nr, fr])
    supported = (fs >= 50) & (ns >= 50) & np.isfinite(pack["y"])
    # Spatial taxonomy is separate from the primary whole-frame IoU threshold.
    label = np.full(len(fs), -1)
    label[supported] = 0
    label[supported & (nr < 0.8) & (fr < 0.5)] = 1
    label[supported & (nr >= 0.8) & (fr < 0.5)] = 2
    label[supported & (nr < 0.8) & (fr >= 0.5)] = 3
    pack["error_type"] = label
    pack["far_gt"] = fs
    pack["near_gt"] = ns
    return pack, schema


def views(schema, recipe):
    base = recipe["tasks"]["quality_gp"]["ordered_features"]
    morph = schema["morphology"]
    distance = [
        n
        for n in schema["relations"]
        if n.startswith("near_minus_far.")
        and n.split(".")[1]
        in [
            "lane_fraction",
            "lane_probability",
            "marking_entropy",
            "lane_weighted_entropy",
            "lane_drivable_disagreement",
        ]
    ]
    return dict(
        current27=base,
        global30=base + [n for n in morph if n.startswith("global.")],
        pooled33=base + [n for n in morph if n.startswith(("far.", "near."))],
        quadrants39=base + [n for n in morph if n.startswith(("far_", "near_"))],
        allmorph42=base
        + [n for n in morph if n.startswith(("global.", "far_", "near_"))],
        distance32=base + distance,
        pooled_distance38=base
        + [n for n in morph if n.startswith(("far.", "near."))]
        + distance,
        common123=quality_views(schema)["base120"] + schema["uncertainty"],
        expanded263=schema["rgb"] + schema["output"] + schema["uncertainty"],
    )


def feature_groups(columns, schema):
    return [
        "appearance"
        if n in schema["rgb"]
        else "prediction"
        if n in schema["output"]
        else "confidence"
        if n in schema["uncertainty"]
        else "structure"
        for n in columns
    ]


def diagnostics(mean, y, labels):
    result = dict(mae=np.nanmean(abs(np.clip(mean, 0, 1) - y), 0).tolist())
    valid = np.isin(labels, [1, 2])
    score = mean[:, 1] - mean[:, 2]
    result["far_vs_both_auc"] = (
        float(roc_auc_score(labels[valid] == 2, score[valid]))
        if len(set(labels[valid])) == 2
        else None
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packs", type=Path, required=True)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--views", nargs="+")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--column-views", type=Path)
    ap.add_argument("--gp-only", action="store_true")
    ap.add_argument("--isotropic-only", action="store_true")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "models").mkdir(exist_ok=True)
    p = read(a.protocol)
    train, s = load(a.packs, p["training_keys"])
    dev, _ = load(a.packs, p["development_keys"])
    assert not set(train["pixel_sha256"]) & set(dev["pixel_sha256"])
    candidates = views(s, read(a.recipe))
    names = a.views or list(candidates)
    if a.column_views:
        candidates = read(a.column_views)
        names = a.views or list(candidates)
    manifest = dict(
        protocol_sha256=sha(a.protocol),
        recipe_sha256=sha(a.recipe),
        schema_sha256=sha(a.packs / "schema.json"),
        packs={
            k: sha(a.packs / (k + ".npz"))
            for k in p["training_keys"] + p["development_keys"]
        },
        source_sha256=sha(Path(__file__)),
        model_source_sha256=sha(
            Path(__file__).parents[3] / "src/quality_estimation/spatial_gp.py"
        ),
        column_views_sha256=sha(a.column_views) if a.column_views else None,
        gp_only=a.gp_only,
        isotropic_only=a.isotropic_only,
        views=names,
        train_rows=len(train["y"]),
        dev_rows=len(dev["y"]),
    )
    manifest_path = a.output / "manifest.json"
    if manifest_path.exists():
        assert read(manifest_path) == manifest
    else:
        save(manifest_path, manifest)
    progress_path = a.output / "progress.json"
    progress = read(progress_path) if progress_path.exists() else []
    done = {r["id"]: r for r in progress}
    summary = []
    folds = [
        np.isin(train["group"], np.asarray(p["training_keys"])[i::3]) for i in range(3)
    ]
    for name in names:
        columns = candidates[name]
        x = feature_matrix(train, s, columns)
        dx = feature_matrix(dev, s, columns)
        engines = ([("RF", False)] if not a.gp_only else []) + [("GP", False)]
        if not a.isotropic_only:
            engines.append(("GP", True))
        for kind, balance in engines:
            groups = feature_groups(columns, s) if balance else None
            settings = (
                [
                    dict(
                        n_estimators=128,
                        min_samples_leaf=leaf,
                        max_features=ratio,
                        max_depth=depth,
                    )
                    for leaf, ratio, depth in itertools.product(
                        [2, 4, 8], [0.5, 1.0], [12, None]
                    )
                ]
                if kind == "RF"
                else [
                    dict(length_multiplier=l, noise_variance=n, task_correlation=0.0)
                    for l, n in itertools.product(
                        [0.7, 1.4, 2.8, 5.6], [0.01, 0.05, 0.15]
                    )
                ]
            )
            rows = []
            for j, params in enumerate(settings):
                identity = f"{name}_{kind}_b{int(balance)}_{j}"
                if identity not in done:
                    start = time.monotonic()
                    losses = []
                    for hold in folds:
                        if kind == "RF":
                            model = RandomForestRegressor(
                                **params, random_state=p["seed"], n_jobs=a.jobs
                            ).fit(x[~hold], train["y"][~hold])
                            prediction = model.predict(x[hold])[:, None]
                            truth = train["y"][hold, None]
                        else:
                            valid = np.isfinite(train["Y"]).all(1)
                            fit = ~hold & valid
                            model = SpatialQualityGP(
                                **params, feature_groups=groups
                            ).fit(x[fit], train["Y"][fit])
                            prediction, _ = model.predict(x[hold])
                            truth = train["Y"][hold]
                        losses.append(
                            np.nanmean(
                                abs(np.clip(prediction, 0, 1) - truth), 0
                            ).tolist()
                        )
                    row = dict(
                        id=identity,
                        view=name,
                        kind=kind,
                        balanced=balance,
                        parameters=params,
                        cv=np.mean(losses, 0).tolist(),
                        folds=losses,
                        seconds=time.monotonic() - start,
                    )
                    progress.append(row)
                    done[identity] = row
                    save(progress_path, progress)
                rows.append(done[identity])
            best = min(rows, key=lambda r: (r["cv"][0], r["id"]))
            params = best["parameters"]
            if kind == "RF":
                model = RandomForestRegressor(
                    **params, random_state=p["seed"], n_jobs=a.jobs
                ).fit(x, train["y"])
                path = a.output / "models" / (best["id"] + ".joblib")
                joblib.dump(model, path, compress=2)
                dev_metrics = dict(
                    mae=[float(abs(np.clip(model.predict(dx), 0, 1) - dev["y"]).mean())]
                )
            else:
                valid = np.isfinite(train["Y"]).all(1)
                model = SpatialQualityGP(**params, feature_groups=groups).fit(
                    x[valid], train["Y"][valid]
                )
                path = a.output / "models" / (best["id"] + ".npz")
                if not path.exists():
                    model.save(path)
                else:
                    model = SpatialQualityGP.load(path)
                mean, cov = model.predict(dx)
                dev_metrics = diagnostics(mean, dev["Y"], dev["error_type"])
            summary.append(
                dict(
                    best,
                    ordered_features=columns,
                    dimensions=len(columns),
                    development=dev_metrics,
                    model_file=path.name,
                    model_sha256=sha(path),
                )
            )
            save(a.output / "summary.json", summary)
            print(
                name,
                kind,
                "balanced",
                balance,
                "cv",
                best["cv"],
                "dev",
                dev_metrics,
                flush=True,
            )


if __name__ == "__main__":
    main()
