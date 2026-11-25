"""Freeze supplementary calibration, then score locked far-diagnostic candidates."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_composition import independent_far_posterior
from quality_estimation.spatial_risk import (
    calibrate_moments,
    apply_calibration,
    spatial_scores,
)
from study import read, save, sha, load


def predict_candidates(r, bundle, data, schema):
    spec = read(bundle / "feature_recipe.json")
    task = spec["tasks"]["quality_spatial_gp"]
    path = bundle / task["artifact_filename"]
    assert sha(path) == task["model_sha256"]
    primary = SpatialQualityGP.load(path)
    mean, cov = primary.predict(feature_matrix(data, schema, task["ordered_features"]))
    result = {"primary": (mean, cov, dict(primary_sha256=sha(path), far_sha256=None))}
    frozen = read(r / "far_followup/fit/frozen.json")
    records = [
        (
            "far28_GP",
            next(
                v
                for v in frozen["models"]
                if v["kind"] == "GP" and v["view"] == "quality28"
            ),
        ),
        ("far31_GP", frozen["selected"]["GP"]),
        ("far31_RF", frozen["selected"]["RF"]),
    ]
    for name, record in records:
        path = r / "far_followup/fit" / record["model_file"]
        assert sha(path) == record["model_sha256"]
        x = feature_matrix(data, schema, record["ordered_features"])
        if record["kind"] == "GP":
            model = SpatialQualityGP.load(path)
            assert model.feature_names_.tolist() == record["ordered_features"]
            assert str(model.checkpoint_sha256_) == spec["checkpoint_sha256"]
            assert str(model.profile_version_) == spec["profile"]
            assert str(model.regions_sha256_) == spec["regions_sha256"]
            fm, fc = model.predict(x)
            combined, combined_cov = independent_far_posterior(mean, cov, fm, fc)
        else:
            model = joblib.load(path)
            combined = mean.copy()
            combined[:, 2] = model.predict(x)
            combined_cov = None
        context = dict(
            primary_sha256=task["model_sha256"],
            far_sha256=sha(path),
            far_columns=record["ordered_features"],
            far_kind=record["kind"],
            composition="primary Q/near marginal and independent far process, zero cross-block covariance",
        )
        result[name] = (combined, combined_cov, context)
    return result


def context_hash(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def metrics(mean, cov, data):
    y = data["Y"]
    sigma = (
        np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
        if cov is not None
        else None
    )
    result = {}
    for j, name in enumerate(["IoU", "near_recall", "far_recall"]):
        valid = np.isfinite(y[:, j])
        if not valid.any():
            result[name] = dict(rows=0, mae=None, bias=None)
            continue
        error = mean[valid, j] - y[valid, j]
        result[name] = dict(
            rows=int(valid.sum()),
            mae=float(abs(np.clip(mean[valid, j], 0, 1) - y[valid, j]).mean()),
            bias=float(error.mean()),
        )
        if sigma is not None:
            result[name].update(
                coverage90=float(
                    (abs(error) <= 1.6448536269514722 * sigma[valid, j]).mean()
                ),
                mean_width90=float((2 * 1.6448536269514722 * sigma[valid, j]).mean()),
            )
    support = (data["near_gt"] >= 50) & (data["far_gt"] >= 50) & np.isfinite(y).all(1)
    far = (y[:, 0] >= 0.7) & (y[:, 1] >= 0.8) & (y[:, 2] < 0.5)
    result["supported_rows"] = int(support.sum())
    result["normal_far_count"] = int(far[support].sum())
    if cov is not None:
        score = spatial_scores(mean, cov)["normal_far_only"]
        mask = support
        result["normal_far_probability"] = dict(
            auc=float(roc_auc_score(far[mask], score[mask]))
            if len(set(far[mask])) == 2
            else None,
            brier=float(((score[mask] - far[mask]) ** 2).mean())
            if mask.any()
            else None,
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--phase", choices=["calibrate", "score"], required=True)
    ap.add_argument("--key")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    r = a.study
    root = r / "far_followup"
    calroot = root / "calibration"
    if a.phase == "calibrate":
        calroot.mkdir(exist_ok=False)
        calibrators = {}
        for label, keys in [
            ("original96", ["release26_calibration"]),
            ("supported192", ["release26_calibration", "release26_far_calibration"]),
        ]:
            data, s = load(r / "packs", keys)
            values = predict_candidates(r, a.bundle, data, s)
            calibrators[label] = {}
            for name, (mean, cov, context) in values.items():
                if cov is None:
                    continue
                calibration = calibrate_moments(mean, cov, data["Y"])
                calibration.update(
                    context_sha256=context_hash(context),
                    context=context,
                    keys=keys,
                    rows=len(data["ids"]),
                    sites=len(set(data["site"])),
                    simulator_release="26.R1",
                    source_pack_sha256={
                        key: sha(r / "packs" / (key + ".npz")) for key in keys
                    },
                    scope="Empirical moments on intentionally enriched calibration sites; no exchangeability or OOD coverage guarantee.",
                )
                namefile = label + "_" + name + ".json"
                save(calroot / namefile, calibration)
                calibrators[label][name] = dict(
                    file=namefile,
                    sha256=sha(calroot / namefile),
                    context_sha256=context_hash(context),
                )
        frozen = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            fit_sha256=sha(root / "fit/frozen.json"),
            primary_recipe_sha256=sha(a.bundle / "feature_recipe.json"),
            test_plan_sha256=sha(root / "test_plan.json"),
            calibrators=calibrators,
            test_scores_opened=False,
            source_sha256=sha(Path(__file__)),
        )
        save(calroot / "frozen.json", frozen)
        print("calibration frozen before supplementary test scoring", flush=True)
    else:
        if not a.key or a.output is None:
            ap.error("score requires key and new output")
        frozen = read(calroot / "frozen.json")
        assert sha(root / "fit/frozen.json") == frozen["fit_sha256"]
        assert sha(a.bundle / "feature_recipe.json") == frozen["primary_recipe_sha256"]
        data, s = load(r / "packs", [a.key])
        values = predict_candidates(r, a.bundle, data, s)
        a.output.mkdir(exist_ok=False)
        summary = []
        for name, (mean, cov, context) in values.items():
            row = dict(name=name, raw=metrics(mean, cov, data))
            if cov is not None:
                for label, records in frozen["calibrators"].items():
                    path = calroot / records[name]["file"]
                    assert sha(path) == records[name]["sha256"]
                    calibration = read(path)
                    assert context_hash(context) == calibration["context_sha256"]
                    cm, cc = apply_calibration(mean, cov, calibration)
                    row[label] = metrics(cm, cc, data)
                np.savez_compressed(
                    a.output / (name + ".npz"),
                    ids=data["ids"],
                    mean=mean,
                    covariance=cov,
                )
            else:
                np.savez_compressed(
                    a.output / (name + ".npz"), ids=data["ids"], mean=mean
                )
            summary.append(row)
        save(a.output / "summary.json", summary)
        save(
            a.output / "receipt.json",
            dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                key=a.key,
                calibration_freeze_sha256=sha(calroot / "frozen.json"),
                source_pack_sha256=sha(r / "packs" / (a.key + ".npz")),
                source_sha256=sha(Path(__file__)),
                rows=len(data["ids"]),
                sites=len(set(data["site"])),
                scope="Prospective supplementary test after candidate/calibration freeze"
                if a.key == "release26_followup_test"
                else "Previously opened diagnostic data; not new independent validation",
            ),
        )
        print(
            "scored",
            a.key,
            [(v["name"], v["raw"]["far_recall"]["mae"]) for v in summary],
            flush=True,
        )


if __name__ == "__main__":
    main()
