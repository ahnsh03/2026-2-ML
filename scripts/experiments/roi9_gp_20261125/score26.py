"""Evaluate frozen point models/calibrators on one prospective26R1 cohort."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import apply_calibration, spatial_scores
from study import load, read, save, sha
from evaluate import metrics


def auc(label, score):
    return float(roc_auc_score(label, score)) if len(set(label)) == 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    a.output.mkdir(exist_ok=False)
    freeze = read(r / "calibration26/frozen.json")
    data, s = load(r / "packs", [a.key])
    allmask = np.ones(len(data["y"]), bool)
    calibration, _ = load(r / "packs", ["release26_calibration"])
    assert not set(data["ids"]) & set(calibration["ids"])
    from scipy.spatial import cKDTree

    distance = cKDTree(calibration["xy"]).query(data["xy"])[0]
    records = []
    for record in freeze["models"]:
        root = (
            r / "coverage"
            if record["name"].startswith("coverage_GP")
            else r / "finalists"
            if record["name"] == "base_GP"
            else r
        )
        records.append(dict(record, root=str(root)))
    for directory, names in [
        (r / "coverage", ["coverage_RF", "coverage_RFaux"]),
        (r / "finalists", ["base_RF"]),
    ]:
        records.extend(
            dict(m, root=str(directory))
            for m in read(directory / "frozen.json")["models"]
            if m["name"] in names
        )
    prior_pixels = set()
    for key in read(r / "extraction_index.json"):
        with np.load(r / "packs" / (key + ".npz")) as z:
            prior_pixels.update(z["pixel_sha256"])
    lock = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        key=a.key,
        calibration_freeze_sha256=sha(r / "calibration26/frozen.json"),
        pack_sha256=sha(r / "packs" / (a.key + ".npz")),
        source_sha256=sha(Path(__file__)),
        rows=len(data["ids"]),
        sites=len(set(data["site"])),
        minimum_calibration_distance=float(distance.min()),
        rgb_pixel_duplicates_with_prior=int(
            sum(p in prior_pixels for p in data["pixel_sha256"])
        ),
        role="Prospective setup transfer at repeated training scenes"
        if a.key == "release26_transfer"
        else "Prospective capture on links held apart from quality training; historical contexts may have been inspected",
        point_and_calibration_selection="Frozen before scoring this cohort. No refit or threshold selection from these scores.",
    )
    save(a.output / "lock.json", lock)
    summary = []
    for record in records:
        path = Path(record["root"]) / record["model_file"]
        assert sha(path) == record["model_sha256"]
        x = feature_matrix(data, s, record["ordered_features"])
        name = record["name"]
        cov = None
        if record["kind"] == "SpatialGP":
            gp = SpatialQualityGP.load(path)
            mean, cov = gp.predict(x)
            cref = next(c for c in freeze["calibrators"] if c["name"] == name)
            cpath = r / "calibration26" / cref["file"]
            assert sha(cpath) == cref["sha256"]
            cal = read(cpath)
            assert cal["model_sha256"] == sha(path)
            cm, cc = apply_calibration(mean, cov, cal)
        else:
            mean = joblib.load(path).predict(x)
            if mean.ndim == 1:
                mean = mean[:, None]
        report = dict(
            name=name,
            kind=record["kind"],
            model_sha256=sha(path),
            raw=metrics(data, mean, cov, allmask, data["site"]),
        )
        arrays = dict(ids=data["ids"], mean=mean)
        if cov is not None:
            report["calibration26"] = metrics(data, cm, cc, allmask, data["site"])
            scores = spatial_scores(cm, cc)
            diagonal = cc * np.eye(3)[None]
            independent_scores = spatial_scores(cm, diagonal)
            supported = data["error_type"] >= 0
            defined = np.isfinite(data["y"])
            labels = data["error_type"]
            report["events"] = {}
            for event, label, score, mask in [
                ("low_iou", data["y"] < 0.7, "global_low", defined),
                (
                    "normal_near_good_far_weak",
                    (data["y"] >= 0.7) & (labels == 2),
                    "normal_far_only",
                    supported,
                ),
            ]:
                report["events"][event] = dict(
                    rows=int(mask.sum()),
                    positives=int(label[mask].sum()),
                    gaussian_auc=auc(label[mask], scores[score][mask]),
                    gaussian_brier=float(
                        brier_score_loss(label[mask], scores[score][mask])
                    )
                    if mask.any()
                    else None,
                    diagonal_brier=float(
                        brier_score_loss(label[mask], independent_scores[score][mask])
                    )
                    if mask.any()
                    else None,
                )
            weak = np.isin(labels, [1, 2])
            report["far_vs_both_weak"] = dict(
                rows=int(weak.sum()),
                sigma_auc=auc(labels[weak] == 2, np.sqrt(cov[weak, 0, 0])),
                near_mean_auc=auc(labels[weak] == 2, mean[weak, 1]),
                contrast_auc=auc(labels[weak] == 2, (mean[:, 1] - mean[:, 2])[weak]),
            )
            arrays.update(covariance=cov, calibrated_mean=cm, calibrated_covariance=cc)
            if name == freeze["selected_primary"]:
                old = read(r / "coverage_evaluation" / (name + "_calibration.json"))
                om, oc = apply_calibration(mean, cov, old)
                report["legacy_calibration"] = metrics(
                    data, om, oc, allmask, data["site"]
                )
        np.savez_compressed(a.output / (name + ".npz"), **arrays)
        summary.append(report)
        save(a.output / "summary.json", summary)
        print(
            name,
            "raw IoU MAE",
            report["raw"]["mae"],
            "calibration26",
            report.get("calibration26", {}).get("mae"),
            flush=True,
        )
    save(
        a.output / "COMPLETE.json",
        dict(
            lock_sha256=sha(a.output / "lock.json"),
            summary_sha256=sha(a.output / "summary.json"),
        ),
    )


if __name__ == "__main__":
    main()
