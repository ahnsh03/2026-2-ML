"""Score frozen quality candidates; preserve calibration and external data roles."""

from __future__ import annotations
import argparse, json, math, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import (
    calibrate_moments,
    apply_calibration,
    spatial_scores,
    conditional_batch,
)
from study import load, read, save, sha


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def finite_mean(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def metrics(data, mean, cov, mask, groups):
    defined = mask & np.isfinite(data["y"])
    error = np.clip(mean[:, 0], 0, 1) - data["y"]
    supported = mask & (data["error_type"] >= 0)
    labels = data["error_type"]
    weak = supported & np.isin(labels, [1, 2])
    group_values = [
        float(abs(error[defined & (groups == g)]).mean())
        for g in np.unique(groups[defined])
    ]
    bootstrap = (
        np.random.default_rng(23411)
        .choice(group_values, (2000, len(group_values)), replace=True)
        .mean(1)
        if group_values
        else np.array([np.nan])
    )
    result = dict(
        rows=int(mask.sum()),
        defined=int(defined.sum()),
        supported=int(supported.sum()),
        mae=finite_mean(abs(error[defined])),
        bias=finite_mean(error[defined]),
        group_macro_mae=finite_mean(group_values),
        group_count=len(group_values),
        group_bootstrap_95=np.quantile(bootstrap, [0.025, 0.975]).tolist()
        if len(group_values) >= 2
        else None,
        low_iou=int((defined & (data["y"] < 0.7)).sum()),
        normal_far_only=int((supported & (data["y"] >= 0.7) & (labels == 2)).sum()),
        spatial_counts={
            str(k): int((mask & (labels == k)).sum()) for k in [-1, 0, 1, 2, 3]
        },
        low_iou_auc=auc(data["y"][defined] < 0.7, -mean[defined, 0]),
    )
    if mean.shape[1] == 3:
        result.update(
            recall_mae=[
                finite_mean(abs(np.clip(mean[mask, j], 0, 1) - data["Y"][mask, j]))
                for j in (1, 2)
            ],
            contrast_far_vs_both_auc=auc(
                labels[weak] == 2, (mean[:, 1] - mean[:, 2])[weak]
            ),
        )
    if cov is not None:
        sigma = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
        coverage = abs(data["Y"] - mean) <= 1.6448536269514722 * sigma
        result.update(
            coverage90=[
                finite_mean(
                    np.where(np.isfinite(data["Y"][mask, j]), coverage[mask, j], np.nan)
                )
                for j in range(3)
            ],
            mean_interval_width=(2 * 1.6448536269514722 * sigma[mask].mean(0)).tolist(),
            sigma_far_vs_both_auc=auc(labels[weak] == 2, sigma[weak, 0]),
            sigma_global_low_auc=auc(data["y"][defined] < 0.7, sigma[defined, 0]),
        )
    return result


def window_review(data, mask, scores, groups):
    chosen = []
    budget_total = 0
    for group in np.unique(groups[mask]):
        indices = np.flatnonzero(mask & (groups == group))
        for start in range(0, len(indices), 64):
            window = indices[start : start + 64]
            budget = int(math.ceil(0.1 * len(window)))
            budget_total += budget
            order = np.lexsort((data["ids"][window], -scores[window]))
            chosen.extend(window[order[:budget]])
    ix = np.asarray(chosen, int)
    defined = np.isfinite(data["y"])
    low = defined & (data["y"] < 0.7)
    far = (data["error_type"] == 2) & (data["y"] >= 0.7)
    union = low | far

    def count(target):
        total = int((target & mask).sum())
        hits = int(target[ix].sum())
        return dict(hits=hits, total=total, recall=hits / total if total else None)

    return dict(
        budget=budget_total,
        selected=len(ix),
        low_iou=count(low),
        normal_far_only=count(far),
        union=count(union),
        precision=float(union[ix].mean()) if len(ix) else None,
        supported_selected=int((data["error_type"][ix] >= 0).sum()),
        unique_25m_cells=len(
            set(map(tuple, np.floor(data["xy"][ix] / 25).astype(int)))
        ),
        selected_ids_sha256=__import__("hashlib")
        .sha256("\n".join(data["ids"][ix]).encode())
        .hexdigest(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("packs", "index", "finalists", "geometry", "protocol", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--controls", type=Path)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=True)
    pred_dir = a.output / "predictions"
    pred_dir.mkdir(exist_ok=True)
    lock = read(a.finalists / "frozen.json")
    p = read(a.protocol)
    keys = list(read(a.index))
    data, schema = load(a.packs, keys)
    ids = data["ids"]
    source_keys = []
    for key in keys:
        with np.load(a.packs / (key + ".npz")) as z:
            source_keys.extend([key] * len(z["ids"]))
    groups = np.asarray(source_keys)
    geometry = {r["id"]: r for r in read(a.geometry)["rows"]}
    assert len(set(ids)) == len(ids) and len(set(data["pixel_sha256"])) == len(ids)
    masks = {
        name: np.isin(groups, ks)
        for name, ks in {
            "development": p["development_keys"],
            "calibration": p["calibration_keys"],
            "history_motion": [k for k in keys if k.startswith("nov23_")],
            "history_geo": [k for k in keys if k.startswith("geo_block_")],
            "prior_eight_sites": ["fresh", "fresh_clock_supplement"],
            "prior_three_sites": ["fresh_uncertainty"],
            "external_static": ["extra_cal", "newroads", "shapes"],
            "external_motion": [k for k in keys if k.startswith("archive_")],
        }.items()
    }
    link = np.array([geometry.get(str(i), {}).get("link", "") for i in ids])
    mask25 = np.array(
        [
            geometry.get(str(i), {}).get("link_distance") is not None
            and geometry[str(i)]["link_distance"] >= 25
            and geometry[str(i)]["point_distance"] >= 25
            for i in ids
        ]
    )
    mask50 = mask25 & np.array(
        [geometry.get(str(i), {}).get("point_distance", 0) >= 50 for i in ids]
    )
    masks.update(external_link25=mask25, external_link50=mask50)
    assert all(mask.any() for mask in masks.values()), "a declared cohort has no rows"
    save(
        a.output / "roles.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            frozen_sha256=sha(a.finalists / "frozen.json"),
            geometry_sha256=sha(a.geometry),
            index_sha256=sha(a.index),
            source_sha256=sha(Path(__file__)),
            model_source_sha256=sha(
                Path(__file__).parents[3] / "src/quality_estimation/spatial_gp.py"
            ),
            counts={n: int(m.sum()) for n, m in masks.items()},
            primary_model=lock["selected"],
            scope="External scores opened after feature/model freeze; augmentation pools are historical diagnostics, and become training for augmented arms. No selection on external metrics.",
        ),
    )
    models = [dict(m, root=str(a.finalists)) for m in lock["models"]]
    if a.controls:
        models += read(a.controls)
    result = []
    policies = {}
    calmask = masks["calibration"]
    for record in models:
        started = time.monotonic()
        name = record["name"]
        path = Path(record["root"]) / record["model_file"]
        assert sha(path) == record["model_sha256"]
        x = feature_matrix(data, schema, record["ordered_features"])
        cov = None
        cal = None
        if record["kind"] == "SpatialGP":
            model = SpatialQualityGP.load(path)
            mean, cov = model.predict(x)
            cal = calibrate_moments(mean[calmask], cov[calmask], data["Y"][calmask])
            calibrated, scaled = apply_calibration(mean, cov, cal)
            save(
                a.output / (name + "_calibration.json"),
                dict(
                    cal,
                    model_sha256=sha(path),
                    calibration_ids_sha256=__import__("hashlib")
                    .sha256("\n".join(ids[calmask]).encode())
                    .hexdigest(),
                    calibration_keys=p["calibration_keys"],
                ),
            )
        else:
            model = joblib.load(path)
            mean = model.predict(x)
            if mean.ndim == 1:
                mean = mean[:, None]
        raw = mean.copy()
        mean = np.clip(mean, 0, 1)
        arrays = dict(ids=ids, mean=raw)
        if cov is not None:
            arrays.update(
                covariance=cov, calibrated_mean=calibrated, calibrated_covariance=scaled
            )
        np.savez_compressed(pred_dir / (name + ".npz"), **arrays)
        row = dict(
            name=name,
            kind=record["kind"],
            model_sha256=sha(path),
            prediction_sha256=sha(pred_dir / (name + ".npz")),
            dimensions=len(record["ordered_features"]),
            training_rows=record.get("rows"),
            cohorts={},
        )
        for cohort, mask in masks.items():
            grouping = (
                link
                if cohort.startswith("external_link")
                else (groups if "motion" in cohort else data["site"])
            )
            row["cohorts"][cohort] = metrics(data, raw, cov, mask, grouping)
            if cov is not None:
                row["cohorts"][cohort]["calibrated"] = metrics(
                    data, calibrated, scaled, mask, grouping
                )
        result.append(row)
        save(a.output / "summary.json", result)
        scores = {"predicted_low": -mean[:, 0]}
        if mean.shape[1] == 3:
            scores["near_far_contrast"] = mean[:, 1] - mean[:, 2]
        if cov is not None:
            scores.update(spatial_scores(calibrated, scaled))
            normal = calmask & (data["y"] >= 0.7)
            cutoff = float(
                np.sort(scores["global_low"][normal])[
                    max(0, int(math.ceil(0.9 * normal.sum())) - 1)
                ]
            )
            cal["alarm_cutoff_at_calibration_normal_10pct"] = cutoff
            alarms = {}
            for cohort, mask in masks.items():
                valid = mask & np.isfinite(data["y"])
                alarm = scores["global_low"] > cutoff
                truth = data["y"] < 0.7
                alarms[cohort] = dict(
                    cutoff=cutoff,
                    low_iou=int((valid & truth).sum()),
                    normal=int((valid & ~truth).sum()),
                    true_positive=int((valid & truth & alarm).sum()),
                    false_positive=int((valid & ~truth & alarm).sum()),
                )
            save(a.output / (name + "_alarms.json"), alarms)
        policies[name] = {}
        for cohort in ("external_motion", "history_motion", "external_link25"):
            mask = masks[cohort]
            grouping = link if cohort.startswith("external_link") else groups
            policies[name][cohort] = {
                n: window_review(data, mask, score, grouping)
                for n, score in scores.items()
            }
        if cov is not None:
            valid = masks["external_motion"] & (data["error_type"] >= 0)
            event = (data["error_type"] == 2) & (data["y"] >= 0.7)
            row["probability_diagnostics"] = {
                event_name: dict(
                    auc=auc(label[valid], scores[score_name][valid]),
                    brier=float(
                        brier_score_loss(label[valid], scores[score_name][valid])
                    ),
                    prevalence=float(label[valid].mean()),
                )
                for event_name, label, score_name in [
                    ("normal_far_only", event, "normal_far_only"),
                    ("global_low", data["y"] < 0.7, "global_low"),
                    ("far_only", data["error_type"] == 2, "far_only"),
                ]
            }
            save(a.output / "summary.json", result)
        save(a.output / "window_review.json", policies)
        print(
            name,
            "external motion",
            row["cohorts"]["external_motion"]["mae"],
            "link25",
            row["cohorts"]["external_link25"]["mae"],
            "seconds",
            round(time.monotonic() - started, 1),
            flush=True,
        )
    save(
        a.output / "data_summary.json",
        dict(
            ids_sha256=__import__("hashlib")
            .sha256("\n".join(ids).encode())
            .hexdigest(),
            rows=len(ids),
            undefined_y=int((~np.isfinite(data["y"])).sum()),
            selected_model=lock["selected"],
            calibration_counts=int(calmask.sum()),
        ),
    )


if __name__ == "__main__":
    main()
