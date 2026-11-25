"""Audit paired uncertainty/quality claims and a privileged condition-only control."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from study import load, read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    p = read(r / "coverage_protocol.json")
    keys = p["training_keys"] + p["augmentation_pool"] + p["motion_training_keys"]
    train, schema = load(r / "packs", keys)
    selected = read(r / "coverage/selection.json")["ids"]
    lookup = {sid: i for i, sid in enumerate(train["ids"])}
    ix = np.array([lookup[sid] for sid in selected])
    ix = ix[np.isfinite(train["Y"][ix]).all(1)]
    frozen = read(r / "coverage/frozen.json")
    record = next(v for v in frozen["models"] if v["name"] == frozen["selected"])
    params = dict(record["parameters"], task_correlation=0.0)
    x = feature_matrix(train, schema, record["ordered_features"])[ix]
    independent = SpatialQualityGP(**params).fit(x, train["Y"][ix])
    path = r / "models/coverage_independent.npz"
    independent.save(path)
    save(
        r / "models/coverage_independent.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            model_sha256=sha(path),
            selected_parent_sha256=record["model_sha256"],
            parameters=params,
            ordered_features=record["ordered_features"],
            role="Same2480 inputs/targets and selected isotropic hyperparameters, rho0 control. Freeze before any26R1 score is inspected.",
        ),
    )
    medians = {}
    for weather, hour in set(zip(train["weather"][ix], train["hour"][ix])):
        mask = (train["weather"][ix] == weather) & (train["hour"][ix] == hour)
        medians[int(weather), int(hour)] = float(np.median(train["y"][ix][mask]))
    data, _ = load(
        r / "packs",
        [k for k in read(r / "extraction_index.json") if k.startswith("archive_")],
    )
    oracle = np.array(
        [
            medians.get((int(w), int(h)), float(np.median(train["y"][ix])))
            for w, h in zip(data["weather"], data["hour"])
        ]
    )
    prediction = r / "coverage_evaluation/predictions/coverage_GP_b0_9.npz"
    with np.load(prediction) as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        di = [lookup[sid] for sid in data["ids"]]
        gp = z["mean"][di, 0]
    with np.load(r / "coverage_evaluation/predictions/coverage_RF.npz") as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        rf = np.array([z["mean"][lookup[sid], 0] for sid in data["ids"]])
    conditions = []
    for w, h in sorted(set(zip(data["weather"], data["hour"]))):
        mask = (data["weather"] == w) & (data["hour"] == h)
        label = data["y"][mask] < 0.7
        if min(int(label.sum()), int((~label).sum())) < 5:
            continue
        conditions.append(
            dict(
                weather=int(w),
                hour=int(h),
                rows=int(mask.sum()),
                low=int(label.sum()),
                gp_auc=float(roc_auc_score(label, -gp[mask])),
                rf_auc=float(roc_auc_score(label, -rf[mask])),
                condition_oracle_auc=0.5,
            )
        )
    sessions = []
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        sessions.append(
            dict(
                group=str(group),
                gp_mae=float(abs(np.clip(gp[mask], 0, 1) - data["y"][mask]).mean()),
                rf_mae=float(abs(rf[mask] - data["y"][mask]).mean()),
                condition_oracle_mae=float(abs(oracle[mask] - data["y"][mask]).mean()),
            )
        )
    delta = np.array([s["gp_mae"] - s["rf_mae"] for s in sessions])
    rng = np.random.default_rng(23411)
    interval = np.quantile(
        rng.choice(delta, (5000, len(delta)), replace=True).mean(1), [0.025, 0.975]
    ).tolist()
    save(
        r / "reports/condition_control.json",
        dict(
            rows=len(gp),
            condition_oracle_mae=float(abs(oracle - data["y"]).mean()),
            gp_mae=float(abs(np.clip(gp, 0, 1) - data["y"]).mean()),
            rf_mae=float(abs(rf - data["y"]).mean()),
            gp_minus_rf_session_macro_delta=float(delta.mean()),
            paired_session_bootstrap95=interval,
            within_condition=conditions,
            sessions=sessions,
            role="Weather/time labels are privileged offline control inputs only. The deployed GP consumes RGB/perception features. Already-open repeated-route archival diagnoses, not new-map evidence.",
        ),
    )
    print(
        "paired session GP-RF delta",
        float(delta.mean()),
        interval,
        "condition-only MAE",
        float(abs(oracle - data["y"]).mean()),
        flush=True,
    )


if __name__ == "__main__":
    main()
