"""Freeze same-budget acquisition lists from an input-only pack, before reading GT."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import (
    apply_calibration,
    spatial_scores,
    conditional_batch,
)
from quality_estimation.review_spatial import select_window
from study import sha, save, read


class CachedCovariances:
    def __init__(self, model, X):
        self.noise_variance = model.noise_variance
        self.y_scale_ = model.y_scale_
        self.q = model.posterior_covariance(X)
        self.d = model.posterior_covariance(X, [0, 1, -1])

    def posterior_covariance(self, X, contrast=None):
        return self.q if contrast is None else self.d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "bundle", "protocol", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    recipe = read(args.bundle / "feature_recipe.json")
    task = recipe["tasks"]["quality_spatial_gp"]
    path = args.bundle / task["artifact_filename"]
    if sha(path) != task["model_sha256"]:
        raise ValueError("parent model changed")
    with np.load(args.inputs, allow_pickle=False) as z:
        if set(z.files) != {"X", "ids", "rf_point_mean", "ordered_features"}:
            raise ValueError("input-only acquisition pack required")
        X = z["X"]
        ids = z["ids"]
        rf = z["rf_point_mean"]
        if z["ordered_features"].tolist() != task["ordered_features"]:
            raise ValueError("input order differs")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pool IDs")
    model = SpatialQualityGP.load(path)
    mean, cov = model.predict(X)
    calibration_path = args.bundle / task["calibration_filename"]
    if sha(calibration_path) != task["calibration_sha256"]:
        raise ValueError("calibration changed")
    calibrated, ccov = apply_calibration(mean, cov, read(calibration_path))
    risk = spatial_scores(calibrated, ccov)
    predictions = [
        dict(
            predicted_iou=float(np.clip(mean[i, 0], 0, 1)),
            risk_scores={k: float(v[i]) for k, v in risk.items()},
        )
        for i in range(len(ids))
    ]
    cached = CachedCovariances(model, X)
    selections = []
    for budget in protocol["budgets"]:
        for method in protocol["methods"]:
            if method == "rf_mean":
                chosen = np.lexsort((ids, rf))[:budget].tolist()
            elif method == "gp_information_only":
                chosen = conditional_batch(
                    np.ones(len(ids)),
                    cached.q,
                    budget,
                    ids,
                    noise_variance=model.noise_variance * model.y_scale_[0] ** 2,
                )
            elif method.startswith("random_"):
                chosen = (
                    np.random.default_rng(int(method.split("_")[-1]))
                    .permutation(len(ids))[:budget]
                    .tolist()
                )
            else:
                chosen = [
                    i
                    for i, _ in select_window(
                        predictions, ids, budget, method, X=X, model=cached
                    )
                ]
            chosen = [int(i) for i in chosen]
            if len(chosen) != budget or len(set(chosen)) != budget:
                raise ValueError("selection budget differs")
            selections.append(
                dict(
                    method=method,
                    budget=budget,
                    indices=chosen,
                    ids=ids[chosen].tolist(),
                )
            )
    args.output.mkdir(exist_ok=False)
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        uses_ground_truth=False,
        pool_rows=len(ids),
        inputs_sha256=sha(args.inputs),
        protocol_sha256=sha(args.protocol),
        parent_model_sha256=sha(path),
        recipe_sha256=sha(args.bundle / "feature_recipe.json"),
        selections=selections,
        source_sha256=sha(Path(__file__)),
    )
    save(args.output / "selection.json", result)
    (args.output / "_SUCCESS").write_text(sha(args.output / "selection.json") + "\n")
    print("Frozen", len(selections), "input-only acquisition lists")


if __name__ == "__main__":
    main()
