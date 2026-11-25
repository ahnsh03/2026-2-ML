"""Score all frozen real-label GP updates on newly captured roads; do not select a winner."""

from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from quality_estimation.conditioned_spatial_gp import (
    ConditionedSpatialGP,
    condition_moments,
)
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from study import load, read, save, sha


def metrics(mean, cov, truth, sites, noise):
    result = {}
    for j, name in enumerate(["Q", "near_recall", "far_recall"]):
        valid = np.isfinite(truth[:, j])
        error = abs(np.clip(mean[:, j], 0, 1) - truth[:, j])
        sd = np.sqrt(np.maximum(cov[:, j, j], 0))
        by_site = {
            str(site): float(error[valid & (sites == site)].mean())
            for site in np.unique(sites)
            if (valid & (sites == site)).any()
        }
        result[name] = dict(
            rows=int(valid.sum()),
            mae=float(error[valid].mean()),
            by_site_mae=by_site,
            raw_gaussian_90_coverage=float(
                (
                    abs(mean[valid, j] - truth[valid, j])
                    <= 1.6448536269514722 * sd[valid]
                ).mean()
            ),
            mean_observation_variance=float(cov[valid, j, j].mean()),
            mean_latent_variance=float(
                np.maximum(cov[valid, j, j] - noise[j], 0).mean()
            ),
        )
    return result


def paired(left, right):
    sites = sorted(set(left) & set(right))
    delta = np.array([left[s] - right[s] for s in sites])
    rng = np.random.default_rng(23411)
    boot = rng.choice(delta, (4000, len(delta)), replace=True).mean(1)
    return dict(
        sites=len(sites),
        site_mae_differences=delta.tolist(),
        mean_difference=float(delta.mean()),
        site_bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "bundle", "models", "index", "data-root", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    frozen = read(args.models / "frozen.json")
    if (args.models / "_SUCCESS").read_text().strip() != sha(
        args.models / "frozen.json"
    ):
        raise ValueError("incomplete GP update freeze")
    data, schema = load(args.study / "packs", [args.key])
    recipe = read(args.bundle / "feature_recipe.json")
    task = recipe["tasks"]["quality_spatial_gp"]
    parent = args.bundle / task["artifact_filename"]
    parent_digest = sha(parent)
    if (
        parent_digest != frozen["parent_model_sha256"]
        or read(args.study / "packs" / (args.key + ".json"))["checkpoint_sha256"]
        != recipe["checkpoint_sha256"]
    ):
        raise ValueError("test/parent perception version differs")
    pool, _ = load(args.study / "packs", ["release26_heldroads"])
    if set(pool["ids"]) & set(data["ids"]) or set(pool["pixel_sha256"]) & set(
        data["pixel_sha256"]
    ):
        raise ValueError("label pool and test overlap")
    index = read(args.index)[args.key]
    captures = [
        readline
        for relative in index["source_manifests"]
        for readline in [
            json.loads(line)
            for line in (args.data_root / relative / "manifest.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
    ]
    first = min(datetime.fromisoformat(row["started_utc"]) for row in captures)
    if not datetime.fromisoformat(frozen["created_utc"]) < first:
        raise ValueError("GP updates were not frozen before test capture")
    model = SpatialQualityGP.load(parent)
    columns = task["ordered_features"]
    X = feature_matrix(data, schema, columns)
    px = feature_matrix(pool, schema, columns)
    mean, cov = model.predict(X)
    cross = model.posterior_cross_covariance(X, px)
    args.output.mkdir(exist_ok=False)
    summary = []
    cache = {}
    noise = model.noise_variance * model.y_scale_**2
    baseline = metrics(mean, cov, data["Y"], data["site"], noise)
    save(args.output / "parent_metrics.json", baseline)
    import joblib

    rf_task = recipe["tasks"]["quality_rf"]
    rf_path = args.bundle / rf_task["artifact_filename"]
    if (
        sha(rf_path) != rf_task["model_sha256"]
        or rf_task["ordered_features"] != columns
    ):
        raise ValueError("frozen RF control differs")
    rf = joblib.load(rf_path)
    rf.set_params(n_jobs=1)
    rf_mean = rf.predict(X)
    valid = np.isfinite(data["Y"][:, 0])
    save(
        args.output / "parent_quality_control.json",
        dict(
            rows=int(valid.sum()),
            gp_mae=baseline["Q"]["mae"],
            rf_mae=float(
                abs(np.clip(rf_mean[valid], 0, 1) - data["Y"][valid, 0]).mean()
            ),
            rf_model_sha256=sha(rf_path),
            scope="Unchanged equal-training2480-row/28-input GP and quality RF on this additional cohort. RF-driven label acquisition is a separate arm, not an updated RF.",
        ),
    )
    verification = None
    for record in frozen["models"]:
        path = args.models / record["filename"]
        if sha(path) != record["sha256"]:
            raise ValueError("updated model changed")
        indices = np.asarray(record["indices"], int)
        if pool["ids"][indices].tolist() != record["ids"]:
            raise ValueError("frozen observation IDs differ")
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            if metadata["parent_model_sha256"] != parent_digest or not np.array_equal(
                z["X"], px[indices]
            ):
                raise ValueError("update source differs")
            c = cross[:, :, indices, :].reshape(len(X) * 3, -1)[:, z["observed"]]
            updated, ucov = condition_moments(
                mean, cov, c, z["cholesky"], z["innovation"]
            )
        if record["method"] == "dual_conditional" and record["budget"] == 16:
            check = ConditionedSpatialGP.load(path, parent)
            a, b = check.predict(X)
            verification = dict(
                max_mean_difference=float(abs(a - updated).max()),
                max_cov_difference=float(abs(b - ucov).max()),
            )
            if max(verification.values()) > 1e-10:
                raise AssertionError("cached and standalone update predictions differ")
        result = metrics(updated, ucov, data["Y"], data["site"], noise)
        item = dict(
            method=record["method"],
            budget=record["budget"],
            metrics=result,
            q_variance_ratio=result["Q"]["mean_observation_variance"]
            / baseline["Q"]["mean_observation_variance"],
            q_latent_variance_ratio=result["Q"]["mean_latent_variance"]
            / baseline["Q"]["mean_latent_variance"],
            pool_unique_sites=len(set(pool["site"][indices])),
            acquired_whole_iou_mean=float(np.nanmean(pool["Y"][indices, 0])),
            model_sha256=record["sha256"],
        )
        summary.append(item)
        cache[(record["method"], record["budget"])] = item
        np.savez_compressed(
            args.output / record["filename"], ids=data["ids"], mean=updated, cov=ucov
        )
    contrasts = {}
    for budget in sorted(set(row["budget"] for row in summary)):
        a = cache[("dual_conditional", budget)]["metrics"]["Q"]["by_site_mae"]
        b = cache[("dual_probability", budget)]["metrics"]["Q"]["by_site_mae"]
        random = [
            cache[(f"random_{seed}", budget)]["metrics"]["Q"]["by_site_mae"]
            for seed in [23411, 23412, 23413]
        ]
        average = {site: float(np.mean([row[site] for row in random])) for site in a}
        contrasts[str(budget)] = dict(
            conditional_minus_probability=paired(a, b),
            conditional_minus_random_seed_mean=paired(a, average),
            conditional_minus_parent=paired(a, baseline["Q"]["by_site_mae"]),
        )
    result = dict(
        scope="Prospective same-map four-road quality-GP label-acquisition comparison. All arms/budgets frozen before fresh capture; no perception retraining, new calibration, operational failure definition or update adoption.",
        test_rows=len(X),
        test_sites=len(set(data["site"])),
        first_test_capture=first.isoformat(),
        models_frozen_utc=frozen["created_utc"],
        parent_model_sha256=parent_digest,
        test_pack_sha256=sha(args.study / "packs" / (args.key + ".npz")),
        contrasts=contrasts,
        models=summary,
        cached_inference_verification=verification,
        source_sha256=sha(Path(__file__)),
    )
    save(args.output / "summary.json", result)
    print(
        json.dumps(
            dict(
                parent_Q_mae=baseline["Q"]["mae"],
                models=[
                    dict(
                        method=v["method"],
                        budget=v["budget"],
                        Q_mae=v["metrics"]["Q"]["mae"],
                        variance_ratio=v["q_variance_ratio"],
                    )
                    for v in summary
                ],
                contrasts=contrasts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
