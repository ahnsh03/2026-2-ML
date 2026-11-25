"""Post-test target-support audit; does not refit or select models on opened roads."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from study import read, save, sha, load


def distribution(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return dict(
        rows=len(a),
        mean=float(a.mean()),
        quantiles=np.quantile(a, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        at_least_half=int((a >= 0.5).sum()),
        at_least_point8=int((a >= 0.8).sum()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    training = np.asarray(
        [v["targets"] for v in read(a.bundle / "training_membership.json")["rows"]]
    )
    protocol = read(r / "coverage_protocol.json")
    cohorts = dict(training=training)
    packs = {}
    for name, keys in [
        ("old_calibration", protocol["calibration_keys"]),
        ("new26_calibration", ["release26_calibration"]),
        ("new26_test", ["release26_heldroads"]),
    ]:
        data, _ = load(r / "packs", keys)
        packs[name] = data
        cohorts[name] = data["Y"]
    targets = {
        name: {
            task: distribution(y[:, i])
            for i, task in enumerate(["IoU", "near_recall", "far_recall"])
        }
        for name, y in cohorts.items()
    }
    data = packs["new26_test"]
    path = r / "test26_heldroads/coverage_GP_b0_9.npz"
    with np.load(path) as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        ix = [lookup[sid] for sid in data["ids"]]
        mean, covariance = z["mean"][ix], z["covariance"][ix]
    cal = read(a.bundle / "calibration_26r1.json")
    calibrated_mean = mean + np.asarray(cal["bias"])
    sigma = np.sqrt(covariance[:, 2, 2]) * cal["scale"][2]
    y = data["Y"][:, 2]
    bins = []
    for left, right in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.00001)]:
        mask = (y >= left) & (y < right)
        bins.append(
            dict(
                true_far_recall_bin=[left, min(right, 1)],
                rows=int(mask.sum()),
                mean_truth=float(y[mask].mean()) if mask.any() else None,
                mean_prediction=float(calibrated_mean[mask, 2].mean())
                if mask.any()
                else None,
                mean_sigma=float(sigma[mask].mean()) if mask.any() else None,
                coverage90=float(
                    (
                        abs(y[mask] - calibrated_mean[mask, 2])
                        <= 1.6448536269514722 * sigma[mask]
                    ).mean()
                )
                if mask.any()
                else None,
            )
        )
    result = dict(
        scope="Post-test explanation audit on already-open96 frames, not a new independent estimate or a causal explanation. No model, calibration or policy was changed.",
        targets=targets,
        new26_far_recall_bins=bins,
        far_gt_pixels_to_abs_error_spearman=float(
            spearmanr(data["far_gt"], abs(y - calibrated_mean[:, 2])).statistic
        ),
        prediction_sha256=sha(path),
        recipe_sha256=sha(a.bundle / "feature_recipe.json"),
        source_sha256=sha(Path(__file__)),
    )
    save(r / "reports/spatial_target_support.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name, values in cohorts.items():
        far = values[:, 2]
        axes[0].hist(
            far,
            bins=np.linspace(0, 1, 21),
            weights=np.full(len(far), 1 / len(far)),
            histtype="step",
            linewidth=2,
            label=f"{name.replace('_', ' ')} (n={len(far)})",
        )
    axes[0].axvline(0.5, linestyle="--", color="gray")
    axes[0].set(
        xlabel="GT far-band recall",
        ylabel="Fraction of each cohort",
        title="No new-calibration frame had far recall >=0.5",
    )
    axes[0].legend(fontsize=8)
    defined = [b for b in bins if b["rows"]]
    positions = np.arange(len(defined))
    axes[1].bar(
        positions - 0.18,
        [b["mean_truth"] for b in defined],
        width=0.36,
        label="GT mean",
    )
    axes[1].bar(
        positions + 0.18,
        [b["mean_prediction"] for b in defined],
        width=0.36,
        yerr=[1.64485 * b["mean_sigma"] for b in defined],
        label="GP mean +/- mean 90% halfwidth",
    )
    axes[1].set_xticks(
        positions,
        [
            f"{b['true_far_recall_bin'][0]}-{b['true_far_recall_bin'][1]}\nn={b['rows']}"
            for b in defined
        ],
    )
    axes[1].set(
        ylabel="Far recall",
        xlabel="Held-road GT bins (post-test diagnostic)",
        title="Higher-recall test frames were underpredicted",
        ylim=(0, 1),
    )
    axes[1].legend(fontsize=8)
    for suffix in ["png", "svg"]:
        fig.savefig(r / "figures" / ("13_far_target_support." + suffix), dpi=180)
    plt.close(fig)
    print(
        "target-support audit",
        {k: v["far_recall"]["at_least_point8"] for k, v in targets.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
