"""Publish target-specific GP/support and fixed environmental-transfer evidence."""

from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, joblib
from quality_estimation.region_feature_views import feature_matrix
from study import read, save, sha, load


def paired_by_site(data, left, right):
    valid = np.isfinite(data["Y"][:, 2])
    values = []
    for site in np.unique(data["site"]):
        mask = valid & (data["site"] == site)
        values.append(
            float(
                (
                    abs(np.clip(left[mask], 0, 1) - data["Y"][mask, 2])
                    - abs(np.clip(right[mask], 0, 1) - data["Y"][mask, 2])
                ).mean()
            )
        )
    delta = np.asarray(values)
    bootstrap = (
        np.random.default_rng(23411)
        .choice(delta, (4000, len(delta)), replace=True)
        .mean(1)
    )
    return dict(
        sites=len(delta),
        macro_mae_difference=float(delta.mean()),
        paired_site_bootstrap95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        site_differences=delta.tolist(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["study", "bundle", "repo"]:
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    root = r / "far_followup"
    out = a.repo / "docs/baselines/roi9_gp_20261126/followup"
    out.mkdir(parents=True, exist_ok=True)
    data, s = load(r / "packs", ["release26_followup_test"])

    def pred(name):
        with np.load(root / "prospective96" / (name + ".npz")) as z:
            lookup = {sid: i for i, sid in enumerate(z["ids"])}
            return z["mean"][[lookup[sid] for sid in data["ids"]]]

    primary = pred("primary")
    far = pred("far31_GP")
    rf = pred("far31_RF")
    pairs = dict(
        far_gp_minus_primary=paired_by_site(data, far[:, 2], primary[:, 2]),
        far_gp_minus_far_rf=paired_by_site(data, far[:, 2], rf[:, 2]),
    )
    spec = read(a.bundle / "feature_recipe.json")
    task = spec["tasks"]["quality_rf"]
    path = a.bundle / task["artifact_filename"]
    assert sha(path) == task["model_sha256"]
    qrf = joblib.load(path).predict(feature_matrix(data, s, task["ordered_features"]))
    mask = np.isfinite(data["y"])
    quality = dict(
        rows=int(mask.sum()),
        sites=len(set(data["site"])),
        gp_raw_iou_mae=float(
            abs(np.clip(primary[mask, 0], 0, 1) - data["y"][mask]).mean()
        ),
        rf_raw_iou_mae=float(abs(qrf[mask] - data["y"][mask]).mean()),
        rf_model_sha256=sha(path),
        scope="Main frozen28-input GP versus the actual frozen quality RF. Auxiliary-far RF is a different target-specific model and must not be described as predicting whole IoU.",
    )
    save(root / "quality_control.json", quality)
    save(root / "paired_effects.json", pairs)
    for src, dest in [
        ("fit/summary.json", "far_feature_comparison.json"),
        ("fit/frozen.json", "far_frozen_models.json"),
        ("calibration/frozen.json", "calibration_freeze.json"),
        ("prospective96/summary.json", "prospective_far_metrics.json"),
        ("old96_diagnosis/summary.json", "previous96_diagnostics.json"),
        ("quality_control.json", "primary_quality_control.json"),
        ("paired_effects.json", "paired_far_effects.json"),
        ("protocol.json", "protocol.json"),
    ]:
        value = read(root / src)
        if dest in ["prospective_far_metrics.json", "previous96_diagnostics.json"]:
            value = dict(
                models=value,
                scope="All variants retain Q/near means from the primary GP; far31_RF predicts only far recall. Use primary_quality_control for the actual quality RF comparison. The older96-frame cohort was already inspected; only the supplementary96-frame set is prospective for these candidates.",
            )
        save(out / dest, value)
    save(
        out / "environment_transfer.json",
        read(r / "environment_transfer26_cpu/summary.json"),
    )
    save(
        out / "environment_support.json", read(r / "reports/environment26_support.json")
    )
    cal, _ = load(r / "packs", ["release26_calibration", "release26_far_calibration"])
    first = min(
        read(root / "calibration/frozen.json")["created_utc"],
        read(root / "fit/frozen.json")["created_utc"],
    )
    frozen_time = read(root / "calibration/frozen.json")["created_utc"]
    captured = [
        json.loads(line)
        for folder in ["capture_test", "capture_test_part2", "capture_test_part3"]
        for line in (root / folder / "manifest.jsonl").read_text().splitlines()
    ]
    assert all(row["started_utc"] > frozen_time for row in captured)
    save(
        out / "collection_and_lock.json",
        dict(
            supplementary_calibration=96,
            supplementary_test=96,
            new26_total=528,
            registered_total=23334,
            test_sites=len(set(data["site"])),
            calibration_sites=len(set(cal["site"])),
            supplementary_calibration_far_at_least_half=int(
                (cal["Y"][:, 2] >= 0.5).sum()
            ),
            calibration_frozen_utc=frozen_time,
            first_test_capture=min(row["started_utc"] for row in captured),
            point_models_and_calibration_before_capture=True,
            test_plan_sha256=sha(root / "test_plan.json"),
            source_receipt_sha256=sha(root / "prospective96/receipt.json"),
            limitations="Calibration supplement selected from high far-recall strata in old calibration data; not a representative deployment population. New roads are on the same map. RPC service reconnection was needed; accepted source manifests and failed/orphan records were preserved.",
        ),
    )
    # Scientific plots use aggregate measured values, not simulator UI screenshots.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = read(root / "old96_diagnosis/summary.json")
    new = read(root / "prospective96/summary.json")
    names = ["Shared GP (28)", "Far GP (28)", "Far GP (31)", "Far RF (31)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, records, title in [
        (axes[0], old, "Earlier 96 frames (already diagnosed)"),
        (axes[1], new, "Prospective 96 frames / four other links"),
    ]:
        values = [row["raw"]["far_recall"]["mae"] for row in records]
        bars = ax.bar(names, values, color=["#376892", "#859dad", "#559d8c", "#bd7961"])
        ax.bar_label(bars, fmt="%.3f")
        ax.set(title=title, ylabel="Far-band recall MAE", ylim=(0, 0.14))
        ax.tick_params(axis="x", labelrotation=12)
    for suffix in ["png", "svg"]:
        fig.savefig(r / "figures" / ("14_far_followup." + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for j, (metric, label) in enumerate(
        [
            ("coverage90", "Observed 90% far-recall coverage"),
            ("mean_width90", "Mean far-recall interval width"),
        ]
    ):
        x = np.arange(3)
        for shift, key, color in [
            (-0.18, "original96", "#6589a9"),
            (0.18, "supported192", "#579a8e"),
        ]:
            values = [row[key]["far_recall"][metric] for row in new[:3]]
            bars = axes[j].bar(
                x + shift,
                values,
                width=0.36,
                label={
                    "original96": "96 calibration frames",
                    "supported192": "192 with added support",
                }[key],
                color=color,
            )
            axes[j].bar_label(bars, fmt="%.2f", fontsize=9)
        axes[j].set_xticks(x, names[:3])
        axes[j].set(title=label)
        if j == 0:
            axes[j].axhline(0.9, color="gray", linestyle="--")
            axes[j].set_ylim(0, 1.1)
        axes[j].legend(fontsize=9, loc="upper left")
        if j == 1:
            axes[j].set_ylim(0, 0.33)
    for suffix in ["png", "svg"]:
        fig.savefig(
            r / "figures" / ("15_calibration_support_tradeoff." + suffix), dpi=180
        )
    plt.close(fig)
    env = read(r / "environment_transfer26_cpu/summary.json")
    support = read(r / "reports/environment26_support.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    matrix = np.asarray(env["weather"]["confusion"])
    axes[0].imshow(matrix, cmap="Blues")
    labels = ["Sunny", "Cloudy", "Foggy", "Storm", "Rainy", "Snowy"]
    axes[0].set_xticks(range(6), labels, rotation=25)
    axes[0].set_yticks(range(6), labels)
    axes[0].set(
        xlabel="Frozen RGB model prediction",
        ylabel="Capture setting",
        title="26R1 weather transfer: 432 RGB frames",
    )
    for y, x in np.ndindex(matrix.shape):
        axes[0].text(
            x,
            y,
            str(matrix[y, x]),
            ha="center",
            va="center",
            color="white" if matrix[y, x] > 36 else "black",
        )
    normal = [v["benchmark_normal"] for v in support["cluster_support"]]
    axes[1].bar(range(16), normal, color="#5e8e94")
    axes[1].axhline(
        20, color="#b56e5e", linestyle="--", label="Minimum 20 benchmark-normal frames"
    )
    axes[1].set(
        title="No group meets the benchmark-support rule",
        xlabel="Frozen e-means cluster ID",
        ylabel="Calibration frames with GT IoU>=0.7",
        ylim=(0, 24),
    )
    axes[1].set_xticks(range(16))
    axes[1].legend(fontsize=9)
    for suffix in ["png", "svg"]:
        fig.savefig(
            r / "figures" / ("16_environment_transfer_support." + suffix), dpi=180
        )
    plt.close(fig)
    for name in [
        "14_far_followup",
        "15_calibration_support_tradeoff",
        "16_environment_transfer_support",
    ]:
        shutil.copyfile(r / "figures" / (name + ".png"), out / (name + ".png"))
    save(
        out / "receipt.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(Path(__file__)),
            files={
                p.name: sha(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "receipt.json"
            },
        ),
    )
    print("published far/coverage/environment follow-up", quality, pairs, flush=True)


if __name__ == "__main__":
    main()
