"""Create reproducible scientific figures from frozen scores and aligned RGB/GT outputs."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_risk import spatial_scores
from study import load, read, save, sha

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)
BLUE = "#235789"
ORANGE = "#e28b30"
GREEN = "#3a9679"
PURPLE = "#8058a5"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()
    root = a.study
    out = root / "figures"
    out.mkdir(exist_ok=True)
    receipts = []

    def finish(fig, name, caption):
        fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(out / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        receipts.append(
            dict(file=name + ".png", sha256=sha(out / (name + ".png")), caption=caption)
        )

    screen = read(root / "feature_study/summary.json")
    views = [
        "current27",
        "global30",
        "pooled33",
        "quadrants39",
        "allmorph42",
        "distance32",
        "pooled_distance38",
        "common123",
        "expanded263",
    ]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(views))
    for offset, (kind, balance, color, label) in enumerate(
        [
            ("RF", False, ORANGE, "RF"),
            ("GP", False, BLUE, "GP"),
            ("GP", True, GREEN, "GP: group-balanced"),
        ]
    ):
        values = [
            next(
                r["cv"][0]
                for r in screen
                if r["view"] == v and r["kind"] == kind and r["balanced"] == balance
            )
            for v in views
        ]
        ax.bar(x + (offset - 1) * 0.24, values, 0.24, color=color, label=label)
    ax.set(
        xticks=x,
        xticklabels=[
            "27 base",
            "+global\n3 morph.",
            "+pooled\n6 morph.",
            "+quadrant\n12 morph.",
            "+global/road\n15 morph.",
            "+near-far\n5 relations",
            "+pooled/\nrelations",
            "123 common",
            "263 expanded",
        ],
        ylabel="Grouped-CV whole-frame IoU MAE",
        ylim=(0.06, 0.105),
        title="More features did not consistently improve quality estimation",
    )
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    finish(
        fig,
        "02_feature_screen",
        "Original six-zone training/development screen, conditional on the previously selected24 fields. Hyperparameters are selected by grouped CV; this is not a fully nested feature-selection estimate.",
    )
    feature_spec = read(root / "export_v3/feature_recipe.json")
    columns = feature_spec["tasks"]["quality_spatial_gp"]["ordered_features"]
    source = read(root / "receipts/replay_input_sources.json")
    records = [
        json.loads(line)
        for line in (root / "replay50/predictions.jsonl").read_text().splitlines()
    ]
    example_row = next(
        r
        for r in read(root / "extraction_index.json")["z00"]["rows"]
        if r["weather"] == 0 and r["hour"] == 13
    )
    example = cv2.cvtColor(
        cv2.imread(str(a.data_root / example_row["image"])), cv2.COLOR_BGR2RGB
    )
    profile = SpatialProfile.load()
    colors = (
        np.array(
            [
                [63, 143, 205],
                [59, 175, 188],
                [87, 158, 92],
                [156, 191, 75],
                [243, 185, 67],
                [239, 136, 55],
                [222, 82, 93],
                [154, 101, 182],
                [109, 117, 128],
            ],
            dtype=float,
        )
        / 255
    )
    fig, (ax, left) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    ax.imshow(example)
    ax.imshow(
        profile.native,
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=8.5,
        alpha=0.28,
        interpolation="nearest",
    )
    for i, name in enumerate(profile.region_names):
        yy, xx = np.where(profile.native == i)
        ax.text(
            np.median(xx),
            np.median(yy),
            name.replace("_", " "),
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        )
    ax.axhline(169, color="white", lw=1)
    ax.axhline(338, color="white", lw=1)
    ax.axis("off")
    ax.set_title("Fixed nine-region camera partition")
    counts = [
        sum(n.startswith(r + ".") for n in columns)
        for r in profile.region_names + ("global", "near_minus_far")
    ]
    labels = [n.replace("_", " ") for n in profile.region_names] + [
        "whole image",
        "near minus far",
    ]
    left.barh(np.arange(len(labels)), counts, color=[*colors, BLUE, PURPLE])
    left.set_yticks(np.arange(len(labels)), labels)
    left.invert_yaxis()
    left.set(
        xlabel="Direct model input columns",
        xlim=(0, max(counts) + 1),
        title="28 selected inputs",
    )
    left.grid(axis="x", alpha=0.15)
    for i, n in enumerate(counts):
        left.text(n + 0.1, i, str(n), va="center")
    fig.text(
        0.5,
        0.01,
        "Far and near pooled outputs are also computed for three relations. Upper RGB descriptors remain in the separate environment study.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(
        fig,
        "01_roi9_selected_inputs",
        "Nine regions are fixed spatial masks, not semantic sky/road segmentation. Direct zero columns do not remove an ROI: pooled far-band relations and whole-frame entropy still use its pixels.",
    )
    evaluation = read(root / "coverage_evaluation/summary.json")
    selected = next(r for r in evaluation if r["name"] == "coverage_GP_b0_9")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ["base_GP", "base_RF", "coverage_GP_b0_9", "coverage_RF"]
    labels = ["GP: 1,728", "RF: 1,728", "GP: 2,480", "RF: 2,480"]
    for ax, cohort, title in zip(
        axes,
        ["external_motion", "external_link25"],
        [
            "Repeated driving: 7,339 frames / 20 sessions",
            "Buffered archived roads: 99 frames / 10 links",
        ],
    ):
        values = [
            next(r for r in evaluation if r["name"] == n)["cohorts"][cohort]
            for n in names
        ]
        ax.bar(
            np.arange(4),
            [v["mae"] for v in values],
            color=[BLUE, ORANGE, BLUE, ORANGE],
            alpha=0.9,
        )
        for i, v in enumerate(values):
            ax.text(i, v["mae"] + 0.002, f"{v['mae']:.4f}", ha="center")
        ax.set(
            xticks=np.arange(4),
            xticklabels=labels,
            ylabel="Whole-frame IoU MAE",
            title=title,
            ylim=(0, 0.15),
        )
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    finish(
        fig,
        "04_gp_rf_coverage",
        "Fixed28 inputs; quality-model training coverage changes, perception checkpoint stays frozen. Archive scores are diagnostic for this follow-up; paired session/link confidence intervals are stored separately.",
    )
    index = read(root / "extraction_index.json")
    keys = [k for k in index if k.startswith("archive_")]
    data, schema = load(root / "packs", keys)
    with np.load(root / "coverage_evaluation/predictions/coverage_GP_b0_9.npz") as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        ix = [lookup[sid] for sid in data["ids"]]
        mean = z["mean"][ix]
        cov = z["covariance"][ix]
        cm = z["calibrated_mean"][ix]
        cc = z["calibrated_covariance"][ix]
    sigma = np.sqrt(cov[:, 0, 0])
    valid = np.isin(data["error_type"], [1, 2])
    label = data["error_type"][valid] == 2
    probability = spatial_scores(cm, cc)
    conditional = probability["far_only"] / np.maximum(
        probability["far_only"] + probability["both_weak"], 1e-12
    )
    values = [sigma, mean[:, 1], mean[:, 1] - mean[:, 2], conditional]
    names = [
        "IoU sigma",
        "Predicted near recall",
        "Near - far mean",
        "Conditional joint score",
    ]
    aucs = [float(roc_auc_score(label, v[valid])) for v in values]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))
    for code, color, labeltext in [
        (1, ORANGE, "Both image bands weak"),
        (2, BLUE, "Near good, far weak"),
    ]:
        ix = data["error_type"] == code
        ax.hist(
            sigma[ix],
            bins=35,
            density=True,
            histtype="step",
            lw=2,
            color=color,
            label=labeltext,
        )
    ax.set(
        xlabel="Uncalibrated IoU observation sigma",
        ylabel="Density",
        title="Sigma did not identify the spatial error type",
    )
    ax.legend(fontsize=9)
    bx.barh(np.arange(4), aucs, color=[ORANGE, BLUE, GREEN, PURPLE])
    bx.set_yticks(np.arange(4), names)
    bx.invert_yaxis()
    bx.axvline(0.5, color="gray", ls="--")
    bx.set(xlim=(0, 1.02), xlabel="AUROC: near-good/far-weak vs both-weak")
    for i, v in enumerate(aucs):
        bx.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=10)
    fig.tight_layout()
    finish(
        fig,
        "03_sigma_and_spatial_means",
        "Same supported far-weak rows for all AUROCs. This taxonomy is partly defined by near recall: its own predicted mean is an essential control. A joint score must not be advertised as a covariance-specific improvement over that control.",
    )
    save(
        out / "uncertainty_auc.json",
        dict(rows=int(valid.sum()), scores=dict(zip(names, aucs))),
    )
    policy = read(root / "policy_study/summary.json")
    policies = policy["policies"]
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    choices = [
        "global_probability",
        "global_conditional",
        "dual_point",
        "dual_probability",
        "dual_conditional",
        "rf_global",
    ]
    for i, name in enumerate(choices):
        v = policies[name]
        ax.scatter(
            v["global_low_hits"],
            v["normal_far_hits"],
            s=90,
            marker=markers[i],
            label=name.replace("_", " "),
        )
        ax.annotate(
            str(v["unique_25m_cells"]) + " cells",
            (v["global_low_hits"], v["normal_far_hits"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    for seed in (23411, 23412, 23413):
        v = policies["random" + str(seed)]
        ax.scatter(
            v["global_low_hits"],
            v["normal_far_hits"],
            s=60,
            c="gray",
            marker="x",
            label="Random (3 seeds)" if seed == 23411 else None,
        )
    ax.set(
        xlabel="Selected whole-frame IoU <0.7 frames",
        ylabel="Selected normal-IoU / near-good / far-weak frames",
        title="Same 805-frame budget: different review priorities",
        xlim=(300, 680),
        ylim=(100, 420),
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    finish(
        fig,
        "05_review_tradeoff",
        "7,339 frames,64-frame nonoverlapping windows and ceil10% of each actual window. Whole-frame low-IoU and normal-IoU/far-weak groups are disjoint. Far weakness is a diagnostic, not an operational failure label; covariance diversification trades some low-IoU hits for other coverage.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cohorts = ["external_motion", "external_link25"]
    x = np.arange(2)
    for ax, metric, title in [
        (axes[0], "coverage90", "90% IoU interval coverage"),
        (axes[1], "mean_interval_width", "Mean IoU interval width"),
    ]:
        raw = [selected["cohorts"][c][metric][0] for c in cohorts]
        cal = [selected["cohorts"][c]["calibrated"][metric][0] for c in cohorts]
        ax.bar(x - 0.18, raw, 0.36, color=ORANGE, label="Raw")
        ax.bar(x + 0.18, cal, 0.36, color=BLUE, label="Separate calibration")
        ax.set(
            xticks=x, xticklabels=["Repeated driving", "Buffered roads"], title=title
        )
        if metric == "coverage90":
            ax.axhline(0.9, color="gray", ls="--")
            ax.set_ylim(0, 1.04)
        ax.legend()
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    finish(
        fig,
        "06_calibration_coverage_width",
        "Calibration on original576 frames. Improved coverage comes with wider intervals; road coverage remains below nominal90%. No exchangeability or domain-shift guarantee is claimed.",
    )
    training_ids = set(read(root / "coverage/selection.json")["ids"])
    for i in [0, 3, 7, 14, 18, 23]:
        item = source[i]
        sid = item["sample_id"]
        row = next(r for r in index[item["key"]]["rows"] if r["sample_id"] == sid)
        rgb = cv2.cvtColor(
            cv2.imread(str(a.data_root / row["image"])), cv2.COLOR_BGR2RGB
        )
        semantic = cv2.imread(str(a.data_root / row["semantic"]))
        assert (
            sha(a.data_root / row["image"]) == row["expected_sha256"]["rgb"]
            and sha(a.data_root / row["semantic"]) == row["expected_sha256"]["semantic"]
        )
        gt_native = np.all(semantic == (255, 255, 255), 2) | np.all(
            semantic == (0, 255, 255), 2
        )
        gt = cv2.resize(
            gt_native.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        with np.load(root / "replay50" / f"{i:06d}.npz") as z:
            pred = np.isin(z["marking_probability"][:, 12:372].argmax(0), [1, 2])
        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt
        union = int((pred | gt).sum())
        q = float(tp.sum() / union) if union else None
        error = np.zeros((360, 640, 3), np.uint8)
        error[tp] = (50, 190, 100)
        error[fp] = (240, 85, 80)
        error[fn] = (55, 165, 235)
        image = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.1))
        for ax, value, title in zip(
            axes.ravel(),
            [image, gt, pred, error],
            [
                "Original RGB",
                "Aligned white/yellow Semantic GT",
                "Fixed v6 prediction",
                "Green TP / red FP / blue FN",
            ],
        ):
            ax.imshow(value, cmap="gray", vmin=0, vmax=1 if value.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
        role = (
            "quality-training example"
            if sid in training_ids
            else "archived diagnostic example"
        )
        fig.suptitle(
            f"Case {i + 1}: true IoU {q:.3f}, GP estimate {records[i]['predicted_iou']:.3f}\n({role})",
            fontsize=13,
        )
        fig.tight_layout(h_pad=2.0, w_pad=1.0, rect=(0, 0, 1, 0.94))
        finish(
            fig,
            f"07_case_{i + 1:02d}",
            "Illustrative archived case, not an independent performance estimate. Actual fixed-v6 softmax and matching source Semantic. Original1280x720 GT is resized by nearest neighbour to640x360; model padding12+12 is removed. Figure IoU is recomputed from this exact replay output, not copied from another batch.",
        )
        receipts[-1].update(
            source_rgb_sha256=sha(a.data_root / row["image"]),
            source_gt_sha256=sha(a.data_root / row["semantic"]),
            probability_sha256=sha(root / "replay50" / f"{i:06d}.npz"),
            true_iou=q,
        )
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    blocks = [
        (0.1, "RGB + fixed v6", "No online GT"),
        (2.7, "Nine-region features", "28 ordered inputs"),
        (5.3, "GP mean / covariance", "IoU + recall auxiliaries"),
        (7.9, "Review candidates", "Quality / spatial diagnostic"),
        (10.5, "Offline validation", "GT + equal-budget retraining"),
    ]
    for i, (xx, title, subtitle) in enumerate(blocks):
        ax.add_patch(
            FancyBboxPatch(
                (xx, 1.2),
                2.25,
                1.2,
                boxstyle="round,pad=.08",
                facecolor="#eef4f8",
                edgecolor=BLUE,
            )
        )
        ax.text(xx + 1.125, 1.98, title, ha="center", va="center", fontsize=11)
        ax.text(
            xx + 1.125,
            1.55,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
            color="#43515b",
        )
        if i < 4:
            ax.annotate(
                "",
                (xx + 2.55, 1.8),
                (xx + 2.3, 1.8),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
            )
    ax.text(
        6.5,
        0.5,
        "Model / feature / calibration versions are locked together. A new perception checkpoint needs quality-model revalidation.",
        ha="center",
        fontsize=11,
    )
    finish(
        fig,
        "08_offline_pipeline",
        "Runtime uses RGB and perception output only. Spatial auxiliary predictions are optional diagnostics. Dataset admission and perception update remain offline decisions after GT/retention checks.",
    )
    save(
        out / "figure_receipt.json",
        dict(figures=receipts, source_sha256=sha(Path(__file__))),
    )
    print("saved", len(receipts), "figures", flush=True)


if __name__ == "__main__":
    main()
