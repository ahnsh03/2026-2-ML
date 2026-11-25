"""Plot prospective setup tests and aligned held-road examples, including remaining failures."""

from __future__ import annotations
import argparse, shutil
from pathlib import Path
import cv2, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_gp import SpatialQualityGP
from study import read, save, sha, load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    out = r / "figures"
    receipts = []

    def finish(fig, name, caption):
        fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(out / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        receipts.append(
            dict(file=name + ".png", sha256=sha(out / (name + ".png")), caption=caption)
        )

    archive = {v["name"]: v for v in read(r / "coverage_evaluation/summary.json")}
    repeated = {v["name"]: v for v in read(r / "test26_transfer/summary.json")}
    held = {v["name"]: v for v in read(r / "test26_heldroads/summary.json")}
    gp = "coverage_GP_b0_9"
    rf = "coverage_RF"
    values = [
        (
            archive[gp]["cohorts"]["external_motion"]["mae"],
            archive[rf]["cohorts"]["external_motion"]["mae"],
        ),
        (
            archive[gp]["cohorts"]["external_link25"]["mae"],
            archive[rf]["cohorts"]["external_link25"]["mae"],
        ),
        (repeated[gp]["raw"]["mae"], repeated[rf]["raw"]["mae"]),
        (held[gp]["raw"]["mae"], held[rf]["raw"]["mae"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(4)
    for j, (color, label) in enumerate([("#235f90", "GP"), ("#ed9f42", "RF")]):
        v = [p[j] for p in values]
        bars = ax.bar(x + (j - 0.5) * 0.32, v, 0.32, color=color, label=label)
        ax.bar_label(bars, labels=[f"{n:.4f}" for n in v], padding=3, fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=[
            "Archived driving\n7,339 /20 sessions",
            "Archived buffered roads\n99 /10 links",
            "26R1: recreated scenes\n144 /6 sites",
            "26R1: held road links\n96 /4 sites",
        ],
        ylabel="Whole-frame IoU MAE",
        ylim=(0, 0.145),
        title="Frozen28-input models: results depend on the evaluation setting",
    )
    ax.grid(axis="y", alpha=0.15)
    ax.legend()
    fig.tight_layout()
    finish(
        fig,
        "09_setup_transfer",
        "The144-frame26R1 set recreates old quality-training scenes. The96-frame set uses four quality-training-held links on the same map. All weights and96-frame calibration were frozen before test scoring. Four held sites do not establish new-map generalization.",
    )
    data, _ = load(r / "packs", ["release26_heldroads"])
    with np.load(r / "test26_heldroads" / f"{gp}.npz") as z:
        mean = z["mean"]
        cm = z["calibrated_mean"]
        cov = z["calibrated_covariance"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, code, title in zip(
        axes,
        [1, 2],
        ["Both bands: reduced recall", "Near recall retained, far recall reduced"],
    ):
        indices = np.flatnonzero(
            (data["error_type"] == code) & ((data["y"] >= 0.7) if code == 2 else True)
        )
        i = int(indices[0])
        ax.add_patch(Rectangle((0.8, -0.2), 0.4, 0.7, color="#d4eade", alpha=0.5))
        ax.add_patch(Rectangle((-0.2, -0.2), 1.0, 0.7, color="#f6dfb7", alpha=0.35))
        c = cov[i, 1:, 1:]
        w, v = np.linalg.eigh(c)
        order = np.argsort(w)[::-1]
        w = w[order]
        v = v[:, order]
        angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
        width, height = 2 * np.sqrt(np.maximum(w, 0) * 5.991464547)
        ax.add_patch(
            Ellipse(
                cm[i, 1:],
                width,
                height,
                angle=angle,
                facecolor="#5c91b0",
                edgecolor="#235f90",
                alpha=0.3,
            )
        )
        ax.scatter(*cm[i, 1:], marker="o", c="#235f90", label="GP calibrated mean")
        ax.scatter(*data["Y"][i, 1:], marker="x", s=90, c="#b54b43", label="Offline GT")
        ax.axvline(0.8, color="gray", ls="--")
        ax.axhline(0.5, color="gray", ls="--")
        ax.plot([0, 1], [0, 1], color="gray", alpha=0.25)
        ax.set(
            xlim=(-0.2, 1.2),
            ylim=(-0.2, 1.2),
            xlabel="Near-band lane recall",
            ylabel="Far-band lane recall",
            title=title,
        )
        ax.legend(fontsize=9)
    fig.suptitle(
        "Spatial diagnostics use auxiliary means and covariance; sigma alone does not locate omissions"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    finish(
        fig,
        "10_spatial_posterior",
        "First eligible held examples by diagnostic type, not selected for prediction accuracy. Ellipses are95% Gaussian contours of empirically adjusted moments, not guaranteed joint coverage. Near recall is not near precision; far undercoverage remains.",
    )
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.8))
    sigma = np.sqrt(cov[:, 2, 2])
    valid = np.isfinite(data["Y"][:, 2])
    for j, site in enumerate(np.unique(data["site"])):
        ix = (data["site"] == site) & valid
        ax.errorbar(
            data["Y"][ix, 2],
            cm[ix, 2],
            yerr=1.644853627 * sigma[ix],
            fmt="o",
            ms=3,
            alpha=0.55,
            label=f"Site {j + 1}",
        )
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
    ax.set(
        xlabel="GT far-band recall",
        ylabel="GP far-band recall ±1.645 sigma",
        xlim=(0, 1.02),
        ylim=(-0.12, 1.02),
        title="Far-recall uncertainty is not ready for automatic admission",
    )
    ax.legend(fontsize=8)
    coverage = held[gp]["calibration26"]["coverage90"]
    bars = bx.bar(
        ["Whole IoU", "Near recall", "Far recall"],
        coverage,
        color=["#235f90", "#66a08a", "#d47a64"],
    )
    bx.axhline(0.9, color="gray", ls="--", label="Nominal90%")
    bx.bar_label(bars, labels=[f"{v:.1%}" for v in coverage], padding=3)
    bx.set(ylim=(0, 1.1), title="Observed interval coverage on96 held-road frames")
    bx.legend()
    fig.tight_layout()
    finish(
        fig,
        "11_far_recall_reliability",
        "Calibration uses96 frames at four separate calibration poses. Held-road global IoU coverage91.7%, far-recall coverage39.6%; score ranking and probability calibration are different properties. Spatial scores remain diagnostic-only.",
    )
    sources = read(r / "receipts/gallery26_sources.json")
    predictions = [
        __import__("json").loads(line)
        for line in (r / "gallery26_replay/predictions.jsonl").read_text().splitlines()
    ]
    profile = SpatialProfile.load()
    regions = profile.native[::2, ::2]
    components = out / "held_components"
    components.mkdir(exist_ok=True)
    gallery = []
    for i, (source, prediction) in enumerate(zip(sources, predictions)):
        assert source["sample_id"] == prediction["sample_id"]
        image = a.data_root / source["image"]
        semantic = a.data_root / source["semantic"]
        assert (
            sha(image) == source["expected_sha256"]["rgb"]
            and sha(semantic) == source["expected_sha256"]["semantic"]
        )
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        sem = cv2.imread(str(semantic))
        target = np.all(sem == (255, 255, 255), 2) | np.all(sem == (0, 255, 255), 2)
        target = cv2.resize(
            target.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        with np.load(r / "gallery26_replay" / f"{i:06d}.npz") as z:
            pred = np.isin(z["marking_probability"][:, 12:372].argmax(0), [1, 2])
        tp = pred & target
        fp = pred & ~target
        fn = ~pred & target
        union = int((pred | target).sum())
        q = float(tp.sum() / union) if union else None
        error = np.zeros((360, 640, 3), np.uint8)
        error[tp] = (50, 190, 100)
        error[fp] = (240, 85, 80)
        error[fn] = (55, 165, 235)
        case = components / f"case_{i + 1:02d}"
        case.mkdir(exist_ok=True)
        shutil.copyfile(image, case / "rgb_original.png")
        shutil.copyfile(semantic, case / "semantic_original.png")
        cv2.imwrite(str(case / "gt_lane_model_grid.png"), target.astype(np.uint8) * 255)
        cv2.imwrite(
            str(case / "prediction_model_grid.png"), pred.astype(np.uint8) * 255
        )
        cv2.imwrite(
            str(case / "error_model_grid.png"), cv2.cvtColor(error, cv2.COLOR_RGB2BGR)
        )
        gallery.append(
            dict(
                case=i + 1,
                rgb_sha256=sha(image),
                semantic_sha256=sha(semantic),
                true_iou=q,
                estimated_iou=prediction["predicted_iou"],
                source_role="Prospective held-road capture; no source-frame role changed after scoring.",
            )
        )
        if i not in [0, 1, 4, 5]:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.1))
        for axis, picture, title in zip(
            axes.ravel(),
            [rgb, target, pred, error],
            [
                "Original RGB",
                "Aligned Semantic GT: white/yellow union",
                "Fixed v6 prediction",
                "Green TP / red FP / blue FN",
            ],
        ):
            axis.imshow(picture, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"Held-road case {i + 1}: true IoU {q:.3f}, GP estimate {prediction['predicted_iou']:.3f}",
            fontsize=14,
        )
        fig.tight_layout(h_pad=2.0, w_pad=1.0, rect=(0, 0, 1, 0.95))
        finish(
            fig,
            f"12_held_case_{i + 1:02d}",
            "Four fixed day/night examples at the first two held sites; selection by site/weather/hour, not error. Matching native source GT and exact replay prediction;12-row padding removed, GT nearest-neighbour mapped to640x360. Full-width component files retain16:9 aspect ratio.",
        )
    save(out / "held_gallery_receipt.json", gallery)
    save(
        out / "figure26_receipt.json",
        dict(figures=receipts, source_sha256=sha(Path(__file__))),
    )
    print(
        "saved26R1 evidence",
        len(receipts),
        "figures and",
        len(gallery),
        "sets of native/component images",
        flush=True,
    )


if __name__ == "__main__":
    main()
