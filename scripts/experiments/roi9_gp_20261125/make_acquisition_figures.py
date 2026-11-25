"""Export fixed16:9 acquisition figures, with small effects and uncertainty explicit."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from study import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    root = args.study
    out = root / "figures"
    result = read(root / "acquisition/prospective96_canonical/summary.json")
    baseline = read(root / "acquisition/prospective96_canonical/parent_metrics.json")
    rows = {(r["method"], r["budget"]): r for r in result["models"]}
    budgets = [4, 8, 16]
    styles = [
        ("rf_mean", "RF mean rank", "#df9c32"),
        ("global_mean", "GP mean rank", "#4f82af"),
        ("global_probability", "GP risk rank", "#8ca9c1"),
        ("dual_probability", "Dual risk rank", "#8a6ba3"),
        ("dual_conditional", "Dual conditional", "#118a7e"),
        ("gp_information_only", "Information only", "#bb6e3e"),
    ]
    receipts = []

    def finish(fig, name, caption):
        fig.savefig(out / (name + ".png"), dpi=200)
        fig.savefig(out / (name + ".svg"))
        plt.close(fig)
        receipts.append(
            dict(
                file=name + ".png",
                sha256=sha(out / (name + ".png")),
                width=2560,
                height=1440,
                caption=caption,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.80, bottom=0.24, wspace=0.46)
    ax = axes[0]
    for method, label, color in styles:
        values = [
            1000 * (rows[(method, b)]["metrics"]["Q"]["mae"] - baseline["Q"]["mae"])
            for b in budgets
        ]
        ax.plot(budgets, values, "o-", label=label, color=color, lw=2, ms=5)
    random = np.array(
        [
            [rows[(f"random_{seed}", b)]["metrics"]["Q"]["mae"] for b in budgets]
            for seed in [23411, 23412, 23413]
        ]
    )
    random = 1000 * (random - baseline["Q"]["mae"])
    ax.fill_between(budgets, random.min(0), random.max(0), color="#999999", alpha=0.16)
    ax.plot(budgets, random.mean(0), "s--", color="#666666", label="Random: 3 seeds")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xticks=budgets,
        xlabel="Acquired GT frames (same budget)",
        ylabel="Change in whole-frame IoU MAE (x10^-3)",
        title="All fixed methods; lower is better",
    )
    ax.grid(alpha=0.15)
    ax = axes[1]
    contrasts = result["contrasts"]["16"]
    for i, (key, label) in enumerate(
        [
            ("conditional_minus_parent", "versus parent"),
            ("conditional_minus_probability", "versus dual risk rank"),
            ("conditional_minus_random_seed_mean", "versus random mean"),
        ]
    ):
        row = contrasts[key]
        center = 1000 * row["mean_difference"]
        low, high = 1000 * np.asarray(row["site_bootstrap95"])
        ax.errorbar(
            center,
            i,
            xerr=[[center - low], [high - center]],
            fmt="o",
            color="#118a7e",
            capsize=5,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        yticks=range(3),
        yticklabels=["versus parent", "versus dual risk rank", "versus random mean"],
        xlabel="Conditional-minus-control MAE (x10^-3)",
        title="16-frame batch: paired 4-site bootstrap",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Real labels update the quality GP: gains are small", fontsize=18, y=0.94
    )
    fig.text(
        0.085,
        0.865,
        f"Parent MAE {baseline['Q']['mae']:.5f}; conditional (B=16) MAE {rows[('dual_conditional', 16)]['metrics']['Q']['mae']:.5f}. Fresh 96 frames / 4 links, same map.",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10),
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.085,
        0.045,
        "Shaded band: three random seeds. Bars: descriptive 95% resampling of only 4 sites. Deep perception is unchanged.",
        fontsize=10,
    )
    finish(
        fig,
        "17_gp_real_label_acquisition",
        "All9methods and3budgets were frozen before fresh capture; random seeds are summarized in the band. Conditional acquisition improves over risk ranking on4 sites, but parent/random contrasts include zero. Not a perception retraining result.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.77, bottom=0.22, wspace=0.30)
    chosen = rows[("dual_conditional", 16)]
    names = ["Q", "near_recall", "far_recall"]
    labels = ["Whole IoU", "Near recall", "Far recall"]
    reductions = [
        100
        * (
            1
            - chosen["metrics"][k]["mean_latent_variance"]
            / baseline[k]["mean_latent_variance"]
        )
        for k in names
    ]
    changes = [1000 * (chosen["metrics"][k]["mae"] - baseline[k]["mae"]) for k in names]
    a = axes[0].bar(labels, reductions, color="#4f82af")
    axes[0].bar_label(a, fmt="%.3f", padding=4)
    axes[0].set(
        ylabel="Reduction in latent posterior variance (%)",
        title="Expected model uncertainty reduction",
        ylim=(0, max(reductions) * 1.35),
    )
    axes[0].grid(axis="y", alpha=0.15)
    a = axes[1].bar(
        labels, changes, color=["#118a7e" if v < 0 else "#c36b53" for v in changes]
    )
    axes[1].bar_label(a, fmt="%+.3f", padding=5)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        ylabel="Change in realized MAE (x10^-3)",
        title="Measured error change: lower is better",
        ylim=(min(changes) * 1.4, max(0.4, max(changes) * 1.8)),
    )
    axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle("Less GP variance does not guarantee less error", fontsize=18, y=0.94)
    fig.text(
        0.09,
        0.865,
        "Fixed GP parameters; 16 real labels from a separate pool. No new interval calibration and no deep-model update.",
        fontsize=11,
    )
    fig.text(
        0.09,
        0.11,
        "Latent variance excludes fixed observation noise. Q/near improve slightly; far-recall error increases in this batch.",
        fontsize=11,
    )
    fig.text(
        0.09,
        0.065,
        "This is a bounded diagnostic on 4 links of the same map; it does not establish operational failure detection.",
        fontsize=10,
    )
    finish(
        fig,
        "18_uncertainty_vs_realized_error",
        "Fixed parent observation noise is unchanged. All three latent variances shrink, while the far-recall point error worsens slightly. Expected information gain and realized accuracy are different quantities.",
    )
    save(
        out / "acquisition_figure_receipt.json",
        dict(
            source_sha256=sha(Path(__file__)),
            results_sha256=sha(
                root / "acquisition/prospective96_canonical/summary.json"
            ),
            figures=receipts,
        ),
    )


if __name__ == "__main__":
    main()
