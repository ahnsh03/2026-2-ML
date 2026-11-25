"""Publish aggregate evidence and scientific figures without private paths or frame IDs."""

from __future__ import annotations
import argparse, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from quality_estimation.spatial_gp import SpatialQualityGP
from study import read, save, sha, load


def paired_delta(root, key, gp_path, rf_path, group_field="site"):
    data, _ = load(root / "packs", [key] if isinstance(key, str) else key)

    def predictions(path):
        with np.load(path) as z:
            lookup = {sid: i for i, sid in enumerate(z["ids"])}
            return np.asarray([z["mean"][lookup[sid], 0] for sid in data["ids"]])

    gp = predictions(gp_path)
    rf = predictions(rf_path)
    group = data[group_field]
    values = []
    for name in np.unique(group):
        mask = (group == name) & np.isfinite(data["y"])
        if mask.any():
            values.append(
                float(
                    (
                        abs(np.clip(gp[mask], 0, 1) - data["y"][mask])
                        - abs(rf[mask] - data["y"][mask])
                    ).mean()
                )
            )
    values = np.asarray(values)
    boot = (
        np.random.default_rng(23411)
        .choice(values, (2000, len(values)), replace=True)
        .mean(1)
    )
    return dict(
        groups=len(values),
        gp_minus_rf_macro_delta=float(values.mean()),
        paired_group_bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "repo", "bundle"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    out = a.repo / "docs/baselines/roi9_gp_20261126"
    out.mkdir(parents=True, exist_ok=True)
    protocols = out / "protocols"
    protocols.mkdir(exist_ok=True)
    for source, name in [
        ("protocol_v3.json", "feature_screen.json"),
        ("coverage_protocol.json", "coverage.json"),
        ("pruning_views.json", "pruning_views.json"),
        ("combined_pruning_views.json", "combined_pruning_views.json"),
        ("uncertainty_minimal_views.json", "uncertainty_minimal_views.json"),
        ("relation_ablation_views.json", "relation_ablation_views.json"),
        ("packs/schema.json", "candidate_schema.json"),
    ]:
        value = read(r / source)
        if name == "feature_screen.json":
            value["selection"] = (
                "Conditional grouped hyperparameter CV given previously selected24 fields; compact feature selection also uses development. Not fully nested feature-selection CV. External archive opened after the first feature/model freeze; later coverage fits use it only diagnostically."
            )
            value["source_plan_sha256"] = sha(r / source)
            value["publication_scope"] = (
                "Post-run description correcting the original plan wording; original plan remains unchanged on the experiment volume."
            )
        save(protocols / name, value)
    save(
        protocols / "selected_features.json",
        {
            "ordered_features": read(a.bundle / "feature_recipe.json")["tasks"][
                "quality_spatial_gp"
            ]["ordered_features"]
        },
    )
    save(out / "feature_screen.json", read(r / "feature_study/summary.json"))
    save(
        out / "feature_pruning.json",
        {
            name: read(r / name / "summary.json")
            for name in ["pruning_study", "combined_pruning", "uncertainty_minimal"]
        },
    )
    save(
        out / "coverage_and_coupling.json",
        dict(
            coverage=read(r / "coverage/summary.json"),
            folds=read(r / "coverage/folds.json"),
            label_eligibility=read(r / "coverage/label_eligibility.json"),
            feature_selection_scope="Conditional grouped comparisons given previously selected24 fields; hyperparameters fit within folds. Development used for compact feature choice. Not a fully nested feature-selection estimate.",
        ),
    )
    # Excluded sample IDs are private provenance, not needed for aggregate public evidence.
    value = read(out / "coverage_and_coupling.json")
    value["label_eligibility"].pop("excluded_ids", None)
    save(out / "coverage_and_coupling.json", value)
    save(out / "additive_bo_random.json", read(r / "additive/summary.json"))
    save(out / "archival_evaluation.json", read(r / "coverage_evaluation/summary.json"))
    save(
        out / "test26_transfer.json",
        dict(
            lock=read(r / "test26_transfer/lock.json"),
            models=read(r / "test26_transfer/summary.json"),
        ),
    )
    save(
        out / "test26_heldroads.json",
        dict(
            lock=read(r / "test26_heldroads/lock.json"),
            models=read(r / "test26_heldroads/summary.json"),
        ),
    )
    save(out / "review_tradeoff.json", read(r / "policy_study/summary.json"))
    condition = read(r / "reports/condition_control.json")
    save(out / "condition_control.json", condition)
    save(
        out / "label_acquisition.json",
        dict(
            models=read(r / "finalists/summary.json"),
            scope="Retrospective input-only192-frame posterior-variance acquisition replay on624 already-diagnosed images. Three random seeds and all624 controls. Perception weights unchanged.",
        ),
    )
    archive_keys = [
        k for k in read(r / "extraction_index.json") if k.startswith("archive_")
    ]
    deltas = dict(
        archival_sessions=paired_delta(
            r,
            archive_keys,
            r / "coverage_evaluation/predictions/coverage_GP_b0_9.npz",
            r / "coverage_evaluation/predictions/coverage_RF.npz",
            "group",
        ),
        recreated_scenes=paired_delta(
            r,
            "release26_transfer",
            r / "test26_transfer/coverage_GP_b0_9.npz",
            r / "test26_transfer/coverage_RF.npz",
        ),
        held_road_sites=paired_delta(
            r,
            "release26_heldroads",
            r / "test26_heldroads/coverage_GP_b0_9.npz",
            r / "test26_heldroads/coverage_RF.npz",
        ),
    )
    save(out / "paired_group_differences.json", deltas)
    hypotheses = {}
    for name, keys, path in [
        (
            "archival_driving",
            archive_keys,
            r / "coverage_evaluation/predictions/coverage_GP_b0_9.npz",
        ),
        (
            "recreated_scenes",
            ["release26_transfer"],
            r / "test26_transfer/coverage_GP_b0_9.npz",
        ),
        (
            "held_roads",
            ["release26_heldroads"],
            r / "test26_heldroads/coverage_GP_b0_9.npz",
        ),
    ]:
        data, _ = load(r / "packs", keys)
        with np.load(path) as z:
            lookup = {sid: i for i, sid in enumerate(z["ids"])}
            ix = [lookup[sid] for sid in data["ids"]]
            mean = z["mean"][ix]
            sigma = np.sqrt(z["covariance"][ix, 0, 0])
        far = (data["error_type"] == 2) & (data["y"] >= 0.7)
        global_low = data["y"] < 0.7
        chosen = far | global_low
        weak = np.isin(data["error_type"], [1, 2])
        label = data["error_type"][weak] == 2
        hypotheses[name] = dict(
            global_low_vs_normal_far_rows=int(chosen.sum()),
            global_low_positive=True,
            sigma_auc_global_low_vs_normal_far=float(
                roc_auc_score(global_low[chosen], sigma[chosen])
            ),
            negative_mean_auc_global_low_vs_normal_far=float(
                roc_auc_score(global_low[chosen], -mean[chosen, 0])
            ),
            far_only_vs_both_recall_weak_rows=int(weak.sum()),
            hypothesis_negative_sigma_auc_far_only=float(
                roc_auc_score(label, -sigma[weak])
            ),
            predicted_near_recall_auc_far_only=float(
                roc_auc_score(label, mean[weak, 1])
            ),
        )
    save(out / "sigma_hypothesis.json", hypotheses)
    inventories = []
    ids = []
    pixels = []
    keys = list(read(r / "extraction_index.json")) + [
        "release26_calibration",
        "release26_transfer",
        "release26_heldroads",
    ]
    for key in keys:
        file = r / "packs" / (key + ".npz")
        receipt = read(file.with_suffix(".json"))
        assert sha(file) == receipt["sha256"]
        with np.load(file) as z:
            ids.extend(z["ids"])
            pixels.extend(z["pixel_sha256"])
            defined = int(np.isfinite(z["y"]).sum())
            inventories.append(
                dict(
                    key=key,
                    rows=len(z["ids"]),
                    defined_y=defined,
                    pack_sha256=sha(file),
                    simulator_release=receipt["simulator_release"],
                )
            )
    save(
        out / "data_inventory.json",
        dict(
            rows=len(ids),
            unique_ids=len(set(ids)),
            unique_rgb_pixels=len(set(pixels)),
            keys=len(keys),
            new26_accepted_frames=336,
            new26_calibration=96,
            new26_recreated_scenes=144,
            new26_held_roads=96,
            historical_release_provenance="Historical version is user-reported25.S4.MolitComp03; current26.R1.H3 was checked against the running application. Older binaries were not re-verified.",
            sources=inventories,
        ),
    )
    runtime = {}
    for name in [
        "all_prior_rtx50_parity",
        "runtime_feature_impact",
        "cross_profile_replay",
        "replay26_feature_parity",
        "gpu50_training_smoke",
        "cross_profile26_summary",
        "offline_integration_final",
        "bundle_verification_rtx40cpu",
    ]:
        file = r / "reports" / (name + ".json")
        if file.exists():
            value = read(file)
            if name == "all_prior_rtx50_parity":
                value.pop("records", None)
            if name == "gpu50_training_smoke":
                value.pop("loss", None)
            runtime[name] = value
    save(out / "runtime_verification.json", runtime)
    save(out / "bundle_receipt.json", read(a.bundle / "bundle_manifest.json"))
    shutil.copyfile(
        a.bundle / "feature_recipe.json",
        a.repo / "config/region_feature_recipe_v3.json",
    )
    figure_names = [
        "01_roi9_selected_inputs",
        "02_feature_screen",
        "03_sigma_and_spatial_means",
        "04_gp_rf_coverage",
        "05_review_tradeoff",
        "06_calibration_coverage_width",
        "08_offline_pipeline",
        "09_setup_transfer",
        "10_spatial_posterior",
        "11_far_recall_reliability",
    ]
    for name in figure_names:
        file = r / "figures" / (name + ".png")
        if file.exists():
            shutil.copyfile(file, out / file.name)
    for file in sorted((r / "figures").glob("12_held_case_*.png")):
        shutil.copyfile(file, out / file.name)
    save(
        out / "publication_receipt.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(Path(__file__)),
            files={
                str(p.relative_to(out)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and p.name != "publication_receipt.json"
            },
        ),
    )
    print(
        "published aggregate evidence",
        len(ids),
        "registered images; private paths and per-frame tables stay on the experiment volume",
        flush=True,
    )


if __name__ == "__main__":
    main()
