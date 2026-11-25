"""Publish acquisition aggregates and provenance; raw data/models remain outside Git."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from study import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    r = args.study
    root = r / "acquisition"
    out = args.repo / "docs/baselines/roi9_gp_20261126/acquisition"
    out.mkdir(exist_ok=True)
    for source, destination in [
        ("acquisition/protocol.json", "protocol.json"),
        (
            "acquisition/prospective96_canonical/summary.json",
            "prospective_metrics.json",
        ),
        (
            "acquisition/prospective96_canonical/parent_metrics.json",
            "parent_uncertainty.json",
        ),
        (
            "acquisition/prospective96_canonical/parent_quality_control.json",
            "parent_quality_control.json",
        ),
        (
            "acquisition/canonicalization_validation.json",
            "id_namespace_validation.json",
        ),
        ("acquisition/source_git_binding.json", "source_binding.json"),
        ("reports/acquisition_data_audit.json", "inventory_audit.json"),
    ]:
        save(out / destination, read(r / source))
    frozen = read(root / "updated_models/frozen.json")
    selection = read(root / "selection/selection.json")
    plan = read(root / "test_plan.json")
    save(
        out / "frozen_updates.json",
        dict(
            created_utc=frozen["created_utc"],
            selection_created_utc=selection["created_utc"],
            selection_sha256=frozen["selection_sha256"],
            parent_model_sha256=frozen["parent_model_sha256"],
            inputs_sha256=selection["inputs_sha256"],
            uses_gt_at_selection=False,
            models=[
                {k: v for k, v in model.items() if k not in ["ids", "indices"]}
                for model in frozen["models"]
            ],
            source_sha256=frozen["source_sha256"],
        ),
    )
    restore = read(r / "reports/acquisition_final_simulator_restoration.json")
    audit = read(r / "reports/acquisition_data_audit.json")
    save(
        out / "collection_scope.json",
        dict(
            simulator_release=plan["simulator_release"],
            raw_frames=96,
            raw_channels=4,
            minimum_planned_point_to_quality_training_m=min(
                p["minimum_prior_m"] for p in plan["poses"]
            ),
            minimum_planned_full_link_to_quality_training_m=min(
                p["link_distance_m"] for p in plan["poses"]
            ),
            minimum_actual_test_to_prior26_m=audit[
                "minimum_test_to_prior26_camera_positions_m"
            ],
            minimum_actual_test_to_all_archived_positions_m=audit[
                "minimum_test_to_all_registered_prior_camera_positions_m"
            ],
            geography_clarification="The50m exclusion covered previous26R1 plans, not every archived trajectory. Protocol wording previously captured was broader than its implementation. This is new captured media held apart from quality fitting; not an all-history-unseen road or new-map test.",
            collection_identity_correction="Requested raw siteIDs400..403 collided with an earlier calibration collection. Raw logs/media stayed immutable; canonical sample/site IDs use acq_20261126__ and retain original capture IDs. All27 predictions/covariances/MAE are unchanged.",
            restored_verified=restore["verified"],
            restored_distance_m=restore["distance_3d"],
            restored_yaw_error_deg=restore["yaw_error_deg"],
            restored_speed_mps=restore["speed_mps"],
            udp_actuation_issued=False,
            native_ros_or_steering_acceptance=False,
        ),
    )
    for name in ["17_gp_real_label_acquisition", "18_uncertainty_vs_realized_error"]:
        shutil.copyfile(r / "figures" / (name + ".png"), out / (name + ".png"))
    save(
        out / "figure_receipt.json", read(r / "figures/acquisition_figure_receipt.json")
    )
    save(
        out / "receipt.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(Path(__file__)),
            files={
                p.name: sha(p)
                for p in sorted(out.iterdir())
                if p.is_file() and p.name != "receipt.json"
            },
        ),
    )
    print("Published acquisition evidence", len(list(out.iterdir())), "files")


if __name__ == "__main__":
    main()
