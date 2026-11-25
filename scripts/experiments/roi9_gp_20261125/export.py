"""Export a feature-ordered GP integration candidate with independent calibration provenance."""

from __future__ import annotations
import argparse, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_gp import SpatialQualityGP
from study import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "calibration", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = read(a.run / "frozen.json")
    record = next(r for r in frozen["models"] if r["name"] == frozen["selected"])
    source = a.run / record["model_file"]
    assert sha(source) == record["model_sha256"]
    columns = record["ordered_features"]
    profile = SpatialProfile.load()
    model = SpatialQualityGP.load(source)
    model.feature_names_ = np.asarray(columns)
    model.profile_version_ = np.asarray(profile.version)
    model.regions_sha256_ = np.asarray(profile.regions_sha256)
    model.checkpoint_sha256_ = np.asarray(a.checkpoint_sha256)
    model.target_names_ = np.asarray(
        ["whole_frame_lane_iou", "near_lane_recall", "far_lane_recall"]
    )
    target = a.output / "quality_spatial_gp.npz"
    model.save(target)
    calibration = read(a.calibration)
    assert calibration["model_sha256"] == sha(source)
    calibration.update(parent_model_sha256=sha(source), model_sha256=sha(target))
    save(a.output / "calibration_legacy.json", calibration)
    task = dict(
        kind="SpatialGP",
        primary_target="one whole-frame white+yellow lane IoU",
        auxiliary_targets=["near image-band lane recall", "far image-band lane recall"],
        ordered_features=columns,
        dimensions=len(columns),
        model_sha256=sha(target),
        calibration_sha256=sha(a.output / "calibration_legacy.json"),
        artifact_filename=target.name,
        calibration_filename="calibration_legacy.json",
        parameters=record["parameters"],
        training_rows=record["rows"],
        required_inputs=["front_rgb", "head_maps", "marking_probability"],
        primary_point_prediction="clipped raw GP mean; calibrated moments are separate risk-score inputs",
    )
    rf = next(r for r in frozen["models"] if r["kind"] == "RF")
    rf_source = a.run / rf["model_file"]
    assert sha(rf_source) == rf["model_sha256"]
    shutil.copyfile(rf_source, a.output / "quality_rf.joblib")
    recipe = dict(
        version="roi9-gp-feature-selection-20261126-v3",
        created_utc=datetime.now(timezone.utc).isoformat(),
        profile=profile.version,
        regions_sha256=profile.regions_sha256,
        checkpoint_sha256=a.checkpoint_sha256,
        target="one whole-frame white+yellow lane IoU",
        tasks=dict(
            quality_spatial_gp=task,
            quality_rf=dict(
                kind="RF",
                ordered_features=columns,
                dimensions=len(columns),
                model_sha256=sha(a.output / "quality_rf.joblib"),
                artifact_filename="quality_rf.joblib",
                training_rows=rf["rows"],
                parameters=rf["parameters"],
                runtime="Use the training scikit-learn dependency version; GP NPZ is the cross-profile inference artifact.",
            ),
        ),
        feature_composition=dict(
            rgb=10, pooled_head=14, raw_softmax=1, near_minus_far_relations=3
        ),
        status="Offline GP integration candidate; final adoption requires independent road policy and perception-retraining retention checks.",
        environment_features="Separate unchanged weather80/hour160/RGB54 e-means study in v2; not automatically appended to quality inputs.",
        selection=dict(
            frozen_run_sha256=sha(a.run / "frozen.json"),
            parent_model_sha256=sha(source),
            calibration_source_sha256=sha(a.calibration),
            limitations="A spatial diagnostic is not an operational failure definition. Region recalls are offline supervision; no online GT or ROI IoU. Gaussian moment calibration has no OOD coverage guarantee.",
        ),
    )
    save(a.output / "feature_recipe.json", recipe)
    save(
        a.output / "receipt.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(Path(__file__)),
            files={p.name: sha(p) for p in a.output.iterdir() if p.is_file()},
            frozen_run_sha256=sha(a.run / "frozen.json"),
            model_weights_parent_sha256=sha(source),
        ),
    )
    print(
        "exported",
        len(columns),
        "features",
        record["rows"],
        "training rows",
        sha(target),
        flush=True,
    )


if __name__ == "__main__":
    main()
