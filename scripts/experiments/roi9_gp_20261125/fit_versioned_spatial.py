"""Fit a lineage-bound g1 from regenerated f1 packs, with separate calibration/diagnostics.

This bounded implementation check does not adopt a perception update or compare
unequal training budgets as evidence of model improvement.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from quality_estimation.perception_identity import resolve_quality_perception
from quality_estimation.spatial_identity import (
    bind_model_perception,
    validate_pack_perception,
)
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import calibrate_moments, apply_calibration
from quality_estimation.region_feature_views import feature_matrix
from study import load, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["packs", "adaptation-run", "parent-recipe", "protocol", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _, identity, exposure = resolve_quality_perception(args.adaptation_run)
    weights = "model_state_dict"
    parent = read(args.parent_recipe)
    protocol = read(args.protocol)
    roles = {
        role: protocol[role + "_keys"]
        for role in ["train", "calibration", "diagnostic"]
    }
    all_keys = [key for keys in roles.values() for key in keys]
    if not all(roles.values()) or len(set(all_keys)) != len(all_keys):
        raise ValueError("nonempty disjoint pack roles required")
    data = {}
    for role, keys in roles.items():
        for key in keys:
            validate_pack_perception(
                read(args.packs / (key + ".json")), identity, weights
            )
        data[role], schema = load(args.packs, keys)
    if (
        parent["profile"] != schema["profile"]
        or parent["regions_sha256"] != schema["regions_sha256"]
    ):
        raise ValueError("parent feature order and pack camera profile differ")
    exposure_ids = {row["sample_id"] for row in exposure}
    exposure_pixels = {row["pixel_sha256"] for row in exposure}
    for role, pack in data.items():
        if exposure_ids.intersection(pack["ids"]) or exposure_pixels.intersection(
            pack["pixel_sha256"]
        ):
            raise ValueError(
                "perception training exposure overlaps quality role: " + role
            )
    role_audit = {}
    for i, left in enumerate(roles):
        for right in list(roles)[i + 1 :]:
            a, b = data[left], data[right]
            if set(a["ids"]) & set(b["ids"]) or set(a["pixel_sha256"]) & set(
                b["pixel_sha256"]
            ):
                raise ValueError("quality roles share IDs or decoded pixels")
            distance = float(cKDTree(a["xy"]).query(b["xy"])[0].min())
            if distance < protocol["minimum_role_distance_m"]:
                raise ValueError("quality roles violate spatial embargo")
            role_audit[left + "_vs_" + right] = dict(
                minimum_distance_m=distance, id_or_pixel_overlap=False
            )
    task = parent["tasks"]["quality_spatial_gp"]
    columns = task["ordered_features"]
    inputs = {
        role: feature_matrix(pack, schema, columns) for role, pack in data.items()
    }
    valid = np.isfinite(data["train"]["Y"]).all(1)
    model = SpatialQualityGP(**task["parameters"]).fit(
        inputs["train"][valid], data["train"]["Y"][valid]
    )
    for key, value in dict(
        feature_names_=columns,
        profile_version_=parent["profile"],
        regions_sha256_=parent["regions_sha256"],
        target_names_=["whole_frame_lane_iou", "near_lane_recall", "far_lane_recall"],
    ).items():
        setattr(model, key, np.asarray(value))
    bind_model_perception(model, identity, weights)
    args.output.mkdir(exist_ok=False)
    model_path = args.output / "quality_spatial_gp.npz"
    model.save(model_path)
    mean, cov = model.predict(inputs["calibration"])
    calibration = calibrate_moments(mean, cov, data["calibration"]["Y"])
    calibration.update(
        model_sha256=sha(model_path),
        simulator_release="26.R1",
        perception_identity=identity,
        perception_weights=weights,
    )
    save(args.output / "calibration_26r1.json", calibration)
    pack = data["train"]
    rows = [
        dict(sample_id=str(pack["ids"][i]), pixel_sha256=str(pack["pixel_sha256"][i]))
        for i in np.flatnonzero(valid)
    ]
    save(
        args.output / "training_membership.json",
        dict(
            ordered_features=columns,
            checkpoint_sha256=identity["checkpoint_sha256"],
            perception_identity=identity,
            perception_weights=weights,
            rows=rows,
        ),
    )
    split = dict(
        training=[row["sample_id"] for row in rows],
        calibration=data["calibration"]["ids"].tolist(),
        diagnostic=data["diagnostic"]["ids"].tolist(),
    )
    save(args.output / "split_roles.json", split)
    spec = dict(
        version="roi9-spatial-version-pair-functional-1",
        profile=parent["profile"],
        regions_sha256=parent["regions_sha256"],
        checkpoint_sha256=identity["checkpoint_sha256"],
        perception_identity=identity,
        perception_weights=weights,
        active_simulator_release="26.R1",
        target=parent["target"],
        parent_recipe_sha256=sha(args.parent_recipe),
        status=protocol["scope"],
        tasks=dict(
            quality_spatial_gp=dict(
                kind="SpatialGP",
                ordered_features=columns,
                dimensions=len(columns),
                model_sha256=sha(model_path),
                calibration_sha256=sha(args.output / "calibration_26r1.json"),
                artifact_filename=model_path.name,
                calibration_filename="calibration_26r1.json",
                parameters=task["parameters"],
                training_rows=len(rows),
            )
        ),
        training_membership_sha256=sha(args.output / "training_membership.json"),
        split_roles_sha256=sha(args.output / "split_roles.json"),
    )
    save(args.output / "feature_recipe.json", spec)
    save(args.output / "protocol.json", protocol)
    mean, cov = model.predict(inputs["diagnostic"])
    adjusted, scaled = apply_calibration(mean, cov, calibration)
    truth = data["diagnostic"]["Y"]
    metrics = {}
    for j, name in enumerate(
        ["whole_frame_lane_iou", "near_lane_recall", "far_lane_recall"]
    ):
        good = np.isfinite(truth[:, j])
        std = np.sqrt(scaled[good, j, j])
        error = abs(truth[good, j] - adjusted[good, j])
        metrics[name] = dict(
            defined_rows=int(good.sum()),
            raw_clipped_mae=float(
                abs(np.clip(mean[good, j], 0, 1) - truth[good, j]).mean()
            ),
            calibrated_90_coverage=float((error <= 1.6448536269514722 * std).mean()),
        )
    np.savez_compressed(
        args.output / "diagnostic.npz",
        ids=data["diagnostic"]["ids"],
        X=inputs["diagnostic"],
        Y=truth,
        mean=mean,
        cov=cov,
    )
    save(
        args.output / "summary.json",
        dict(
            scope=protocol["scope"],
            perception_identity=identity,
            perception_weights=weights,
            train_rows=len(rows),
            excluded_undefined_train=int((~valid).sum()),
            calibration_rows=len(data["calibration"]["ids"]),
            diagnostic_rows=len(data["diagnostic"]["ids"]),
            role_audit=role_audit,
            admission_approved=False,
            metrics=metrics,
        ),
    )
    source_names = [
        Path(__file__),
        Path("src/quality_estimation/spatial_identity.py"),
        Path("src/quality_estimation/spatial_gp.py"),
        Path("src/quality_estimation/spatial_predictor.py"),
        Path("src/quality_estimation/infer_spatial_images.py"),
        Path("scripts/experiments/roi9_gp_20261125/extract.py"),
    ]
    sources = {}
    for path in source_names:
        relative = path.name if path.is_absolute() else str(path)
        dest = args.output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        sources[relative] = sha(path)
    save(args.output / "code_manifest.json", sources)
    save(
        args.output / "bundle_manifest.json",
        dict(
            version="roi9-gp-bundle-1",
            training_rows=len(rows),
            artifact_sha256={
                str(p.relative_to(args.output)): sha(p)
                for p in args.output.rglob("*")
                if p.is_file()
            },
            limitations=protocol["scope"],
        ),
    )
    print(
        json.dumps(
            dict(
                training=len(rows),
                diagnostic=len(truth),
                role_audit=role_audit,
                metrics=metrics,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
