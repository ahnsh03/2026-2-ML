"""Verify a portable GP bundle, its ordered inputs, calibration and training membership."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from .spatial_predictor import FrozenSpatialPredictor, file_sha
from .spatial_features import SpatialProfile


def verify_bundle(directory, *, simulator_release):
    root = Path(directory).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    if manifest["version"] != "roi9-gp-bundle-1":
        raise ValueError("unknown bundle version")
    for name, digest in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("bundle path escapes root") from exc
        if file_sha(path) != digest:
            raise ValueError("bundle artifact checksum differs: " + name)
    required = {
        "feature_recipe.json",
        "training_membership.json",
        "split_roles.json",
        "code_manifest.json",
    }
    if not required <= manifest["artifact_sha256"].keys():
        raise ValueError("incomplete bundle manifest")
    recipe = json.loads((root / "feature_recipe.json").read_text())
    task = recipe["tasks"]["quality_spatial_gp"]
    predictor = FrozenSpatialPredictor(
        root / "feature_recipe.json",
        root / task["artifact_filename"],
        SpatialProfile.load(),
        calibration_path=root / task["calibration_filename"],
        simulator_release=simulator_release,
    )
    membership = json.loads((root / "training_membership.json").read_text())
    if (
        membership["ordered_features"] != predictor.columns
        or membership["checkpoint_sha256"] != recipe["checkpoint_sha256"]
    ):
        raise ValueError("training membership input contract differs")
    ids = [r["sample_id"] for r in membership["rows"]]
    if len(ids) != len(set(ids)) or len(ids) != task["training_rows"]:
        raise ValueError("training membership differs")
    if len(ids) != len(predictor.model.X_) or len(ids) != manifest["training_rows"]:
        raise ValueError("fitted model training row count differs")
    if (
        file_sha(root / "training_membership.json")
        != recipe["training_membership_sha256"]
    ):
        raise ValueError("recipe membership differs")
    roles = json.loads((root / "split_roles.json").read_text())
    if file_sha(root / "split_roles.json") != recipe["split_roles_sha256"]:
        raise ValueError("recipe split roles differ")
    if any(len(values) != len(set(values)) for values in roles.values()):
        raise ValueError("duplicate IDs within split role")
    training = set(ids)
    if training != set(roles["training"]):
        raise ValueError("split training IDs differ")
    for name, values in roles.items():
        if name != "training" and training & set(values):
            raise ValueError("training overlaps " + name)
    names = [name for name in roles if name != "training"]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if set(roles[left]) & set(roles[right]):
                raise ValueError("evaluation/calibration roles overlap")
    return dict(
        status="verified",
        dimensions=len(predictor.columns),
        training_rows=len(ids),
        model_sha256=predictor.model_sha256,
        recipe_sha256=predictor.recipe_sha256,
        declared_simulator_release=simulator_release,
        roles_disjoint=True,
        scope="Artifact and membership integrity; this is not final model/policy adoption or a live ROS acceptance test.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--simulator-release", required=True)
    a = ap.parse_args()
    print(
        json.dumps(
            verify_bundle(a.bundle, simulator_release=a.simulator_release), indent=2
        )
    )


if __name__ == "__main__":
    main()
