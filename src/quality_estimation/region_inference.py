"""Run a frozen nine-region feature recipe on one recorded original front image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from .selected_region_features import extract_selected_features
from .spatial_features import SpatialProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--head-maps", type=Path)
    parser.add_argument("--marking-probability", type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--perception-checkpoint-sha256")
    parser.add_argument("--simulator-release")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    specification = json.loads(args.recipe.read_text())
    if args.task not in specification["tasks"]:
        parser.error("task is absent from the recipe")
    recipe = specification["tasks"][args.task]
    profile = SpatialProfile.load()
    if (
        specification["profile"] != profile.version
        or specification["regions_sha256"] != profile.regions_sha256
    ):
        parser.error("feature recipe and camera profile differ")
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error("could not read original RGB")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    maps = np.load(args.head_maps, allow_pickle=False) if args.head_maps else None
    columns = recipe["ordered_features"]
    probability = (
        np.load(args.marking_probability, allow_pickle=False)
        if args.marking_probability
        else None
    )
    vector = extract_selected_features(
        rgb, maps, profile, columns, marking_probability=probability
    )
    result = dict(
        task=args.task,
        profile=profile.version,
        regions_sha256=profile.regions_sha256,
        ordered_features=columns,
        values=vector.tolist(),
        recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        rgb_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
    )
    if args.head_maps:
        result["head_maps_sha256"] = hashlib.sha256(
            args.head_maps.read_bytes()
        ).hexdigest()
    if args.marking_probability:
        result["marking_probability_sha256"] = hashlib.sha256(
            args.marking_probability.read_bytes()
        ).hexdigest()
    if args.head_maps or args.marking_probability:
        result["perception_checkpoint_expected_sha256"] = specification[
            "checkpoint_sha256"
        ]
    if args.model:
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if digest != recipe["model_sha256"]:
            parser.error("model checksum differs from frozen recipe")
        result["model_sha256"] = digest
        if recipe["kind"] == "SpatialGP":
            from .spatial_predictor import FrozenSpatialPredictor

            if args.perception_checkpoint_sha256 != specification["checkpoint_sha256"]:
                parser.error(
                    "actual perception checkpoint SHA is required for the spatial GP input contract"
                )
            predictor = FrozenSpatialPredictor(
                args.recipe,
                args.model,
                profile,
                task=args.task,
                calibration_path=args.calibration,
                simulator_release=args.simulator_release,
            )
            result.update(predictor.predict_vector(vector))
        elif recipe["kind"] == "centroids":
            with np.load(args.model, allow_pickle=False) as artifact:
                if artifact["ordered_features"].tolist() != columns:
                    parser.error("centroid feature order differs")
                scaled = vector[artifact["selected"]].copy()
                scaled -= artifact["mean"]
                scaled /= artifact["scale"]
                distances = np.sqrt(
                    cdist(scaled[None], artifact["centers"], "sqeuclidean")[0]
                )
            result["cluster_id"] = int(distances.argmin())
            result["center_distance"] = float(distances.min())
        else:
            portable = recipe.get("format") == "portable-random-forest-1"
            if portable:
                from .portable_forest import PortableForest

                if recipe["kind"] not in ["RF", "SceneRF"]:
                    parser.error("portable forest requires RF or SceneRF task")
                checkpoint = (
                    specification["checkpoint_sha256"]
                    if recipe["kind"] == "RF"
                    else None
                )
                if (
                    checkpoint is not None
                    and args.perception_checkpoint_sha256 != checkpoint
                ):
                    parser.error(
                        "actual perception checkpoint SHA is required for quality RF inputs"
                    )
                model = PortableForest.load(args.model)
                model.validate_context(
                    columns, profile.version, profile.regions_sha256, checkpoint
                )
                expected = "regression" if recipe["kind"] == "RF" else "classification"
                if model.kind != expected:
                    parser.error("portable forest target kind differs")
            else:
                import joblib

                model = joblib.load(args.model)
            if model.n_features_in_ != len(columns):
                parser.error("model input dimension differs")
            if recipe["kind"] != "GP" and not portable:
                model.set_params(n_jobs=1)
            if recipe["kind"] == "GP":
                mean, std = model.predict(vector[None], return_std=True)
                result.update(
                    predicted_iou=float(np.clip(mean[0], 0, 1)),
                    raw_mean=float(mean[0]),
                    observation_std=float(std[0]),
                )
            elif recipe["kind"] == "RF":
                result["predicted_iou"] = float(
                    np.clip(model.predict(vector[None])[0], 0, 1)
                )
            else:
                result["class_id"] = int(model.predict(vector[None])[0])
                result["class_probabilities"] = dict(
                    zip(
                        (str(int(c)) for c in model.classes_),
                        model.predict_proba(vector[None])[0].tolist(),
                    )
                )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
