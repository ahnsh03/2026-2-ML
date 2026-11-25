"""Load a frozen, ordered, hash-checked GP artifact without GT or simulator labels."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from .spatial_gp import SpatialQualityGP
from .spatial_risk import apply_calibration, spatial_scores
from .spatial_identity import validate_model_perception


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenSpatialPredictor:
    def __init__(
        self,
        recipe,
        model_path,
        profile,
        *,
        task="quality_spatial_gp",
        calibration_path=None,
        simulator_release=None,
    ):
        self.specification = json.loads(Path(recipe).read_text())
        self.recipe_sha256 = file_sha(recipe)
        self.task = self.specification["tasks"][task]
        self.columns = self.task["ordered_features"]
        if (
            self.task["kind"] != "SpatialGP"
            or self.specification["profile"] != profile.version
            or self.specification["regions_sha256"] != profile.regions_sha256
        ):
            raise ValueError("spatial GP recipe and camera profile differ")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("invalid feature order")
        self.model_sha256 = file_sha(model_path)
        if self.model_sha256 != self.task["model_sha256"]:
            raise ValueError("model checksum differs from frozen recipe")
        self.model = SpatialQualityGP.load(model_path)
        self.perception_identity, self.perception_weights = validate_model_perception(
            self.model, self.specification
        )
        if (
            self.model.n_features_in_ != len(self.columns)
            or self.model.feature_names_.tolist() != self.columns
            or str(self.model.profile_version_) != profile.version
            or str(self.model.regions_sha256_) != profile.regions_sha256
            or str(self.model.checkpoint_sha256_)
            != self.specification["checkpoint_sha256"]
        ):
            raise ValueError("embedded artifact input contract differs")
        if self.model.target_names_.tolist() != [
            "whole_frame_lane_iou",
            "near_lane_recall",
            "far_lane_recall",
        ]:
            raise ValueError("unexpected GP target order")
        self.calibration = None
        if calibration_path is not None:
            expected_release = self.specification.get("active_simulator_release")
            if expected_release and simulator_release != expected_release:
                raise ValueError(
                    "declare the matching simulator release for this calibration"
                )
            if file_sha(calibration_path) != self.task["calibration_sha256"]:
                raise ValueError("calibration checksum differs")
            self.calibration = json.loads(Path(calibration_path).read_text())
            if self.calibration["model_sha256"] != self.model_sha256:
                raise ValueError("calibration belongs to another model")
            if (
                self.perception_identity["kind"] != "baseline"
                or "perception_identity" in self.calibration
            ) and (
                self.calibration.get("perception_identity") != self.perception_identity
                or self.calibration.get("perception_weights") != self.perception_weights
            ):
                raise ValueError("updated calibration perception lineage/state differs")
            if (
                expected_release
                and self.calibration.get("simulator_release") != expected_release
            ):
                raise ValueError("calibration simulator release differs from recipe")

    def predict_vector(self, vector):
        vector = np.asarray(vector, np.float32)
        if vector.shape != (len(self.columns),) or not np.isfinite(vector).all():
            raise ValueError("invalid ordered input vector")
        mean, cov = self.model.predict(vector[None])
        raw_mean = mean.copy()
        raw_cov = cov.copy()
        if self.calibration is not None:
            mean, cov = apply_calibration(mean, cov, self.calibration)
        score = spatial_scores(mean, cov)
        contrast, contrast_std = self.model.contrast(mean, cov)
        observation_std = np.sqrt(np.maximum(np.diag(cov[0]), 0))
        latent_variance = (
            np.diag(raw_cov[0]) - self.model.noise_variance * self.model.y_scale_**2
        )
        latent_std = np.sqrt(np.maximum(latent_variance, 0))
        if self.calibration is not None:
            latent_std *= np.asarray(self.calibration["scale"])
        return dict(
            predicted_iou=float(np.clip(raw_mean[0, 0], 0, 1)),
            calibrated_iou=float(np.clip(mean[0, 0], 0, 1)),
            raw_mean=raw_mean[0].tolist(),
            posterior_mean=mean[0].tolist(),
            observation_std=observation_std.tolist(),
            latent_std=latent_std.tolist(),
            posterior_covariance=cov[0].tolist(),
            predicted_near_recall=float(np.clip(raw_mean[0, 1], 0, 1)),
            predicted_far_recall=float(np.clip(raw_mean[0, 2], 0, 1)),
            near_minus_far_mean=float(contrast[0]),
            near_minus_far_std=float(contrast_std[0]),
            risk_scores={key: float(value[0]) for key, value in score.items()},
            calibrated=self.calibration is not None,
            model_sha256=self.model_sha256,
            point_mean_source="uncalibrated_model",
            risk_moments_source="calibrated" if self.calibration else "uncalibrated",
            recipe_sha256=self.recipe_sha256,
            spatial_scope="Image-band lane recall auxiliaries; no GT or ROI IoU at inference. Diagnostic scores are not an operational failure definition.",
        )
