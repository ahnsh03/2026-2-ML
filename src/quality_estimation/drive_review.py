"""GT-free inference from a frozen perception/ROI/GP/calibration pair.

Environment groups are RGB descriptors, not weather or failure truth. Priority
is a ranking score; only the four spatial outputs form a joint distribution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from .drive_calibration import (
    apply_moments, apply_probability_map, apply_monotone_binary,
    spatial_probabilities,
)
from .region_feature_views import feature_matrix
from .spatial_composition import independent_far_posterior
from .spatial_gp import SpatialQualityGP


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_environment(features, columns, profile):
    if list(columns) != profile['ordered_features']:
        raise ValueError('RGB environment feature order differs')
    x = np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != len(columns) or not np.isfinite(x).all():
        raise ValueError('finite ordered RGB descriptors required')
    x = (x[:, profile['selected']] - profile['mean']) / profile['scale']
    distances = np.linalg.norm(x[:, None] - np.asarray(profile['centers'])[None], axis=2)
    fine = distances.argmin(1)
    distance = distances[np.arange(len(x)), fine]
    coarse = np.asarray(profile['policy_map'])[fine].copy()
    coarse[distance > profile['maximum_reference_distance']] = -1
    return fine, distance, coarse


def review_priority(global_probability, spatial_probability, environment_probability, weights):
    if weights is None:
        return np.asarray(global_probability).copy()
    far_weight, environment_weight = weights
    if not 0 <= far_weight <= 1 or not 0 <= environment_weight <= 1:
        raise ValueError('policy weights must be in [0,1]')
    return ((1 - environment_weight) * global_probability
            + environment_weight * environment_probability
            + .5 * (spatial_probability[:, 1] + spatial_probability[:, 3])
            + far_weight * spatial_probability[:, 2])


def calibrated_review(mean, covariance, environments, calibration, weights):
    """Accept no labels, GT masks, simulator weather, time or road IDs."""
    events = calibration['event_definition']
    m, c, fallback = apply_moments(mean, covariance, environments, calibration['moments'])
    spatial = apply_probability_map(
        spatial_probabilities(m, c, events['near_recall'], events['far_recall']),
        calibration['joint_map'],
    )
    sd = np.sqrt(np.maximum(c[:, 0, 0], 1e-12))
    low = apply_monotone_binary(ndtr((events['global_iou'] - m[:, 0]) / sd), calibration['global_map'])
    thresholds = np.array([
        calibration['environment_thresholds'].get(str(int(e)), {'threshold': events['global_iou']})['threshold']
        for e in environments
    ])
    relative = ndtr((thresholds - m[:, 0]) / sd)
    unchanged = np.isclose(thresholds, events['global_iou'])
    relative[unchanged] = low[unchanged]
    return dict(mean=m, covariance=c, spatial=spatial, global_probability=low,
                environment_relative_probability=relative, environment_threshold=thresholds,
                conditional_fallback=fallback,
                priority=review_priority(low, spatial, relative, weights))


class DriveReviewPredictor:
    def __init__(self, feature_study, calibration, policy, *, checkpoint_sha256, regions_sha256):
        root, caldir = Path(feature_study), Path(calibration)
        self.selection = json.loads((root / 'selection_frozen.json').read_text())
        self.calibration = json.loads((caldir / 'calibration_frozen.json').read_text())
        self.policy = json.loads(Path(policy).read_text())
        self.environment = json.loads((caldir / 'environment_profile.json').read_text())
        c = self.calibration
        if (c['selection_sha256'] != file_sha(root / 'selection_frozen.json')
            or c['environment_profile_sha256'] != file_sha(caldir / 'environment_profile.json')
            or self.policy['calibration_sha256'] != file_sha(caldir / 'calibration_frozen.json')
            or c['checkpoint_sha256'] != checkpoint_sha256
            or c['regions_sha256'] != regions_sha256):
            raise ValueError('review input/perception/calibration binding differs')
        record = self.selection['chosen']
        path = root / record['model']
        if file_sha(path) != record['model_sha256'] or record['model_sha256'] != c['model_sha256']:
            raise ValueError('primary GP checksum differs')
        self.columns = record['columns']
        self.model = SpatialQualityGP.load(path)
        self._check_model(self.model, self.columns, checkpoint_sha256, regions_sha256)
        self.auxiliary = self.selection.get('far_auxiliary')
        if self.auxiliary != c['far_auxiliary'] or self.auxiliary != self.policy['far_auxiliary']:
            raise ValueError('far auxiliary composition differs')
        self.far_model = None
        if self.auxiliary:
            path = root / self.auxiliary['model']
            if file_sha(path) != self.auxiliary['model_sha256']:
                raise ValueError('far GP checksum differs')
            self.far_model = SpatialQualityGP.load(path)
            self._check_model(self.far_model, self.auxiliary['columns'], checkpoint_sha256, regions_sha256)

    @staticmethod
    def _check_model(model, columns, checkpoint, regions):
        if (model.feature_names_.tolist() != columns
            or str(model.checkpoint_sha256_) != checkpoint
            or str(model.regions_sha256_) != regions):
            raise ValueError('GP ordered features/perception/ROI differ')

    def predict_features(self, features, schema):
        return self.predict_vectors(
            feature_matrix(features, schema, self.columns),
            feature_matrix(features, schema, self.auxiliary['columns']) if self.auxiliary else None,
            feature_matrix(features, schema, self.environment['ordered_features']),
        )

    def predict_vectors(self, primary, far, rgb_environment):
        mean, covariance = self.model.predict(primary)
        if self.far_model is not None:
            fm, fc = self.far_model.predict(far)
            mean, covariance = independent_far_posterior(mean, covariance, fm, fc)
        fine, distance, environment = assign_environment(
            rgb_environment,
            self.environment['ordered_features'], self.environment,
        )
        result = calibrated_review(mean, covariance, environment, self.calibration, self.policy['policy']['weights'])
        return dict(result, raw_mean=mean, raw_covariance=covariance, fine_environment=fine,
                    environment=environment, environment_distance=distance)
