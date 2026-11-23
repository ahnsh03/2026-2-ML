"""Offline regional lane-quality estimator for the fixed front-camera profile and Medium v6 baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roi10 import REGIONS


def label_counts(predicted_lane, true_lane, model_regions):
    """Return TP, FP, FN per region. Every unpadded model pixel contributes once."""
    if (predicted_lane.shape != (360, 640) or true_lane.shape != predicted_lane.shape
            or model_regions.shape != predicted_lane.shape or predicted_lane.dtype != bool or true_lane.dtype != bool
            or model_regions.dtype != np.uint8 or np.any(model_regions >= len(REGIONS))):
        raise ValueError("expected two unpadded boolean masks and ten valid region IDs")
    events = (predicted_lane & true_lane, predicted_lane & ~true_lane, ~predicted_lane & true_lane)
    return np.stack([np.bincount(model_regions[event], minlength=len(REGIONS)) for event in events], axis=1)


def local_inputs(x56, nodes, heads, predicted_pixels, region_pixels):
    """Build one row per region using only observations available when GT is absent."""
    n = len(x56)
    if (x56.shape != (n, 56) or nodes.shape != (n, 10, 8) or heads.shape != (n, 10, 4)
            or predicted_pixels.shape != (n, 10) or region_pixels.shape != (10,)):
        raise ValueError("regional feature shape differs")
    if not all(np.all(np.isfinite(a)) for a in (x56, nodes, heads, predicted_pixels, region_pixels)):
        raise ValueError("non-finite regional input")
    if (np.any(region_pixels <= 0) or np.any(predicted_pixels < 0)
            or np.any(predicted_pixels > region_pixels)):
        raise ValueError("regional pixel counts outside physical range")
    region_id = np.broadcast_to(np.eye(10, dtype=np.float32), (n, 10, 10))
    global_features = np.broadcast_to(x56[:, None, :8], (n, 10, 8))
    return np.concatenate([heads, nodes, (predicted_pixels / region_pixels)[:, :, None],
                           region_id, global_features], axis=2)


@dataclass
class RegionalQualityModel:
    """Separate precision and log missed-area regressors per physical region."""
    heads: list

    @classmethod
    def fit(cls, x, predicted_pixels, region_pixels, counts, *, jobs=1, seed=23409):
        from sklearn.ensemble import RandomForestRegressor

        n = len(x)
        if n == 0 or x.shape != (n, 10, 31) or counts.shape != (n, 10, 3):
            raise ValueError("invalid regional training dimensions")
        if (not np.isfinite(x).all() or not np.isfinite(counts).all() or np.any(counts < 0)
                or np.any(counts.sum(2) > region_pixels)):
            raise ValueError("invalid regional training observations")
        if not np.array_equal(counts[:, :, :2].sum(2), predicted_pixels):
            raise ValueError("paired GT tallies disagree with RGB-only predicted areas")
        models = []
        for j in range(10):
            parameters = dict(n_estimators=64, min_samples_leaf=4, max_features=.65,
                              random_state=seed, n_jobs=jobs)
            predicted = predicted_pixels[:, j]
            positive = predicted > 0
            precision = None
            if positive.any():
                precision = RandomForestRegressor(**parameters).fit(
                    x[positive, j], counts[positive, j, 0] / predicted[positive])
                precision.n_jobs = 1
            missed = RandomForestRegressor(**parameters).fit(x[:, j], np.log1p(1000 * counts[:, j, 2] / region_pixels[j]))
            missed.n_jobs = 1
            models.append((precision, missed))
        return cls(models)

    def predict(self, x, predicted_pixels, region_pixels):
        n = len(x)
        if (not n or x.shape != (n, 10, 31) or predicted_pixels.shape != (n, 10)
                or region_pixels.shape != (10,) or len(self.heads) != 10):
            raise ValueError("invalid regional prediction dimensions")
        true_positive = np.zeros((n, 10))
        missed = np.zeros((n, 10))
        for j, (precision, missing) in enumerate(self.heads):
            if precision is not None:
                true_positive[:, j] = predicted_pixels[:, j] * np.clip(precision.predict(x[:, j]), 0., 1.)
            fraction = np.clip(np.expm1(missing.predict(x[:, j])) / 1000., 0., 1.)
            missed[:, j] = fraction * region_pixels[j]
        union = predicted_pixels + missed
        denominator = union.sum(1)
        # Zero estimated support is distinguishable from predicting zero correct lane pixels.
        estimate = np.divide(true_positive.sum(1), denominator, out=np.zeros(n), where=denominator > 0)
        local_iou = np.divide(true_positive, union, out=np.zeros_like(union), where=union > 0)
        return dict(estimated_iou=estimate, estimated_union=denominator, per_region_iou=local_iou,
                    per_region_union=union, estimated_true_positive=true_positive, estimated_missed=missed)
