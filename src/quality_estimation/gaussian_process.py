"""Offline Gaussian-process reference for predicting the lane IoU from frame features."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y


def fit_quality_gp(X, y, *, length_multiplier=1.0, noise_variance=0.1, max_rows=2500):
    """Fit on training rows only. Returned pipeline accepts predict(X, return_std=True).

    Length is expressed in units of sqrt(nonconstant dimensions) after scaling.
    The fixed white-noise term is a variance in normalized target space. The
    returned standard deviation includes observation noise and is in IoU units.
    """
    X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
    if len(y) > max_rows or len(y) < 2:
        raise ValueError("choose a representative training subset within the exact GP row budget")
    if np.any((y < 0) | (y > 1)) or np.ptp(y) == 0:
        raise ValueError("expected variable IoU training labels within [0, 1]")
    if not np.isfinite([length_multiplier, noise_variance]).all() or min(length_multiplier, noise_variance) <= 0:
        raise ValueError("length multiplier and observation-noise variance must be positive")
    selector = VarianceThreshold().fit(X)
    scaler = StandardScaler().fit(selector.transform(X))
    scaled = scaler.transform(selector.transform(X))
    kernel = Matern(length_scale=length_multiplier * np.sqrt(scaled.shape[1]),
                    length_scale_bounds="fixed", nu=1.5) + WhiteKernel(
                        noise_level=noise_variance, noise_level_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-8, normalize_y=True)
    model.fit(scaled, y)
    return Pipeline([("nonconstant", selector), ("scale", scaler), ("regressor", model)])
