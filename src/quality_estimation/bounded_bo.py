"""Finite offline expected-improvement search over measured development objectives."""

from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def next_expected_improvement(coordinates, indices, losses, initial, *, jitter=0.001):
    """Minimize an observed scalar loss; candidates are normalized parameter vectors.

    This surrogate GP models experiment outcomes, not perception IoU at runtime.
    No final-test target or simulator metadata is an input to this function.
    """
    x = np.asarray(coordinates, float)
    indices = np.asarray(indices, int)
    loss = np.asarray(losses, float)
    if (
        x.ndim != 2
        or not np.isfinite(x).all()
        or (x < 0).any()
        or (x > 1).any()
        or indices.shape != loss.shape
        or not np.isfinite(loss).all()
        or len(set(indices)) != len(indices)
        or (indices < 0).any()
        or (indices >= len(x)).any()
        or not 0 <= jitter < 1
    ):
        raise ValueError("invalid bounded BO history")
    for index in initial:
        if index not in indices:
            return int(index)
    remaining = np.array([i for i in range(len(x)) if i not in indices])
    if not len(remaining):
        raise ValueError("search space exhausted")
    kernel = ConstantKernel(1.0, "fixed") * Matern(
        length_scale=[0.35] * x.shape[1], length_scale_bounds="fixed", nu=2.5
    )
    surrogate = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, alpha=0.005, normalize_y=True
    )
    mean, std = surrogate.fit(x[indices], loss).predict(x[remaining], return_std=True)
    delta = loss.min() - mean - jitter
    z = delta / np.maximum(std, 1e-12)
    ei = delta * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return int(remaining[np.argmax(ei)])
