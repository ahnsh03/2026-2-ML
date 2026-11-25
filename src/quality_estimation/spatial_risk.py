"""GT-free GP risk scoring and nonredundant batch proposals; calibration is offline."""

from __future__ import annotations
import numpy as np
from functools import lru_cache
from scipy.special import ndtr, ndtri
from .spatial_gp import low_quality_probability, expected_shortfall


@lru_cache(maxsize=4)
def _quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


def gaussian_lower_probability(bound, mean, covariance, order=40):
    """Deterministic 2/3-D Gaussian lower-rectangle quadrature.

    Genz-style conditional integration on unit-uniform coordinates with fixed
    Gauss-Legendre nodes. This avoids SciPy-version-specific CDF RNG interfaces
    on the separate Python/CUDA profiles. Numerical accuracy is regression-tested.
    """
    bound = np.asarray(bound, float)
    mean = np.asarray(mean, float)
    covariance = np.asarray(covariance, float)
    if (
        mean.shape not in [(2,), (3,)]
        or bound.shape != mean.shape
        or covariance.shape != (len(mean), len(mean))
    ):
        raise ValueError("two or three Gaussian dimensions required")
    sd = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    a = (bound - mean) / sd
    correlation = covariance / (sd[:, None] * sd[None, :])
    L = np.linalg.cholesky(correlation + np.eye(len(mean)) * 1e-12)
    nodes, weights = _quadrature(order)
    p1 = ndtr(a[0] / L[0, 0])
    eps = np.finfo(float).eps
    z1 = ndtri(np.clip(nodes * p1, eps, 1 - eps))
    p2 = ndtr((a[1] - L[1, 0] * z1) / L[1, 1])
    if len(mean) == 2:
        return float(p1 * np.dot(weights, p2))
    z2 = ndtri(np.clip(p2[:, None] * nodes[None, :], eps, 1 - eps))
    p3 = ndtr((a[2] - L[2, 0] * z1[:, None] - L[2, 1] * z2) / L[2, 2])
    return float(p1 * np.dot(weights, p2 * (p3 @ weights)))


def calibrate_moments(mean, covariance, truth, coverage=0.9):
    """Fit a per-task mean offset and marginal scale on a separate calibration set.

    Framewise empirical scaling is descriptive under correlated simulator frames;
    it supplies no exchangeability or domain-shift coverage guarantee.
    """
    mean = np.asarray(mean, float)
    truth = np.asarray(truth, float)
    covariance = np.asarray(covariance, float)
    if mean.shape != truth.shape or covariance.shape != (
        len(mean),
        mean.shape[1],
        mean.shape[1],
    ):
        raise ValueError("calibration shapes differ")
    if not 0 < coverage < 1:
        raise ValueError("invalid calibration coverage")
    bias = []
    scale = []
    counts = []
    for j in range(mean.shape[1]):
        valid = np.isfinite(truth[:, j])
        residual = truth[valid, j] - mean[valid, j]
        if len(residual) < 20:
            raise ValueError("too few defined calibration rows")
        offset = float(np.median(residual))
        sigma = np.sqrt(np.maximum(covariance[valid, j, j], 1e-12))
        ratio = abs(residual - offset) / sigma
        factor = float(
            np.quantile(ratio, coverage, method="higher") / ndtri((1 + coverage) / 2)
        )
        bias.append(offset)
        scale.append(max(factor, 1e-6))
        counts.append(int(valid.sum()))
    return dict(
        version="spatial-moment-calibration-1",
        bias=bias,
        scale=scale,
        counts=counts,
        coverage=coverage,
        scope="Independent calibration groups only; framewise empirical scale, not a guaranteed interval",
    )


def apply_calibration(mean, covariance, calibration):
    bias = np.asarray(calibration["bias"])
    scale = np.asarray(calibration["scale"])
    return mean + bias, covariance * scale[None, :, None] * scale[None, None, :]


def spatial_scores(mean, covariance, *, seed=23411):
    """Posterior global low-quality and joint near-good/far-weak probabilities."""
    mean = np.asarray(mean, float)
    covariance = np.asarray(covariance, float)
    if (
        mean.ndim != 2
        or mean.shape[1] != 3
        or covariance.shape != (len(mean), 3, 3)
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("three-task finite Gaussian moments required")
    sigma = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    low = low_quality_probability(mean[:, 0], sigma[:, 0], 0.7)
    near_good = 1 - low_quality_probability(mean[:, 1], sigma[:, 1], 0.8)
    far_weak = low_quality_probability(mean[:, 2], sigma[:, 2], 0.5)
    normal_far = np.empty(len(mean))
    far_only = np.empty(len(mean))
    both_weak = np.empty(len(mean))
    sign = np.array([-1.0, -1.0, 1.0])
    bound = np.array([-0.7, -0.8, 0.5])
    for i in range(len(mean)):
        c = covariance[i] * sign[:, None] * sign[None, :]
        if np.max(abs(c - np.diag(np.diag(c)))) < 1e-14:
            normal_far[i] = (1 - low[i]) * near_good[i] * far_weak[i]
            far_only[i] = near_good[i] * far_weak[i]
            both_weak[i] = (1 - near_good[i]) * far_weak[i]
        else:
            normal_far[i] = gaussian_lower_probability(bound, mean[i] * sign, c)
            near_far = covariance[i, 1:, 1:]
            s = np.array([-1.0, 1.0])
            far_only[i] = gaussian_lower_probability(
                [-0.8, 0.5], mean[i, 1:] * s, near_far * s[:, None] * s[None, :]
            )
            both_weak[i] = gaussian_lower_probability([0.8, 0.5], mean[i, 1:], near_far)
    return dict(
        global_low=low,
        far_only=np.clip(far_only, 0, 1),
        both_weak=np.clip(both_weak, 0, 1),
        normal_far_only=np.clip(normal_far, 0, 1),
        expected_shortfall=expected_shortfall(mean[:, 0], sigma[:, 0], 0.7),
        near_minus_far=mean[:, 1] - mean[:, 2],
        sigma=sigma[:, 0],
    )


def conditional_batch(score, covariance, budget, ids, *, noise_variance=1e-3):
    """Risk-weighted anticipated information gain on a fixed candidate window.

    Only variance is conditioned as if selected rows will be labelled later;
    no unseen labels or fictitious posterior means are inserted.
    """
    score = np.asarray(score, float)
    cov = np.asarray(covariance, float).copy()
    ids = np.asarray(ids)
    if (
        cov.shape != (len(score), len(score))
        or len(ids) != len(score)
        or not 0 <= budget <= len(score)
        or noise_variance <= 0
        or not np.isfinite(score).all()
        or not np.isfinite(cov).all()
        or (score < 0).any()
    ):
        raise ValueError("invalid review scores/covariance/budget")
    chosen = []
    available = np.ones(len(score), bool)
    for _ in range(budget):
        variance = np.maximum(np.diag(cov), 0)
        acquisition = score * np.log1p(variance / noise_variance)
        order = np.lexsort((ids, -acquisition))
        i = int(next(i for i in order if available[i]))
        chosen.append(i)
        available[i] = False
        column = cov[:, i].copy()
        cov -= np.outer(column, column) / (max(cov[i, i], 0) + noise_variance)
        cov = (cov + cov.T) / 2
    return np.asarray(chosen, dtype=int)
