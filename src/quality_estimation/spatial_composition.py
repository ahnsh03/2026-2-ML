"""Compose independently fitted far diagnostics with the frozen primary Q/near GP."""

from __future__ import annotations
import numpy as np


def independent_far_posterior(
    primary_mean, primary_covariance, far_mean, far_covariance
):
    """Keep the primary Q/near marginal; assume zero covariance to a separate far GP.

    Copying old near/far cross-covariance into a new far variance is invalid. This
    explicit block-diagonal assumption gives a PSD composition, but does not claim
    that the separately trained GPs have learned their joint cross-covariance.
    """
    p = np.asarray(primary_mean, float)
    c = np.asarray(primary_covariance, float)
    f = np.asarray(far_mean, float)
    v = np.asarray(far_covariance, float)
    n = len(p)
    if (
        p.shape != (n, 3)
        or c.shape != (n, 3, 3)
        or f.shape != (n, 1)
        or v.shape != (n, 1, 1)
        or not all(np.isfinite(a).all() for a in (p, c, f, v))
        or (v[:, 0, 0] < 0).any()
    ):
        raise ValueError(
            "aligned finite primary3-task and separate far1-task moments required"
        )
    if (
        not np.allclose(c, c.transpose(0, 2, 1), atol=1e-10, rtol=0)
        or np.linalg.eigvalsh(c).min() < -1e-10
    ):
        raise ValueError("primary covariance must be symmetric PSD")
    mean = p.copy()
    mean[:, 2] = f[:, 0]
    covariance = np.zeros_like(c)
    covariance[:, :2, :2] = c[:, :2, :2]
    covariance[:, 2, 2] = v[:, 0, 0]
    return mean, covariance
