"""Offline Bayesian update of a frozen GP after a small batch has received real GT.

Parent standardization, task coupling, kernel and noise remain fixed. This updates
the quality estimator, not the deep perception network. Selection must precede GT.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from .spatial_gp import SpatialQualityGP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condition_moments(mean, covariance, cross, cholesky, innovation):
    """Apply a frozen real-label update to matching parent moments/cross-covariance."""
    mean = np.array(mean, float, copy=True)
    covariance = np.array(covariance, float, copy=True)
    n, tasks = mean.shape
    k = len(innovation)
    if (
        covariance.shape != (n, tasks, tasks)
        or cross.shape != (n * tasks, k)
        or cholesky.shape != (k, k)
        or not k
        or not all(
            np.isfinite(a).all()
            for a in [mean, covariance, cross, cholesky, innovation]
        )
    ):
        raise ValueError("finite matching parent moments and update factors required")
    mean += (cross @ innovation).reshape(n, tasks)
    projected = solve_triangular(cholesky, cross.T, lower=True).T.reshape(n, tasks, k)
    covariance -= np.einsum("ntk,nuk->ntu", projected, projected)
    covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
    if np.diagonal(covariance, axis1=1, axis2=2).min() < -1e-8:
        raise ValueError("conditional covariance has negative variance")
    return mean, covariance


class ConditionedSpatialGP:
    @classmethod
    def fit(cls, parent_path, X, Y):
        model = cls()
        model.parent_sha256 = sha(parent_path)
        model.parent = SpatialQualityGP.load(parent_path)
        x, y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        tasks = len(model.parent.y_mean_)
        if (
            x.ndim != 2
            or x.shape[1] != model.parent.n_features_in_
            or not 1 <= len(x) <= 32
            or y.shape != (len(x), tasks)
            or not np.isfinite(x).all()
            or np.isinf(y).any()
            or ((y[np.isfinite(y)] < 0) | (y[np.isfinite(y)] > 1)).any()
        ):
            raise ValueError(
                "bounded batch of ordered X and real bounded/undefined GT required"
            )
        model.observed = np.flatnonzero(np.isfinite(y.ravel()))
        if not len(model.observed):
            raise ValueError("no acquired GT values")
        model.X = x
        mean, _ = model.parent.predict(x, observation=False)
        covariance = model.parent.posterior_cross_covariance(x, x).reshape(
            len(x) * tasks, -1
        )
        covariance = covariance[np.ix_(model.observed, model.observed)]
        noise = np.tile(model.parent.noise_variance * model.parent.y_scale_**2, len(x))[
            model.observed
        ]
        covariance += np.diag(noise)
        model.cholesky = np.linalg.cholesky(covariance)
        residual = (y - mean).ravel()[model.observed]
        model.innovation = cho_solve((model.cholesky, True), residual)
        return model

    def predict(self, X, *, observation=True):
        x = np.asarray(X, np.float64)
        mean, covariance = self.parent.predict(x, observation=observation)
        tasks = len(self.parent.y_mean_)
        cross = self.parent.posterior_cross_covariance(x, self.X).reshape(
            len(x) * tasks, -1
        )
        cross = cross[:, self.observed]
        return condition_moments(
            mean, covariance, cross, self.cholesky, self.innovation
        )

    def save(self, path):
        if Path(path).exists():
            raise FileExistsError(path)
        np.savez_compressed(
            path,
            metadata=json.dumps(
                dict(
                    version="conditioned-spatial-gp-1",
                    parent_model_sha256=self.parent_sha256,
                    scope="Real offline labels condition a fixed GP; no perception training or synthetic labels",
                )
            ),
            X=self.X,
            observed=self.observed,
            cholesky=self.cholesky,
            innovation=self.innovation,
        )

    @classmethod
    def load(cls, path, parent_path):
        model = cls()
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            if (
                metadata["version"] != "conditioned-spatial-gp-1"
                or sha(parent_path) != metadata["parent_model_sha256"]
            ):
                raise ValueError("conditional update parent model/version differs")
            model.parent_sha256 = metadata["parent_model_sha256"]
            model.X = z["X"]
            model.observed = z["observed"]
            model.cholesky = z["cholesky"]
            model.innovation = z["innovation"]
        model.parent = SpatialQualityGP.load(parent_path)
        n = len(model.observed)
        tasks = len(model.parent.y_mean_)
        if (
            model.X.ndim != 2
            or model.X.shape[1] != model.parent.n_features_in_
            or not 1 <= len(model.X) <= 32
            or model.observed.dtype.kind not in "iu"
            or model.observed.shape != (n,)
            or not 1 <= n <= len(model.X) * tasks
            or len(np.unique(model.observed)) != n
            or (model.observed < 0).any()
            or (model.observed >= len(model.X) * tasks).any()
            or model.cholesky.shape != (n, n)
            or model.innovation.shape != (n,)
            or not all(
                np.isfinite(a).all()
                for a in [model.X, model.cholesky, model.innovation]
            )
            or not np.array_equal(model.cholesky, np.tril(model.cholesky))
            or (np.diag(model.cholesky) <= 0).any()
        ):
            raise ValueError("invalid conditional update arrays")
        return model
