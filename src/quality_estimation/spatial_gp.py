"""Exact separable GP for whole-frame quality and offline-trained near/far recall auxiliaries.

The primary quality target remains one whole-frame IoU. Auxiliary recall labels are
computed from GT only during offline training/evaluation. Online predictions use X.
Posterior task covariance permits a near-minus-far contrast with its own uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import ndtr


class SpatialQualityGP:
    def __init__(
        self,
        length_multiplier=2.8,
        noise_variance=0.05,
        task_correlation=0.5,
        feature_groups=None,
        kernel_mode="isotropic",
        kernel_group_weights=None,
    ):
        self.length_multiplier = length_multiplier
        self.noise_variance = noise_variance
        self.task_correlation = task_correlation
        self.feature_groups = feature_groups
        self.kernel_mode = kernel_mode
        self.kernel_group_weights = kernel_group_weights

    def kernel(self, a, b):
        if self.kernel_mode == "additive":
            value = np.zeros((len(a), len(b)))
            for group, weight in enumerate(self.kernel_weights_):
                indices = self.kernel_groups_ == group
                radius = (
                    np.sqrt(3)
                    * cdist(a[:, indices], b[:, indices])
                    / (self.length_multiplier * np.sqrt(indices.sum()))
                )
                value += weight * (1 + radius) * np.exp(-radius)
            return value
        distance = cdist(a, b) / self.length_
        r = np.sqrt(3) * distance
        return (1 + r) * np.exp(-r)

    def fit(self, X, Y):
        x = np.asarray(X, np.float64)
        y = np.asarray(Y, np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if (
            x.ndim != 2
            or y.ndim != 2
            or len(x) != len(y)
            or not 2 <= len(y) <= 2500
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or (y < 0).any()
            or (y > 1).any()
        ):
            raise ValueError(
                "finite aligned inputs and bounded targets within exact GP row budget required"
            )
        if (
            not np.isfinite(
                [self.length_multiplier, self.noise_variance, self.task_correlation]
            ).all()
            or self.length_multiplier <= 0
            or self.noise_variance <= 0
            or not 0 <= self.task_correlation < 1
            or self.kernel_mode not in ("isotropic", "additive")
        ):
            raise ValueError("invalid GP hyperparameters")
        self.n_features_in_ = x.shape[1]
        self.selected_ = np.flatnonzero(x.var(0) > 0)
        if not len(self.selected_):
            raise ValueError("all input columns are constant")
        self.x_weight_ = np.ones(len(self.selected_))
        if self.feature_groups is not None:
            if len(self.feature_groups) != self.n_features_in_:
                raise ValueError("feature group order differs")
            groups = np.asarray(self.feature_groups)[self.selected_]
            names, counts = np.unique(groups, return_counts=True)
            if self.kernel_mode == "isotropic":
                self.x_weight_ = np.asarray(
                    [
                        np.sqrt(
                            len(groups)
                            / (len(names) * counts[np.where(names == g)[0][0]])
                        )
                        for g in groups
                    ]
                )
            else:
                _, self.kernel_groups_ = np.unique(groups, return_inverse=True)
                weights = (
                    np.ones(len(names))
                    if self.kernel_group_weights is None
                    else np.asarray(self.kernel_group_weights, float)
                )
                if (
                    weights.shape != (len(names),)
                    or not np.isfinite(weights).all()
                    or (weights <= 0).any()
                ):
                    raise ValueError(
                        "positive additive weights must match sorted nonconstant feature groups"
                    )
                self.kernel_weights_ = weights / weights.sum()
        elif self.kernel_mode == "additive":
            raise ValueError("additive kernel requires explicit feature groups")
        x = x[:, self.selected_]
        self.x_mean_ = x.mean(0)
        self.x_scale_ = x.std(0)
        self.X_ = (x - self.x_mean_) / self.x_scale_ * self.x_weight_
        self.y_mean_ = y.mean(0)
        self.y_scale_ = y.std(0)
        self.y_scale_[self.y_scale_ == 0] = 1.0
        z = (y - self.y_mean_) / self.y_scale_
        corr = (z.T @ z) / len(z)
        np.fill_diagonal(corr, 1.0)
        self.B_ = (1 - self.task_correlation) * np.eye(
            y.shape[1]
        ) + self.task_correlation * corr
        self.task_eigenvalues_, self.task_eigenvectors_ = eigh(self.B_)
        self.length_ = self.length_multiplier * np.sqrt(len(self.selected_))
        eigenvalues, self.input_eigenvectors_ = eigh(self.kernel(self.X_, self.X_))
        if eigenvalues.min() < -1e-8:
            raise ValueError("input covariance is not PSD")
        self.input_eigenvalues_ = np.maximum(eigenvalues, 0)
        self.denominator_ = (
            self.input_eigenvalues_[:, None] * self.task_eigenvalues_[None, :]
            + self.noise_variance
        )
        transformed = self.input_eigenvectors_.T @ z @ self.task_eigenvectors_
        self.weights_ = self.input_eigenvectors_ @ (transformed / self.denominator_)
        self.log_marginal_likelihood_ = float(
            -0.5
            * (
                np.sum(transformed**2 / self.denominator_)
                + np.log(self.denominator_).sum()
                + y.size * np.log(2 * np.pi)
            )
        )
        return self

    def predict(self, X, *, observation=True, batch_size=512):
        x = np.asarray(X, np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("GP input columns or finite values differ")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        x = (x[:, self.selected_] - self.x_mean_) / self.x_scale_ * self.x_weight_
        means = []
        covariances = []
        for start in range(0, len(x), batch_size):
            k = self.kernel(x[start : start + batch_size], self.X_)
            mean = (k @ self.weights_) * self.task_eigenvalues_[None, :]
            mean = mean @ self.task_eigenvectors_.T
            projected = k @ self.input_eigenvectors_
            variance = self.task_eigenvalues_[None, :] - (projected**2) @ (
                self.task_eigenvalues_[None, :] ** 2 / self.denominator_
            )
            if variance.min() < -1e-7:
                raise ValueError("negative GP posterior variance")
            variance = np.maximum(variance, 0)
            if observation:
                variance += self.noise_variance
            covariance = np.einsum(
                "aj,nj,bj->nab",
                self.task_eigenvectors_,
                variance,
                self.task_eigenvectors_,
            )
            means.append(mean * self.y_scale_ + self.y_mean_)
            covariances.append(
                covariance * self.y_scale_[None, :, None] * self.y_scale_[None, None, :]
            )
        return np.concatenate(means), np.concatenate(covariances)

    @staticmethod
    def contrast(mean, covariance, left=1, right=2):
        mu = mean[:, left] - mean[:, right]
        var = (
            covariance[:, left, left]
            + covariance[:, right, right]
            - 2 * covariance[:, left, right]
        )
        return mu, np.sqrt(np.maximum(var, 0))

    def posterior_covariance(self, X, contrast=None):
        """Latent covariance across a small review window, for a physical task contrast.

        Used to anticipate redundant labels in a batch, without accessing those labels.
        Default contrast is whole-frame IoU; [0, 1, -1] addresses near-minus-far recall.
        """
        x = np.asarray(X, np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("GP input columns or finite values differ")
        v = (
            np.zeros(len(self.y_mean_))
            if contrast is None
            else np.asarray(contrast, np.float64)
        )
        if contrast is None:
            v[0] = 1
        if v.shape != self.y_mean_.shape or not np.isfinite(v).all():
            raise ValueError("invalid task contrast")
        x = (x[:, self.selected_] - self.x_mean_) / self.x_scale_ * self.x_weight_
        scaled = v * self.y_scale_
        prior = float(scaled @ self.B_ @ scaled)
        projected = self.kernel(x, self.X_) @ self.input_eigenvectors_
        coefficients = (
            self.task_eigenvectors_.T @ scaled
        ) ** 2 * self.task_eigenvalues_**2
        weights = (coefficients[None, :] / self.denominator_).sum(1)
        covariance = (
            prior * self.kernel(x, x) - (projected * weights[None, :]) @ projected.T
        )
        return (covariance + covariance.T) / 2

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        names = [
            "n_features_in_",
            "selected_",
            "x_mean_",
            "x_scale_",
            "x_weight_",
            "X_",
            "y_mean_",
            "y_scale_",
            "B_",
            "task_eigenvalues_",
            "task_eigenvectors_",
            "length_",
            "input_eigenvalues_",
            "input_eigenvectors_",
            "denominator_",
            "weights_",
            "log_marginal_likelihood_",
        ]
        names += [
            n
            for n in (
                "feature_names_",
                "profile_version_",
                "regions_sha256_",
                "checkpoint_sha256_",
                "target_names_",
                "perception_identity_json_",
                "perception_weights_",
                "kernel_groups_",
                "kernel_weights_",
            )
            if hasattr(self, n)
        ]
        metadata = dict(
            version="spatial-quality-gp-1",
            length_multiplier=self.length_multiplier,
            noise_variance=self.noise_variance,
            task_correlation=self.task_correlation,
            feature_groups=self.feature_groups,
            kernel_mode=self.kernel_mode,
            kernel_group_weights=self.kernel_group_weights,
        )
        np.savez_compressed(
            path, metadata=json.dumps(metadata), **{n: getattr(self, n) for n in names}
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta.pop("version") != "spatial-quality-gp-1":
                raise ValueError("unknown GP artifact version")
            model = cls(**meta)
            for name in z.files:
                if name != "metadata":
                    setattr(model, name, z[name])
        model.n_features_in_ = int(model.n_features_in_)
        model.length_ = float(model.length_)
        if not hasattr(model, "x_weight_"):
            model.x_weight_ = np.ones(len(model.selected_))
        return model


def low_quality_probability(mean, std, threshold=0.7):
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("invalid Gaussian moments")
    z = np.divide(threshold - mean, std, out=np.zeros_like(mean), where=std > 0)
    return np.where(std > 0, ndtr(z), (mean < threshold).astype(float))


def expected_shortfall(mean, std, threshold=0.7):
    """Gaussian E[(threshold-Q)+], combining predicted severity and uncertainty."""
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    probability = low_quality_probability(mean, std, threshold)
    z = np.divide(threshold - mean, std, out=np.zeros_like(mean), where=std > 0)
    return (threshold - mean) * probability + std * np.exp(-0.5 * z * z) / np.sqrt(
        2 * np.pi
    )
