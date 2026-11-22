"""GT-free quality prediction and explicit, train-only preprocessing."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .features import FEATURE_NAMES, feature_schema


class RankStableOLS(RegressorMixin, BaseEstimator):
    """OLS on numerically resolved directions of the training design matrix.

    Global means are weighted combinations of grid means. Float32 feature
    storage introduces tiny residual directions that ordinary float64 OLS
    can amplify. The relative SVD cutoff is explicit and fitted on X_train.
    """
    def __init__(self, rcond=1e-6):
        self.rcond = rcond

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        if not 0 < self.rcond < 1:
            raise ValueError("OLS relative rank cutoff must be in (0,1)")
        self.n_features_in_ = X.shape[1]
        x_mean, y_mean = X.mean(axis=0), y.mean()
        self.coef_, _, self.rank_, self.singular_ = np.linalg.lstsq(
            X-x_mean, y-y_mean, rcond=self.rcond)
        self.intercept_ = float(y_mean-x_mean@self.coef_)
        return self

    def predict(self, X):
        check_is_fitted(self, ("coef_", "intercept_", "n_features_in_"))
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("OLS feature dimension differs from training")
        return X@self.coef_+self.intercept_


def feature_indices(kind):
    if kind == "full":
        return list(range(len(FEATURE_NAMES)))
    if kind == "global_only":
        return list(range(8))
    if kind == "output_only":
        return [i for i in range(len(FEATURE_NAMES)) if i % 8 < 4]
    if kind == "rgb_only":
        return [i for i in range(len(FEATURE_NAMES)) if i % 8 >= 4]
    raise ValueError("unknown feature set: " + kind)


def make_estimator(kind, params, seed, n_jobs=4):
    steps = [("nonconstant", VarianceThreshold(threshold=0))]
    if kind == "OLS":
        if set(params) not in (set(), {"rcond"}):
            raise ValueError("OLS accepts only an explicit relative rank cutoff")
        regressor = RankStableOLS(**params) if params else LinearRegression()
        steps += [("scale", StandardScaler()), ("regressor", regressor)]
    elif kind == "Ridge":
        if set(params) != {"alpha"} or params["alpha"] <= 0:
            raise ValueError("invalid Ridge parameters")
        steps += [("scale", StandardScaler()), ("regressor", Ridge(alpha=params["alpha"], solver="svd"))]
    elif kind == "RandomForest":
        allowed = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
        if set(params) != allowed:
            raise ValueError("invalid RandomForest parameters")
        steps += [("regressor", RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs))]
    else:
        raise ValueError("unknown model: " + kind)
    return Pipeline(steps)


def predict_array(bundle, X, ordered_features=FEATURE_NAMES):
    """Operational function: X only, never GT or capture/weather metadata."""
    if list(ordered_features) != list(FEATURE_NAMES) or bundle["ordered_features"] != list(FEATURE_NAMES):
        raise ValueError("input feature order differs from trained bundle")
    if bundle["feature_schema"] != feature_schema():
        raise ValueError("feature/target definitions differ from trained bundle")
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or not np.isfinite(X).all():
        raise ValueError("expected finite N x 56 X")
    raw = bundle["estimator"].predict(X[:, bundle["column_indices"]].astype(np.float64))
    if not np.isfinite(raw).all():
        raise ValueError("quality predictions are non-finite")
    estimate = np.clip(raw, 0, 1)
    return {"raw_estimate": raw, "estimated_iou": estimate, "risk_score": 1-estimate}
