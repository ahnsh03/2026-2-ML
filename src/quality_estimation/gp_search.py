"""Paired finite-grid BO/random search and spatially purged GP evaluation."""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .gaussian_process import fit_quality_gp


def purged_folds(groups, xy, station, bins=3, distance=20.):
    """Hold out each parent x route bin, excluding that parent and nearby points."""
    groups, xy, station = np.asarray(groups), np.asarray(xy), np.asarray(station)
    if (xy.shape != (len(groups), 2) or station.shape != groups.shape
            or not np.isfinite(xy).all() or not np.isfinite(station).all()
            or bins < 2 or distance <= 0 or len(np.unique(groups)) < 2):
        raise ValueError("finite poses/stations, multiple parents and positive spatial purge required")
    edges = np.quantile(station, np.arange(1, bins)/bins)
    bucket = np.digitize(station, edges)
    result = []
    for parent in np.unique(groups):
        for b in range(bins):
            valid = np.flatnonzero((groups == parent) & (bucket == b))
            train = np.flatnonzero((groups != parent) & (bucket != b))
            if not len(valid) or not len(train):
                raise ValueError("empty spatial fold; revise grouping before search")
            train = train[cdist(xy[train], xy[valid]).min(axis=1) >= distance]
            if len(train) < 2:
                raise ValueError("spatial purge left insufficient training rows")
            result.append((str(parent)+"/bin"+str(b), train, valid))
    held = np.concatenate([v for _, _, v in result])
    if not np.array_equal(np.sort(held), np.arange(len(groups))):
        raise ValueError("each training row must be validated exactly once")
    return result


def metrics(y, mean, std):
    """Uncalibrated observation intervals use raw means, never clipped centers."""
    y, mean, std = (np.asarray(a, dtype=float) for a in (y, mean, std))
    if (y.ndim != 1 or not len(y) or y.shape != mean.shape or y.shape != std.shape
            or not all(np.isfinite(a).all() for a in (y, mean, std))
            or np.any(std < 0) or np.any((y < 0) | (y > 1))):
        raise ValueError("expected aligned finite targets, predictions and nonnegative std")
    clipped = np.clip(mean, 0, 1)
    radius = norm.ppf(.95)*std
    low, high = mean-radius, mean+radius
    absolute = abs(mean-y)
    correlation = None
    if len(y) > 1 and np.ptp(std) > 0 and np.ptp(absolute) > 0:
        correlation = float(spearmanr(std, absolute).statistic)
    normal = y >= .7
    failure = ~normal
    alarm = clipped < .7
    return dict(n=len(y), clipped_mae=float(np.mean(abs(clipped-y))),
                raw_mae=float(absolute.mean()), bias=float(np.mean(clipped-y)),
                rmse=float(np.sqrt(np.mean((clipped-y)**2))),
                coverage90_raw=float(np.mean((y >= low) & (y <= high))),
                width90_raw=float(np.mean(high-low)),
                coverage90_display_clipped=float(np.mean((y >= np.maximum(low, 0)) & (y <= np.minimum(high, 1)))),
                width90_display_clipped=float(np.mean(np.maximum(0, np.minimum(high, 1)-np.maximum(low, 0)))),
                sigma_absolute_error_spearman=correlation,
                normal_count=int(normal.sum()), failure_count=int(failure.sum()),
                fp_at_fixed_07=int((normal & alarm).sum()), fn_at_fixed_07=int((failure & ~alarm).sum()))


def evaluate_cv(x, y, folds, parameters):
    start = time.perf_counter()
    mean, std = np.full(len(y), np.nan), np.full(len(y), np.nan)
    scores = []
    for name, train, valid in folds:
        model = fit_quality_gp(x[train], y[train], **parameters)
        mean[valid], std[valid] = model.predict(x[valid], return_std=True)
        scores.append(dict(fold=name, n_train=len(train), n_valid=len(valid),
                           mae=float(np.mean(abs(np.clip(mean[valid], 0, 1)-y[valid])))))
    return dict(cv_mae=float(np.mean([r["mae"] for r in scores])),
                folds=scores, seconds=time.perf_counter()-start), mean, std


def search_space(protocol):
    axes = [np.geomspace(*protocol["bounds"][name], protocol["grid_points_per_axis"])
            for name in ("length_multiplier", "noise_variance")]
    points = np.array(np.meshgrid(*axes, indexing="ij")).reshape(2, -1).T
    base = np.array([protocol["baseline"][k] for k in ("length_multiplier", "noise_variance")])
    points = np.unique(np.vstack([base, points]), axis=0)
    bounds = np.log(np.array([protocol["bounds"][k] for k in ("length_multiplier", "noise_variance")]))
    if np.any(base < np.exp(bounds[:, 0])) or np.any(base > np.exp(bounds[:, 1])):
        raise ValueError("baseline must be inside search bounds")
    coordinates = (np.log(points)-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])
    base_index = int(np.flatnonzero(np.all(points == base, axis=1))[0])
    return points, coordinates, base_index


def shared_start(size, baseline_index, seed, count):
    if not 1 <= count <= size:
        raise ValueError("invalid shared initialization budget")
    others = np.delete(np.arange(size), baseline_index)
    return [baseline_index]+np.random.default_rng(seed).choice(others, count-1, replace=False).tolist()


def expected_improvement(mean, std, incumbent, xi=0.):
    mean, std = np.asarray(mean), np.asarray(std)
    delta = incumbent-mean-xi
    z = delta/np.maximum(std, 1e-15)
    return np.where(std > 1e-15, delta*norm.cdf(z)+std*norm.pdf(z), np.maximum(delta, 0))


def next_point(coordinates, history, initial, method, seed, bo):
    if len(history) < len(initial):
        return initial[len(history)]
    seen = [r["index"] for r in history]
    remaining = np.array([i for i in range(len(coordinates)) if i not in seen])
    if not len(remaining):
        raise ValueError("search space exhausted")
    if method == "random":
        order = np.random.default_rng(seed+100000).permutation(len(coordinates))
        return next(int(i) for i in order if i not in seen)
    if method != "bo":
        raise ValueError("unknown search method")
    surrogate = GaussianProcessRegressor(
        kernel=Matern(length_scale=bo["length_scale"], length_scale_bounds="fixed", nu=2.5),
        alpha=bo["alpha"], normalize_y=True, optimizer=None)
    losses = np.array([r["cv_mae"] for r in history])
    surrogate.fit(coordinates[seen], losses)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Predicted variances smaller than 0.*")
        mean, std = surrogate.predict(coordinates[remaining], return_std=True)
    utility = expected_improvement(mean, std, losses.min(), bo["xi"])
    return int(remaining[np.argmax(utility)])
