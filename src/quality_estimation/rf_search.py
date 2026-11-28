"""Fixed-space random-forest tuning on the same training folds as the GP."""
import itertools
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def rf_space(spec):
    parameters, coordinates = [], []
    leaves, fractions, depths = (spec[k] for k in ("min_samples_leaf", "max_features", "max_depth"))
    for leaf, fraction, depth in itertools.product(leaves, fractions, depths):
        parameters.append(dict(n_estimators=spec["trees"], min_samples_leaf=leaf, max_features=fraction, max_depth=depth))
        coordinates.append([np.log2(leaf), fraction, depths.index(depth)])
    coordinates = np.asarray(coordinates, dtype=float)
    width = np.ptp(coordinates, axis=0)
    coordinates = (coordinates-coordinates.min(axis=0))/np.where(width > 0, width, 1)
    return parameters, coordinates, parameters.index(spec["baseline"])


def fit_rf(x, y, parameters, seed):
    return RandomForestRegressor(**parameters, random_state=seed, n_jobs=1).fit(x, y)


def evaluate_rf(x, y, folds, parameters, seed):
    start = time.perf_counter()
    scores = []
    prediction = np.full(len(y), np.nan)
    for name, train, valid in folds:
        model = fit_rf(x[train], y[train], parameters, seed)
        prediction[valid] = np.clip(model.predict(x[valid]), 0, 1)
        scores.append(dict(fold=name, n_train=len(train), n_valid=len(valid),
                           mae=float(np.mean(abs(prediction[valid]-y[valid])))))
    return dict(cv_mae=float(np.mean([r["mae"] for r in scores])), folds=scores,
                seconds=time.perf_counter()-start), prediction
