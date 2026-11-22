"""Numerical alarm calibration shared by artifact workflows and offline experiments."""
import math

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def fit_alarm_cutoff(y, estimated_iou, max_false_positive_rate=.1, target_iou=.7, min_normals=20):
    """Largest strict-< cutoff retaining the requested empirical normal FPR.

    This is an empirical development constraint, not a population guarantee.
    Ties remain together; no evaluation labels or weather metadata are used.
    """
    y, estimate = np.asarray(y), np.asarray(estimated_iou, np.float64)
    _require(y.ndim == 1 and y.shape == estimate.shape and y.dtype.kind == "f", "invalid calibration arrays")
    _require(0 <= max_false_positive_rate < 1 and 0 < target_iou <= 1 and min_normals >= 1,
            "invalid calibration policy")
    _require(np.isfinite(estimate).all() and not np.isinf(y).any(), "nonfinite calibration data")
    valid = np.isfinite(y)
    _require(((y[valid] >= 0) & (y[valid] <= 1)).all(), "target IoU outside range")
    normal = valid & (y >= np.asarray(target_iou, dtype=y.dtype))
    _require(normal.sum() >= min_normals, "insufficient development normal frames")
    values = np.sort(np.clip(estimate[normal], 0, 1))
    budget = math.floor(len(values)*max_false_positive_rate + 1e-12)
    cutoff = float(values[budget])
    observed = int((values < cutoff).sum())
    _require(observed <= budget, "calibration false-positive budget exceeded")
    return dict(target_iou=float(target_iou), alarm_threshold=cutoff, operator="<",
                max_development_false_positive_rate=float(max_false_positive_rate),
                normal_rows=len(values), allowed_false_positives=budget,
                actual_false_positives=observed, actual_development_false_positive_rate=observed/len(values),
                undefined_targets=int((~valid).sum()), score_dtype="float64",
                rule="normal score order statistic at floor(n*FPR), strict <; ties stay together",
                guarantee_scope="empirical development constraint only; held-out FPR can increase")


def fit_group_cutoffs(y, estimated_iou, groups, *, min_normals=20, maximum_fpr=.1, target_iou=.7):
    """Fit cutoffs on disjoint calibration rows. A sparse or unseen group uses the global cutoff."""
    y = np.asarray(y)
    estimates = np.asarray(estimated_iou)
    groups = np.asarray(groups)
    _require(groups.ndim == 1 and groups.shape == y.shape and groups.dtype.kind in "iu",
             "expected integer environment assignments for every calibration row")
    policy = fit_alarm_cutoff(y, estimates, maximum_fpr, target_iou, min_normals)
    by_group = {}
    for label in np.unique(groups):
        mask = groups == label
        normals = int(np.count_nonzero(y[mask] >= np.asarray(target_iou, dtype=y.dtype)))
        if normals >= min_normals:
            entry = fit_alarm_cutoff(y[mask], estimates[mask], maximum_fpr, target_iou, min_normals)
            by_group[str(label)] = entry["alarm_threshold"]
    return {"global": policy, "group_thresholds": by_group, "min_normals": min_normals}


def group_alarm_thresholds(policy, groups):
    """Resolve calibrated cutoffs in row order without labels or weather metadata."""
    groups = np.asarray(groups)
    _require(groups.ndim == 1 and groups.dtype.kind in "iu", "expected a vector of integer assignments")
    return np.array([policy["group_thresholds"].get(str(label), policy["global"]["alarm_threshold"])
                     for label in groups], dtype=np.float64)
