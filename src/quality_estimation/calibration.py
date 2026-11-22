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


def fit_group_cutoffs(y, estimated_iou, groups, *, min_normals=20, maximum_fpr=.1, target_iou=.7,
                      source_groups=None, min_source_groups=2):
    """Fit group cutoffs, requiring normal support across source groups when supplied."""
    y = np.asarray(y)
    estimates = np.asarray(estimated_iou)
    groups = np.asarray(groups)
    _require(groups.ndim == 1 and groups.shape == y.shape and groups.dtype.kind in "iu",
             "expected integer environment assignments for every calibration row")
    policy = fit_alarm_cutoff(y, estimates, maximum_fpr, target_iou, min_normals)
    if source_groups is not None:
        source_groups = np.asarray(source_groups)
        _require(source_groups.shape == groups.shape and source_groups.dtype.kind in "iuU"
                 and (source_groups.dtype.kind != "U" or np.all(source_groups != "")),
                 "expected one integer or nonempty string source group per calibration row")
        _require(type(min_source_groups) is int and min_source_groups > 0, "invalid minimum source groups")
    by_group = {}
    support = {}
    for label in np.unique(groups):
        mask = groups == label
        normal_mask = mask & (y >= np.asarray(target_iou, dtype=y.dtype))
        normals = int(normal_mask.sum())
        sources = None if source_groups is None else len(np.unique(source_groups[normal_mask]))
        support[str(label)] = {"normal_rows": normals, "normal_source_groups": sources}
        if normals >= min_normals and (sources is None or sources >= min_source_groups):
            entry = fit_alarm_cutoff(y[mask], estimates[mask], maximum_fpr, target_iou, min_normals)
            by_group[str(label)] = entry["alarm_threshold"]
    estimates = np.clip(estimates.astype(np.float64), 0, 1)
    normal = np.isfinite(y) & (y >= np.asarray(target_iou, dtype=y.dtype))
    limits = np.array([by_group.get(str(g), policy["alarm_threshold"]) for g in groups])
    false_alarms = normal & (estimates < limits)
    before = int(false_alarms.sum())
    excess = before - policy["allowed_false_positives"]
    if excess > 0:
        # Sparse groups keep the global fallback. Tighten supported groups at
        # the closest normal scores to their own thresholds. Strict comparison
        # keeps tied scores together and can remove more false alarms than needed.
        adjustable = [i for i in np.flatnonzero(false_alarms) if str(groups[i]) in by_group]
        closest = sorted(adjustable, key=lambda i: (limits[i] - estimates[i], groups[i], estimates[i]))
        for i in closest[:excess]:
            label = str(groups[i])
            by_group[label] = min(by_group[label], float(estimates[i]))
    result = {"global": policy, "group_thresholds": by_group, "min_normals": min_normals,
              "combined_false_positives_before_cap": before, "group_support": support,
              "min_source_groups": min_source_groups if source_groups is not None else None}
    after = int((normal & (estimates < group_alarm_thresholds(result, groups))).sum())
    _require(after <= policy["allowed_false_positives"], "combined calibration budget exceeded")
    result["combined_false_positives"] = after
    return result


def group_alarm_thresholds(policy, groups):
    """Resolve calibrated cutoffs in row order without labels or weather metadata."""
    groups = np.asarray(groups)
    _require(groups.ndim == 1 and groups.dtype.kind in "iu", "expected a vector of integer assignments")
    return np.array([policy["group_thresholds"].get(str(label), policy["global"]["alarm_threshold"])
                     for label in groups], dtype=np.float64)
