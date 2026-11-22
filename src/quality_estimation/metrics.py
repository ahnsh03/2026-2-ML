"""Frame regression and equal-budget selection, including undefined-target rows."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr


def failure_detection_metrics(y, estimated_iou, threshold=.7, alarm_threshold=None):
    """Evaluate a fixed IoU alarm; undefined targets never become negatives.

    AUROC is undefined for a single target class. AP is undefined with no
    failures. Neither score nor a threshold tuned on this evaluation is fitted.
    Strict comparisons use each input array's stored floating precision.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    alarm_threshold = threshold if alarm_threshold is None else alarm_threshold
    y, estimate = np.asarray(y), np.asarray(estimated_iou)
    if (y.ndim != 1 or y.shape != estimate.shape or not 0 < threshold <= 1
            or not np.isfinite(alarm_threshold) or not 0 <= alarm_threshold <= 1
            or y.dtype.kind != "f" or estimate.dtype.kind != "f"
            or not np.isfinite(estimate).all() or np.isinf(y).any()
            or ((y[np.isfinite(y)] < 0) | (y[np.isfinite(y)] > 1)).any()):
        raise ValueError("invalid detection inputs")
    valid = np.isfinite(y)
    truth = y[valid] < np.asarray(threshold, dtype=y.dtype)
    pred = np.clip(estimate[valid], 0, 1)
    alarm = pred < np.asarray(alarm_threshold, dtype=estimate.dtype)
    tp, fp = int((truth & alarm).sum()), int((~truth & alarm).sum())
    fn, tn = int((truth & ~alarm).sum()), int((~truth & ~alarm).sum())
    n, positives = int(valid.sum()), int(truth.sum())
    return dict(n=n, undefined=int((~valid).sum()), threshold=threshold, alarm_threshold=float(alarm_threshold),
                target_dtype=str(y.dtype), estimate_dtype=str(estimate.dtype),
                threshold_comparison="strict < at each stored array precision",
                failures=positives, prevalence=positives/n if n else None,
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                alarm_fraction=(tp+fp)/n if n else None,
                auroc=float(roc_auc_score(truth, -pred)) if 0 < positives < n else None,
                average_precision=float(average_precision_score(truth, -pred)) if positives else None)


def regression_metrics(y, raw_prediction):
    y, raw = np.asarray(y, np.float64), np.asarray(raw_prediction, np.float64)
    if y.shape != raw.shape or y.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("invalid regression arrays")
    valid = np.isfinite(y)
    if not valid.any():
        return {"n": 0, "undefined": int(len(y)), "clipped_MAE": None}
    truth, estimate = y[valid], np.clip(raw[valid], 0, 1)
    error = estimate - truth
    correlation = float(spearmanr(truth, estimate).statistic) if np.ptp(truth) and np.ptp(estimate) else None
    return {"n": int(valid.sum()), "undefined": int((~valid).sum()),
            "clipped_MAE": float(np.abs(error).mean()), "raw_MAE": float(np.abs(raw[valid]-truth).mean()),
            "clipped_RMSE": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean()),
            "spearman": correlation, "clipped_predictions": int(((raw < 0) | (raw > 1)).sum()),
            "mean_actual_iou": float(truth.mean()), "mean_predicted_iou": float(estimate.mean())}


def _selected_metrics(y, selected, threshold):
    failures = np.isfinite(y) & (y < threshold)
    total = int(failures.sum())
    hits = int(failures[selected].sum())
    return {"selected": int(len(selected)), "found_failures": hits,
            "total_failures": total, "recall": hits/total if total else None,
            "precision": hits/len(selected) if len(selected) else None,
            "selected_undefined": int((~np.isfinite(y[selected])).sum())}


def selection_metrics(y, ids, risk_scores, budgets=(.05, .1, .2), threshold=.7, repeats=100, seed=20261116):
    """Sort every eligible row before opening labels; undefined rows consume budget."""
    y, ids = np.asarray(y), np.asarray(ids, str)
    # Preserve the label precision for strict threshold comparisons. Promoting a
    # stored float32(0.7) to float64 first makes it incorrectly fall below 0.7.
    if y.dtype not in (np.dtype("float32"), np.dtype("float64")):
        y = y.astype(np.float64)
    n = len(y)
    if y.ndim != 1 or len(ids) != n or len(set(ids)) != n or n == 0 or repeats < 1:
        raise ValueError("invalid candidate pool")
    if not 0 < threshold <= 1 or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("invalid threshold/budgets")
    scores = dict(risk_scores)
    # Oracle is explicitly diagnostic and never used as an operational ranker.
    scores["Oracle"] = -np.where(np.isfinite(y), y, np.inf)
    result = {"pool_rows": n, "defined_rows": int(np.isfinite(y).sum()), "failure_threshold": threshold,
              "target_dtype": str(y.dtype), "threshold_comparison": "strict < at stored target precision",
              "budget_rounding": "ceil", "tie_order": "sample_id ascending", "methods": {}}
    for name, score in scores.items():
        score = np.asarray(score, np.float64)
        if score.shape != y.shape or (name != "Oracle" and not np.isfinite(score).all()):
            raise ValueError("invalid risk score: " + name)
        order = np.lexsort((ids, -score))
        result["methods"][name] = []
        for budget in budgets:
            k = max(1, min(n, math.ceil(n * budget)))
            selected = order[:k]
            result["methods"][name].append(dict(budget_fraction=budget, **_selected_metrics(y, selected, threshold)))
    random_rows = []
    rng = np.random.RandomState(seed)
    orders = [rng.permutation(n) for _ in range(repeats)]
    for budget in budgets:
        k = max(1, min(n, math.ceil(n * budget)))
        trials = [_selected_metrics(y, order[:k], threshold) for order in orders]
        recalls = [t["recall"] for t in trials if t["recall"] is not None]
        random_rows.append({"budget_fraction": budget, "selected": k, "repeats": repeats,
                           "total_failures": trials[0]["total_failures"],
                           "recall_mean": float(np.mean(recalls)) if recalls else None,
                           "recall_std": float(np.std(recalls)) if recalls else None,
                           "analytic_expected_recall": k/n if recalls else None,
                           "found_failures_mean": float(np.mean([t["found_failures"] for t in trials])),
                           "precision_mean": float(np.mean([t["precision"] for t in trials])),
                           "selected_undefined_mean": float(np.mean([t["selected_undefined"] for t in trials]))})
    result["methods"]["Random"] = random_rows
    return result
