"""GT-free score adapters and same-budget static/chronological policy replay."""
from collections import defaultdict
import hashlib
import math

import numpy as np
from scipy.special import ndtr

from .review_queue import choose_for_review, review_windows


def low_quality_mask(y, threshold=.7):
    """Compare at the target's stored precision; an exact float32 threshold is normal."""
    y = np.asarray(y)
    if y.dtype.kind != "f" or not np.isfinite(y).all():
        raise ValueError("finite floating IoU labels required")
    return y < np.asarray(threshold, dtype=y.dtype)


def score_rows(metadata, rf, mean, sigma, clusters, policy):
    if not (len(metadata) == len(rf) == len(mean) == len(sigma) == len(clusters)):
        raise ValueError("unaligned review scores")
    audit, priority = .0, "quality"
    if policy == "random":
        audit = 1.
    elif policy == "gp_mean_audit":
        audit = .2
    elif policy == "gp_sigma_audit":
        audit, priority = .2, "quality_uncertainty"
    elif policy == "gp_environment":
        audit, priority = .2, "cluster_round_robin"
    elif policy not in ("rf_mean", "gp_mean", "gp_sigma_only", "gp_lcb"):
        raise ValueError("unknown review policy")
    result = []
    for row, r, m, s, c in zip(metadata, rf, mean, sigma, clusters):
        if not np.isfinite([r, m, s]).all() or s < 0:
            raise ValueError("nonfinite review score or negative sigma")
        score = r if policy == "rf_mean" else m
        if policy == "gp_sigma_only":
            score = 1/(1+s)
        if policy == "gp_lcb":
            # A monotone bounded transform retains raw mean-minus-sigma ordering.
            score = ndtr(m-s)
        result.append(dict(sample_id=row["sample_id"], pixel_sha256=row["pixel_sha256"],
                           estimated_iou=float(np.clip(score, 0, 1)), prediction_std=float(s),
                           environment_cluster=int(c), run_id=row["run_id"], drive_order=row["drive_order"]))
    return result, audit, priority


def replay_policy(rows, mode, percent, seed, audit, priority, window_frames=64):
    """Return selection and actual eligible population; never accepts targets."""
    selected, population, windows = [], [], []
    if mode == "static":
        unique = {}
        for row in sorted(rows, key=lambda r:r["sample_id"]):
            unique.setdefault(row["pixel_sha256"], row)
        population = list(unique.values())
        budget = math.ceil(len(population)*percent/100)
        selected, detail = choose_for_review(population, budget, audit, seed, priority_policy=priority)
        windows.append(dict(run_id="static_pool", window=0, frames=len(population), budget=budget,
                            audit_count=detail["audit_count"]))
    elif mode == "windows":
        runs = defaultdict(list)
        for row in rows:
            if not row["run_id"] or row["drive_order"] == "":
                raise ValueError("chronological replay requires run and drive order")
            runs[row["run_id"]].append(row)
        for run, entries in sorted(runs.items()):
            orders = [int(r["drive_order"]) for r in entries]
            if len(orders) != len(set(orders)):
                raise ValueError("ambiguous drive order")
            entries.sort(key=lambda r:int(r["drive_order"]))
            unique = {}
            for row in entries:
                unique.setdefault(row["pixel_sha256"], row)
            population.extend(unique.values())
            run_seed = seed+int(hashlib.sha256(run.encode()).hexdigest()[:8], 16)
            for choices, info in review_windows(entries, window_frames, percent, audit, run_seed, priority_policy=priority):
                selected.extend(choices)
                windows.append(dict(info, run_id=run))
    else:
        raise ValueError("unknown replay mode")
    if len({r["sample_id"] for r,_ in selected}) != len(selected):
        raise ValueError("duplicate selected IDs")
    return selected, population, windows
