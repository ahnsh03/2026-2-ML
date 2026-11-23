"""Measure frozen review yield and estimate finite-pool prevalence from delayed GT."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from .offline_review import replay_queue, write_receipt
from .roi10_study import _completed, _requested_identity


def total_interval(population, draws, positives):
    """Invert two hypergeometric tails for the integer total in a fixed finite pool."""
    if not 0 <= positives <= draws <= population or draws == 0:
        raise ValueError("invalid finite-population audit counts")
    # Probabilities of at least x and at most x are monotone in the total.
    def first_true(predicate):
        left, right = positives, population - draws + positives + 1
        while left < right:
            middle = (left + right) // 2
            if predicate(middle):
                right = middle
            else:
                left = middle + 1
        return left
    lower = first_true(lambda total: hypergeom.sf(positives - 1, population, total, draws) >= .025)
    upper = first_true(lambda total: hypergeom.cdf(positives, population, total, draws) < .025) - 1
    return [lower, upper]


def audit_counts(queue, labels, cutoff):
    if not 0 < cutoff < 1:
        raise ValueError("low-IoU cutoff must be strictly between zero and one")
    ids, pixels, counts = labels["ids"], labels["pixel_sha256"], labels["counts"]
    if (len(ids) != len(set(ids)) or pixels.shape != ids.shape or counts.shape != (len(ids), 10, 3)
            or counts.dtype.kind not in "iu" or np.any(counts < 0)):
        raise ValueError("invalid paired GT counts or frame identities")
    lookup = {sid: i for i, sid in enumerate(ids)}
    stats = {k: dict(selected=0, low=0, missing_pair=0, empty_union=0) for k in ("priority", "audit")}
    for selected in queue["selected"]:
        group = stats[selected["reason"]]
        group["selected"] += 1
        i = lookup.get(selected["sample_id"])
        if i is None:
            group["missing_pair"] += 1
            continue
        if pixels[i] != selected["pixel_sha256"]:
            raise ValueError("selected RGB differs from paired GT")
        tp, fp, fn = counts[i].sum(axis=0)
        if tp + fp + fn == 0:
            group["empty_union"] += 1
        else:
            group["low"] += int(tp / (tp + fp + fn) < cutoff)
    total = queue["population"]
    core, audit = stats["priority"], stats["audit"]
    missing = sum(s["missing_pair"] + s["empty_union"] for s in stats.values())
    report = dict(population=total, low_iou_cutoff=cutoff, selection=stats, finite_pool_estimate=None)
    if missing:
        report["estimate_status"] = "incomplete_selected_GT"
    elif core["selected"] == total:
        report.update(estimate_status="census", finite_pool_estimate=dict(
            low_count=core["low"], prevalence=core["low"]/total, count_interval_95=[core["low"]] * 2))
    elif audit["selected"]:
        complement = total - core["selected"]
        estimate = core["low"] + complement * audit["low"] / audit["selected"]
        interval = [core["low"] + x for x in total_interval(complement, audit["selected"], audit["low"])]
        report.update(estimate_status="uniform_complement_audit", finite_pool_estimate=dict(
            low_count=estimate, prevalence=estimate/total, count_interval_95=interval))
    else:
        report["estimate_status"] = "unobserved_complement"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("queue", "scores", "labels", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--adaptation-run", type=Path)
    parser.add_argument("--low-iou", type=float, default=.7)
    args = parser.parse_args()
    identity = _requested_identity(args)
    queue = replay_queue(args.queue, args.scores, identity)
    paired = _completed(args.labels, "rows.npz", identity)
    if not paired["paired"]:
        raise ValueError("audit requires a delayed Semantic pack")
    with np.load(args.labels / "rows.npz", allow_pickle=False) as z:
        result = audit_counts(queue, z, args.low_iou)
    result.update(queue_sha256=hashlib.sha256(args.queue.read_bytes()).hexdigest(),
                  paired_sha256=paired["sha256"]["rows.npz"])
    write_receipt(args.output, result)


if __name__ == "__main__":
    main()
