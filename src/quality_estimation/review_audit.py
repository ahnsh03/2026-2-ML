"""Estimate the number of low-IoU frames using only labels acquired for a completed review queue."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .cycle_gate import completed_run
from .dataset_io import load_dataset, require
from .perception_identity import require_matching_identity


def estimate_window(frames, priority, audit):
    """Priority rows are fully observed. Audit rows are uniform without replacement in the remainder."""
    sampled = priority + audit
    require(type(frames) is int and frames >= len(sampled) and frames > 0, "invalid eligible frame count")
    require(all(y is None or type(y) is bool for y in sampled), "invalid audited low-IoU indicator")
    observed = sampled.count(True)
    remainder = frames - len(priority)
    unseen = remainder - len(audit)
    identifiable = not unseen or bool(audit)
    projected = None
    if identifiable:
        projected = priority.count(True) + (remainder * audit.count(True) / len(audit) if audit else 0)
    # These bounds use only observed outcomes. They hold even if no audit slot was assigned.
    return dict(frames=frames, priority=len(priority), audit=len(audit), reviewed=len(sampled),
                undefined_labels=sampled.count(None), observed_low=observed,
                lower_low=observed, upper_low=observed + unseen, estimated_low=projected)


def summarize(windows):
    require(bool(windows), "empty review population")
    totals = {key: sum(w[key] for w in windows) for key in
              ("frames", "priority", "audit", "reviewed", "undefined_labels", "observed_low", "lower_low", "upper_low")}
    unsupported = [w for w in windows if w["estimated_low"] is None]
    estimate = None if unsupported else sum(w["estimated_low"] for w in windows)
    return dict(totals, windows=len(windows), windows_without_audit_support=len(unsupported),
                frames_without_audit_support=sum(w["frames"] for w in unsupported), estimated_low=estimate,
                estimated_low_fraction=estimate/totals["frames"] if estimate is not None else None,
                estimated_missed_low=estimate-totals["observed_low"] if estimate is not None else None)


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_review(directory, dataset):
    directory = Path(directory)
    run = completed_run(directory)
    require(run.get("uses_ground_truth") is False, "expected a GT-free review run")
    require_matching_identity(run["checkpoint_sha256"], run.get("perception_identity"), dataset.metadata.get("perception_identity"))
    inventory = run["artifact_sha256"]
    require("queue.jsonl" in inventory, "missing signed review queue")
    queue = json_lines(directory / "queue.jsonl")
    ids = [r["sample_id"] for r in queue]
    require(len(ids) == len(set(ids)) and set(ids) == set(dataset.ids),
            "provide labels for exactly the reviewed sample IDs")
    labels = {}
    for row, value, defined in zip(dataset.samples, dataset.y, dataset.defined):
        labels[row["sample_id"]] = (row, bool(value < np.float32(.7)) if defined else None)
    slots = defaultdict(lambda: ([], []))
    seen = set()
    for q in queue:
        row, low = labels[q["sample_id"]]
        pixel = q["pixel_sha256"]
        rgb = q.get("rgb_sha256") or q.get("expected_sha256", {}).get("rgb")
        require(pixel and pixel == row.get("pixel_sha256") and pixel not in seen, "reviewed pixel mismatch or duplicate")
        require(rgb and rgb == row["input_sha256"]["rgb"], "reviewed RGB hash differs from labeled RGB")
        seen.add(pixel)
        idx = q.get("window", 0)
        require(type(idx) is int and idx >= 0, "invalid queue window")
        reason = q["review_reason"]
        require(reason in ("random_remainder_audit", "low_estimated_iou", "environment_round_robin",
                           "largest_environment_shortfall", "high_prediction_std"), "unknown review sampling design")
        slots[idx][int(reason == "random_remainder_audit")].append(low)

    if "windows.jsonl" in inventory:
        windows = json_lines(directory / "windows.jsonl")
    elif "windows.json" in inventory:
        windows = json.loads((directory / "windows.json").read_text())
    else:
        policy = run["policy"]
        windows = [dict(window=0, frames=policy["pixel_unique_rows"], budget=policy["budget"],
                        audit_count=policy["audit_count"], unique_seen=policy["pixel_unique_rows"])]
    results = []
    cumulative = 0
    for index, window in enumerate(windows):
        require(window["window"] == index, "window sequence differs")
        priority, audit = slots.pop(index, ([], []))
        require(window["budget"] == len(priority)+len(audit) and window["audit_count"] == len(audit),
                "queue counts differ from signed window")
        cumulative += window["frames"]
        require(window["unique_seen"] == cumulative, "population count differs from signed window")
        results.append(dict(window=index, **estimate_window(window["frames"], priority, audit)))
    require(not slots, "queue references absent windows")
    return dict(schema="ml-review-audit-v1", target="defined lane IoU < 0.7", population="eligible unique frames of this run",
                total=summarize(results), by_window=results,
                review_run_sha256=sha256_file(directory / "run.json"),
                dataset_sha256=sha256_file(dataset.path / "dataset.json"),
                limitations=["Undefined IoU is counted separately from low IoU. The population denominator includes all eligible frames.",
                             "Bounds reflect unobserved outcomes, not confidence intervals. A point estimate requires a sampled remainder in each non-census window.",
                             "The sampling correction applies to this review run only; earlier filtering or another review stage changes its population."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Native X/y built only for the reviewed IDs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "audit output already exists")
    metadata = json.loads((args.review_run / "run.json").read_text())
    dataset = load_dataset(args.dataset_dir, perception_identity=metadata.get("perception_identity"))
    result = audit_review(args.review_run, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
