"""Freeze a priority-plus-random offline review list from GT-free ROI10 scores."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np


def review_sample(ids, pixels, scores, budget, seed, *, ranked_only=False):
    ids, pixels, scores = map(np.asarray, (ids, pixels, scores))
    n = ids.size
    if not n or not 0 < budget <= 1 or any(a.shape != (n,) for a in (ids, pixels, scores)):
        raise ValueError("invalid budget or candidate pool shape")
    if not all(isinstance(i, str) and i.strip() for i in ids):
        raise ValueError("candidate IDs must be nonempty strings")
    if not all(isinstance(p, str) and re.fullmatch(r"[0-9a-f]{64}", p) for p in pixels):
        raise ValueError("invalid decoded RGB digest")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("invalid random seed")
    if len(set(ids)) != n or len(set(pixels)) != n:
        raise ValueError("candidate IDs and decoded pixels must be unique")
    if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
        raise ValueError("invalid estimated IoU")
    count = math.ceil(n * budget)
    core = count if ranked_only else count // 2
    ordered = np.lexsort((ids, scores))
    remaining = ordered[core:]
    remaining = remaining[np.argsort(ids[remaining])]
    sampled = np.random.default_rng(seed).permutation(remaining)[:count-core]
    result = []
    for rank, index in enumerate(np.concatenate((ordered[:core], sampled)), 1):
        priority = rank <= core
        result.append(dict(sample_id=str(ids[index]), pixel_sha256=str(pixels[index]),
                           score=float(scores[index]), rank=rank, reason="priority" if priority else "audit",
                           inclusion_probability=1.0 if priority else (count-core)/(n-core)))
    return result


def make_queue(folder, budget, seed, ranked_only=False, identity=None):
    from .roi10_study import _completed
    meta = _completed(folder, "scores.npz", identity)
    if meta.get("overlapping_training_rows") != 0:
        raise ValueError("review pool overlaps quality-model training data")
    if meta.get("uses_gt_for_prediction") is not False or meta.get("input_paired") is True:
        raise ValueError("review requires a GT-free score receipt")
    with np.load(folder / "scores.npz", allow_pickle=False) as scores:
        selected = review_sample(scores["ids"], scores["pixel_sha256"], scores["estimated_iou"],
                                 budget, seed, ranked_only=ranked_only)
        population = len(scores["ids"])
    return dict(arm="FrozenQualityRank" if ranked_only else "QualityRankAudit", seed=seed, budget=budget,
                population=population, inference_run_sha256=hashlib.sha256((folder / "run.json").read_bytes()).hexdigest(),
                model_run_sha256=meta.get("model_run_sha256", meta["model_sha256"]), selected=selected)


def replay_queue(path, scores, identity=None):
    queue = json.loads(path.read_text())
    expected = make_queue(scores, queue["budget"], queue["seed"], queue["arm"] == "FrozenQualityRank", identity)
    if queue != expected:
        raise ValueError("review decisions differ from the frozen score receipt")
    return queue


def write_once(path, content):
    # Serialize and validate before reserving the destination. Publish the complete
    # file with a hard link so a concurrent writer cannot replace a frozen receipt.
    from tempfile import TemporaryDirectory
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / "receipt"
        staged.write_text(content)
        os.link(staged, path)


def write_receipt(path, payload):
    write_once(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    from .roi10_study import _requested_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="Sealed ROI10 lane score directory")
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ranked-only", action="store_true")
    parser.add_argument("--adaptation-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = make_queue(args.scores, args.budget, args.seed, args.ranked_only, _requested_identity(args))
    write_receipt(args.output, queue)


if __name__ == "__main__":
    main()
