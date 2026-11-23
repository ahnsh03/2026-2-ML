"""Attach delayed candidate annotations to a reproducible ROI10 review list for retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .offline_review import replay_queue, write_once
from .roi10_study import _requested_identity


def join_candidates(queue, candidates, queue_sha256):
    by_id = {r["sample_id"]: r for r in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate candidate ID")
    result = []
    for chosen in queue["selected"]:
        row = by_id.get(chosen["sample_id"])
        if row is None or row.get("preregistered_quality_role") != "candidate_pool" or row.get("view") != "front":
            raise ValueError("reviewed frame needs its reserved front-camera candidate annotation")
        expected = row.get("expected_sha256", {})
        recorded = row.get("input_sha256", {})
        if not isinstance(expected, dict) or not isinstance(recorded, dict):
            raise ValueError("invalid candidate SHA-256 inventory")
        hashes = {**expected, **recorded}
        if (not row.get("image") or not row.get("semantic") or not row.get("parent_session_id")
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(hashes.get(k, ""))) for k in ("rgb", "semantic"))
                or any(expected[k] != recorded[k] for k in expected.keys() & recorded.keys())):
            raise ValueError("missing or inconsistent candidate annotation, physical session or source hash")
        if row.get("pixel_sha256") != chosen["pixel_sha256"]:
            raise ValueError("candidate pixels differ from reviewed RGB")
        evidence = dict(chosen, method=queue["arm"], budget=queue["budget"], population=queue["population"],
                        seed=queue["seed"] if queue["arm"] == "QualityRankAudit" else None,
                        queue_sha256=queue_sha256, inference_run_sha256=queue["inference_run_sha256"],
                        model_run_sha256=queue["model_run_sha256"])
        result.append(dict(row, expected_sha256=hashes, input_sha256=hashes, selection=evidence))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("queue", "scores", "candidates", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--adaptation-run", type=Path)
    args = parser.parse_args()
    queue = replay_queue(args.queue, args.scores, _requested_identity(args))
    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    admitted = join_candidates(queue, candidates, hashlib.sha256(args.queue.read_bytes()).hexdigest())
    write_once(args.output, "".join(json.dumps(r, allow_nan=False) + "\n" for r in admitted))


if __name__ == "__main__":
    main()
