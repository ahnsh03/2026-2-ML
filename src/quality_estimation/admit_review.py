"""Join a frozen environment review queue to reserved candidate annotations by verified RGB identity."""
import argparse
import json
import math
from pathlib import Path

from common.bridge import sha256_file
from .artifacts import write_json
from .cycle_gate import completed_run
from .dataset_io import require
from .splits import eligible_rows


def admission_rows(candidates, predictions, queue, fraction, evidence):
    """Return the locked queue order with annotation references, without opening any image or label."""
    require(0 < fraction <= 1 and bool(queue), "invalid or empty review budget")
    by_id = {r["sample_id"]: r for r in candidates}
    ids = [r["sample_id"] for r in predictions]
    require(len(by_id) == len(candidates) == len(set(ids)) == len(ids) and set(ids) == set(by_id),
            "candidate manifest and inference IDs differ")
    ordered = []
    for pred in predictions:
        row = by_id[pred["sample_id"]]
        require(row.get("view") == "front" and row.get("preregistered_quality_role") == "candidate_pool",
                "expected reserved front-camera candidates")
        expected = row.get("expected_sha256", {})
        require(expected.get("rgb") == pred["rgb_sha256"] and isinstance(expected.get("semantic"), str)
                and len(expected["semantic"]) == 64 and all(c in "0123456789abcdef" for c in expected["semantic"])
                and bool(row.get("semantic")), "missing or mismatched candidate hashes")
        require(row.get("pixel_sha256", pred["pixel_sha256"]) == pred["pixel_sha256"], "candidate RGB pixels differ")
        prior_hashes = row.get("input_sha256", {})
        require(all(k not in prior_hashes or prior_hashes[k] == v for k, v in expected.items()), "candidate file hashes disagree")
        ordered.append(dict(row, pixel_sha256=pred["pixel_sha256"], input_sha256={**prior_hashes, **expected}))
    keep, _ = eligible_rows(ordered)
    eligible = {r["sample_id"]: r for r, ok in zip(ordered, keep) if ok}
    require(len(eligible) == evidence["pool_rows"] and len(queue) == math.ceil(fraction * len(eligible)),
            "queue budget or candidate thinning differs; supply the eligible RGB pool")
    require(len({r["sample_id"] for r in queue}) == len(queue), "duplicate review ID")
    expected_predictions = {r["sample_id"]: r for r in predictions}
    result = []
    for rank, chosen in enumerate(queue, 1):
        sid = chosen["sample_id"]
        require(sid in eligible and chosen["queue_index"] == rank and chosen["purpose"] == "candidate_pool",
                "review ID, rank or purpose differs")
        pred = expected_predictions[sid]
        require(chosen["expected_sha256"]["rgb"] == pred["rgb_sha256"] and chosen["pixel_sha256"] == pred["pixel_sha256"]
                and chosen["environment_cluster"] == pred["environment_cluster"] and chosen["estimated_iou"] == pred["estimated_iou"],
                "review frame or frozen estimate differs")
        reason = chosen["review_reason"]
        require(reason in ("environment_round_robin", "random_remainder_audit"), "unexpected review reason")
        row = dict(eligible[sid])
        if "expected_pixel_sha256" not in row:
            row.update(expected_pixel_sha256=pred["pixel_sha256"], expected_pixel_color_order="RGB")
        row["selection"] = dict(method="EnvironmentAudit", budget=fraction, rank=rank, seed=None,
                                score=1-pred["estimated_iou"], environment_cluster=pred["environment_cluster"],
                                review_reason=reason, review_seed=evidence["review_seed"],
                                queue_run_sha256=evidence["queue_run_sha256"],
                                environment_centers_sha256=evidence["environment_centers_sha256"],
                                inference_run_sha256=evidence["inference_run_sha256"], model_run_sha256=evidence["model_run_sha256"])
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--budget-fraction", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    queue = completed_run(args.review_queue)
    inference = completed_run(args.inference_dir)
    require(queue.get("purpose") == "candidate_pool" and queue.get("uses_ground_truth") is False
            and inference.get("uses_ground_truth") is False, "expected RGB-only candidate review")
    require("queue.jsonl" in queue["artifact_sha256"] and "predictions.jsonl" in inference["artifact_sha256"],
            "missing frozen queue or predictions")
    source_digest = sha256_file(args.inference_dir / "run.json")
    require(queue["source_inference_run_sha256"] == source_digest and queue["model_run_sha256"] == inference["model_run_sha256"]
            and queue["checkpoint_sha256"] == inference["checkpoint_sha256"], "review and inference models differ")
    policy = queue["policy"]
    centers = policy.get("environment_centers_sha256")
    require(bool(centers) and centers == inference.get("environment_centers_sha256"), "expected matching environment centers")
    evidence = dict(queue_run_sha256=sha256_file(args.review_queue / "run.json"), inference_run_sha256=source_digest,
                    model_run_sha256=inference["model_run_sha256"], environment_centers_sha256=centers,
                    review_seed=policy["seed"], pool_rows=policy["pixel_unique_rows"])

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    rows = admission_rows(read(args.candidate_manifest), read(args.inference_dir / "predictions.jsonl"),
                          read(args.review_queue / "queue.jsonl"), args.budget_fraction, evidence)
    require(len(rows) == policy["budget"] and sum(r["selection"]["review_reason"] == "random_remainder_audit"
                                                   for r in rows) == policy["audit_count"], "audit allocation differs")
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "new_manifest.jsonl"
    with manifest.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(out / "run.json", dict(evidence, status="complete", method="EnvironmentAudit", selected_rows=len(rows),
                                      candidate_manifest_sha256=sha256_file(args.candidate_manifest), annotation_pixels_read=False,
                                      artifact_sha256={manifest.name: sha256_file(manifest)}))
    (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")


if __name__ == "__main__":
    main()
