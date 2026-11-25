"""Attach verified offline GT after GP review selection, without approving admission."""

from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from .spatial_identity import validate_pack_perception


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def join_candidates(
    candidates,
    pack,
    source_rows,
    *,
    checkpoint_sha256,
    profile,
    regions_sha256,
    perception_identity=None,
    perception_weights=None,
):
    if profile != "roi9-front-v1":
        raise ValueError("nine-region label counts required")
    lookup = {str(sid): i for i, sid in enumerate(pack["ids"])}
    sources = {r["sample_id"]: r for r in source_rows}
    if len(lookup) != len(pack["ids"]) or len(sources) != len(source_rows):
        raise ValueError("duplicate source IDs")
    result = []
    seen = set()
    for candidate in candidates:
        sid = candidate["sample_id"]
        if sid in seen:
            raise ValueError("duplicate candidate ID")
        seen.add(sid)
        if candidate["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("GT belongs to another perception checkpoint")
        if perception_identity is not None:
            validate_pack_perception(candidate, perception_identity, perception_weights)
        if (
            candidate["profile"] != profile
            or candidate["regions_sha256"] != regions_sha256
        ):
            raise ValueError("ROI profile differs")
        if sid not in lookup or sid not in sources:
            raise ValueError("selected source ID not found")
        source = sources[sid]
        index = lookup[sid]
        if candidate["rgb_sha256"] != source["expected_sha256"]["rgb"]:
            raise ValueError("source RGB checksum differs")
        counts = pack["counts"][index]
        if (
            counts.shape != (9, 3)
            or counts.dtype.kind not in "ui"
            or (counts < 0).any()
        ):
            raise ValueError("invalid offline counts")
        tp, fp, fn = map(int, counts.sum(0))
        union = tp + fp + fn
        q = tp / union if union else None
        cached = float(pack["y"][index])
        if (q is None and np.isfinite(cached)) or (
            q is not None and (not np.isfinite(cached) or abs(cached - q) > 1e-6)
        ):
            raise ValueError("whole-frame label and counts differ")
        far = counts[4:6].sum(0)
        near = counts[6:8].sum(0)
        ns = int(near[0] + near[2])
        fs = int(far[0] + far[2])
        nr = float(near[0] / ns) if ns else None
        fr = float(far[0] / fs) if fs else None
        supported = ns >= 50 and fs >= 50
        spatial = (
            (
                "near_recall_retained_far_recall_reduced"
                if nr >= 0.8 and fr < 0.5
                else "both_bands_low_recall"
                if nr < 0.8 and fr < 0.5
                else "near_band_low_recall"
                if nr < 0.8
                else "both_bands_recall_retained"
            )
            if supported
            else "insufficient_gt_band_support"
        )
        result.append(
            dict(
                candidate,
                source_image=source["image"],
                semantic=source["semantic"],
                semantic_sha256=source["expected_sha256"]["semantic"],
                true_lane_iou=q,
                iou_defined=q is not None,
                near_lane_recall=nr,
                far_lane_recall=fr,
                near_gt_pixels=ns,
                far_gt_pixels=fs,
                spatial_diagnostic=spatial,
                benchmark_low_iou=(q < 0.7) if q is not None else None,
                status="gt_attached_awaiting_operating_policy_review",
                admission_approved=False,
                label_definition="White/yellow union on all nonpadding pixels. Near/far image-band recalls diagnose omissions; low IoU alone is not an operational failure label.",
            )
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("queue-dir", "packs", "source-index", "output-dir"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    run = json.loads((a.queue_dir / "run.json").read_text())
    candidates_path = a.queue_dir / "candidates.jsonl"
    if run["status"] != "complete" or (
        a.queue_dir / "_SUCCESS"
    ).read_text().strip() != sha(a.queue_dir / "run.json"):
        raise ValueError("incomplete candidate queue")
    if sha(candidates_path) != run["candidates_sha256"]:
        raise ValueError("candidate checksum differs")
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text().splitlines()
        if line.strip()
    ]
    pack_path = a.packs / (a.key + ".npz")
    receipt = json.loads(pack_path.with_suffix(".json").read_text())
    schema_path = a.packs / "schema.json"
    schema = json.loads(schema_path.read_text())
    if (
        sha(pack_path) != receipt["sha256"]
        or sha(schema_path) != receipt["schema_sha256"]
        or sha(a.source_index) != receipt["index_sha256"]
    ):
        raise ValueError("GT pack provenance differs")
    source = json.loads(a.source_index.read_text())[a.key]["rows"]
    with np.load(pack_path, allow_pickle=False) as pack:
        if pack["ids"].tolist() != [r["sample_id"] for r in source]:
            raise ValueError("GT/source row order differs")
        joined = join_candidates(
            candidates,
            pack,
            source,
            checkpoint_sha256=receipt["checkpoint_sha256"],
            profile=schema["profile"],
            regions_sha256=schema["regions_sha256"],
            perception_identity=receipt.get("perception_identity"),
            perception_weights=receipt.get("perception_weights"),
        )
    a.output_dir.mkdir(parents=True, exist_ok=False)
    out = a.output_dir / "gt_review.jsonl"
    out.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in joined))
    metadata = dict(
        status="complete",
        uses_ground_truth=True,
        stage="offline_after_candidate_selection",
        rows=len(joined),
        queue_run_sha256=sha(a.queue_dir / "run.json"),
        source_pack_sha256=sha(pack_path),
        source_index_sha256=sha(a.source_index),
        gt_review_sha256=sha(out),
        admission_approved=False,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        perception_identity=receipt.get("perception_identity"),
        perception_weights=receipt.get("perception_weights"),
    )
    (a.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (a.output_dir / "_SUCCESS").write_text(sha(a.output_dir / "run.json") + "\n")
    print("linked offline GT", len(joined), "candidates; admission remains unapproved")


if __name__ == "__main__":
    main()
