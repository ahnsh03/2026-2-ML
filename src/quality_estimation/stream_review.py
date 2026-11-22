"""Score incoming front RGB files and preserve selected images as each review window closes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from common.bridge import sha256_file
from .artifacts import write_json
from .build_dataset import collate
from .dataset_io import require
from .image_io import resolve_input
from .infer_images import RGBQualityScorer, load_rgb_row
from .models import QUALITY_MODELS
from .review_queue import PRIORITY_POLICIES, review_windows


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def run_stream(lines, scorer, data_root, out, window_frames, budget_percent,
               audit_fraction=.2, seed=20261116, priority_policy="quality"):
    """Consume one chronological session. Source RGBs must be available until their window closes."""
    require(priority_policy != "cluster_round_robin" or scorer.centers is not None,
            "environment-aware review requires frozen centers")
    require(priority_policy != "quality_uncertainty" or scorer.metadata["model"] == "GaussianProcess",
            "uncertainty review requires a frozen GP")
    out.mkdir(parents=True, exist_ok=False)
    copies = out / "selected_rgb"
    copies.mkdir()
    record = dict(scorer.metadata, status="running", started_utc=utc(),
                  window_frames=window_frames, budget_percent=budget_percent, audit_fraction=audit_fraction,
                  seed=seed, priority_policy=priority_policy, device=str(scorer.device), training_admission_performed=False)
    record["source_sha256"] = {name: sha256_file(Path(__file__).with_name(name)) for name in
                               ("stream_review.py", "infer_images.py", "review_queue.py", "features.py",
                                "environment_features.py", "environment_clusters.py", "calibration.py", "image_io.py")}
    write_json(out / "run.json", record)
    if scorer.centers_bytes is not None:
        (out / "environment_centers.json").write_bytes(scorer.centers_bytes)
    manifest_digest = hashlib.sha256()
    counts = dict(input_rows=0, unique_pixels=0, selected=0, windows=0)

    def predictions(score_log):
        for line in lines:
            if not line.strip():
                continue
            received = utc()
            start = time.perf_counter()
            row = json.loads(line)
            require(isinstance(row, dict) and set(row) <= {"sample_id", "view", "image", "expected_sha256"},
                    "stream input must use the RGB-only manifest fields")
            require(set(row.get("expected_sha256", {})) <= {"rgb"}, "stream input includes a non-RGB hash")
            require(isinstance(row.get("sample_id"), str) and row["sample_id"], "missing sample ID")
            item = load_rgb_row(data_root, row, counts["input_rows"])
            results, _, _ = scorer.score(*collate([item]))
            result = dict(results[0], image=row["image"], received_utc=received,
                          scored_utc=utc(), service_seconds=time.perf_counter()-start)
            manifest_digest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
            counts["input_rows"] += 1
            append(score_log, result)
            yield result

    try:
        with (out / "scores.jsonl").open("x") as scores, (out / "queue.jsonl").open("x") as queue, \
                (out / "windows.jsonl").open("x") as windows:
            for selected, window in review_windows(predictions(scores), window_frames, budget_percent,
                                                   audit_fraction, seed, priority_policy=priority_policy):
                decision_utc = utc()
                for row, reason in selected:
                    source = resolve_input(data_root, row["image"])
                    rank = counts["selected"] + 1
                    destination = copies / (f"{rank:06d}" + source.suffix)
                    shutil.copyfile(source, destination)
                    require(sha256_file(destination) == row["rgb_sha256"], "selected RGB changed during review")
                    append(queue, dict(row, queue_index=rank, window=window["window"], review_reason=reason,
                                       stored_image=str(destination.relative_to(out)), decision_utc=decision_utc))
                    counts["selected"] = rank
                counts.update(unique_pixels=window["unique_seen"], windows=window["window"] + 1)
                append(windows, dict(window, decision_utc=decision_utc, images_saved_utc=utc()))
        require(counts["input_rows"] > 0, "empty input stream")
        hashes = {str(f.relative_to(out)): sha256_file(f) for f in out.rglob("*") if f.is_file() and f.name != "run.json"}
        write_json(out / "run.json", dict(record, **counts, status="complete", finished_utc=utc(),
                                          ordered_manifest_sha256=manifest_digest.hexdigest(), artifact_sha256=hashes))
        (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")
        return counts
    except BaseException as exc:
        write_json(out / "run.json", dict(record, **counts, status="failed", error=str(exc), finished_utc=utc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-run", required=True, type=Path)
    parser.add_argument("--model", choices=("selected", *QUALITY_MODELS), default="selected")
    parser.add_argument("--adaptation-run", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--alarm-calibration", type=Path)
    parser.add_argument("--environment-centers", type=Path)
    parser.add_argument("--window-frames", required=True, type=int)
    parser.add_argument("--budget-percent", required=True, type=int)
    parser.add_argument("--audit-fraction", default=.2, type=float)
    parser.add_argument("--seed", default=20261116, type=int)
    parser.add_argument("--priority-policy", choices=PRIORITY_POLICIES, default="quality")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    scorer = RGBQualityScorer(args.model_run, args.model, args.device, adaptation_run=args.adaptation_run,
                              checkpoint=args.checkpoint, alarm_calibration=args.alarm_calibration,
                              environment_centers=args.environment_centers)
    counts = run_stream(sys.stdin, scorer, args.data_root, args.output_dir, args.window_frames,
                        args.budget_percent, args.audit_fraction, args.seed, args.priority_policy)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
