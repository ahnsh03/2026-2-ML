#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `scripts/sync_capture_mode_data.py` (원작성: 안승현)
"""Validate and copy completed MORAI Capture Mode runs from Windows to WSL."""

from __future__ import print_function

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "ros" / "src" / "data_collection" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from data_collection.capture_sync import (  # noqa: E402
    CaptureSyncError,
    load_successful_records,
    preflight_run,
    select_records,
    sync_run,
)


def default_data_root():
    configured = os.environ.get("ML_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT.parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline MORAI Capture Mode sync (source files are never deleted)"
    )
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--sensor-root",
        default="/mnt/c/MoraiLauncher_Win/SensorData",
        help="MORAI Windows SensorData path as seen from WSL",
    )
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--exclude-sequence",
        action="append",
        default=[],
        metavar="RUN_ID:SEQUENCE",
        help=(
            "Exclude one successful but incomplete capture from the synchronized "
            "dataset; repeat as needed"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all expected files without copying",
    )
    parser.add_argument(
        "--skip-png-check",
        action="store_true",
        help="Skip PNG signature and native-resolution validation",
    )
    return parser.parse_args()


def parse_exclusions(values):
    result = {}
    for value in values:
        try:
            run_id, sequence_text = value.rsplit(":", 1)
            sequence = int(sequence_text)
        except (AttributeError, ValueError):
            raise CaptureSyncError(
                "--exclude-sequence must be RUN_ID:SEQUENCE, got {!r}".format(value)
            )
        if not run_id or sequence < 0:
            raise CaptureSyncError(
                "--exclude-sequence must contain a non-empty run and non-negative sequence"
            )
        result.setdefault(run_id, set()).add(sequence)
    return result


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    capture_root = data_root / "capture_runs"
    dataset_root = data_root / "datasets"
    try:
        exclusions = parse_exclusions(args.exclude_sequence)
        unknown_runs = sorted(set(exclusions) - set(args.run_id))
        if unknown_runs:
            raise CaptureSyncError(
                "exclusions reference run(s) not passed with --run-id: {}".format(
                    unknown_runs
                )
            )
        for run_id in args.run_id:
            excluded_sequences = exclusions.get(run_id, set())
            if args.dry_run:
                manifest = capture_root / run_id / "capture_manifest.jsonl"
                source_records = load_successful_records(manifest)
                records, excluded_records = select_records(
                    source_records, excluded_sequences=excluded_sequences
                )
                _, total_bytes = preflight_run(
                    args.sensor_root,
                    records,
                    validate_png=not args.skip_png_check,
                )
                result = {
                    "run_id": run_id,
                    "status": "preflight_ok",
                    "frame_count": len(records),
                    "source_successful_frame_count": len(source_records),
                    "excluded_frame_count": len(excluded_records),
                    "excluded_sequences": [
                        int(record["sequence"]) for record in excluded_records
                    ],
                    "file_count": len(records) * 18,
                    "source_bytes": total_bytes,
                }
            else:
                result = sync_run(
                    sensor_root=args.sensor_root,
                    capture_root=capture_root,
                    dataset_root=dataset_root,
                    run_id=run_id,
                    jobs=args.jobs,
                    validate_png=not args.skip_png_check,
                    excluded_sequences=excluded_sequences,
                )
                result["status"] = "sync_complete"
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (CaptureSyncError, OSError, ValueError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
