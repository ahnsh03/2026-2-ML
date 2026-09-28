#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `scripts/build_perception_dataset.py` (원작성: 안승현)
"""Build a versioned MORAI perception dataset from completed run caches."""

from __future__ import print_function

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "ros" / "src" / "data_collection" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from data_collection.perception_dataset import (  # noqa: E402
    DEFAULT_CACHE_NAME,
    DEFAULT_DATASET_VERSION,
    PerceptionDatasetError,
    build_dataset_version,
)


def default_data_root():
    configured = os.environ.get("ML_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT.parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create run-exclusive train/val/test manifests for MORAI perception"
    )
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--cache-name", default=DEFAULT_CACHE_NAME)
    parser.add_argument("--train-run", action="append", default=[])
    parser.add_argument("--val-run", action="append", default=[])
    parser.add_argument("--test-run", action="append", default=[])
    parser.add_argument(
        "--exclude-frame",
        action="append",
        default=[],
        metavar="RUN_ID:FRAME_ID",
        help="Exclude one complete three-camera capture frame; repeat as needed",
    )
    parser.add_argument(
        "--training-head",
        action="append",
        choices=("lane", "drivable", "road_marking", "stopline"),
        help="Active loss heads; default: lane and drivable",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-mask-audit",
        action="store_true",
        help="Check paths only; skip full mask shape/value verification",
    )
    return parser.parse_args()


def parse_excluded_frames(values):
    result = {}
    for value in values:
        try:
            run_id, frame_text = value.rsplit(":", 1)
            frame_id = int(frame_text)
        except (ValueError, AttributeError):
            raise ValueError(
                "--exclude-frame must use RUN_ID:FRAME_ID, got {!r}".format(value)
            )
        if not run_id or frame_id < 0:
            raise ValueError(
                "--exclude-frame must use a run id and non-negative frame id"
            )
        result.setdefault(run_id, set()).add(frame_id)
    return result


def main():
    args = parse_args()
    try:
        excluded_frames = parse_excluded_frames(args.exclude_frame)
        summary = build_dataset_version(
            data_root=args.data_root,
            train_runs=args.train_run,
            val_runs=args.val_run,
            test_runs=args.test_run,
            dataset_name=args.dataset_name,
            cache_name=args.cache_name,
            training_heads=args.training_head or ("lane", "drivable"),
            overwrite=args.overwrite,
            verify_masks=not args.skip_mask_audit,
            excluded_frames_by_run=excluded_frames,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    except (PerceptionDatasetError, OSError, ValueError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
