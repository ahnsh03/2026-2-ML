#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `scripts/curate_perception_frames.py` (원작성: 안승현)
"""Build per-run selection manifests for stationary Capture Mode frames."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ros" / "src" / "data_collection" / "src"))

from data_collection.perception_curation import (  # noqa: E402
    DEFAULT_CURATION_NAME,
    PerceptionCurationError,
    curate_run,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--data-root", default=os.environ.get("ML_DATA_ROOT", str(REPO_ROOT.parent / "data"))
    )
    parser.add_argument("--output-name", default=DEFAULT_CURATION_NAME)
    parser.add_argument("--pose-threshold-m", type=float, default=0.01)
    parser.add_argument("--min-group-frames", type=int, default=3)
    parser.add_argument("--max-stationary-gap-frames", type=int, default=5)
    parser.add_argument("--focus-change-ratio", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.data_root).expanduser().resolve() / "datasets"
    try:
        for run_id in args.run_id:
            result = curate_run(
                dataset_root / run_id,
                output_name=args.output_name,
                pose_threshold_m=args.pose_threshold_m,
                min_group_frames=args.min_group_frames,
                max_stationary_gap_frames=args.max_stationary_gap_frames,
                focus_change_ratio=args.focus_change_ratio,
                overwrite=args.overwrite,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (PerceptionCurationError, OSError, ValueError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
