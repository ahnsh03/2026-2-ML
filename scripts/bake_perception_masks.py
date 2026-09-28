#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `scripts/bake_perception_masks.py` (원작성: 안승현)
"""Bake native perception targets and versioned TwinLite mask caches."""

from __future__ import print_function

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "ros" / "src" / "data_collection" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from data_collection.mask_baker import (  # noqa: E402
    DEFAULT_NATIVE_OUTPUT_NAME,
    DEFAULT_OUTPUT_NAME,
    MaskBakeError,
    bake_native_run,
    bake_run,
    build_model_cache,
)


def default_data_root():
    configured = os.environ.get("ML_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT.parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake MORAI native targets and/or TwinLite caches without modifying Silver data"
    )
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument(
        "--stage",
        choices=("native", "twinlite", "all"),
        default="all",
        help="native targets, TwinLite cache from completed native targets, or both",
    )
    parser.add_argument(
        "--native-output-name", default=DEFAULT_NATIVE_OUTPUT_NAME
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--policy",
        help=(
            "Frozen target-policy JSON; enables ignore/occupancy and optional "
            "categorical road-marking targets"
        ),
    )
    parser.add_argument(
        "--curation-name",
        help="Completed derived curation directory whose selected frames are baked",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate every source sample without writing derived data",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the selected derived output directory after a successful bake",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only the first N manifest frames (smoke/diagnostics only)",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        help="Process N frames evenly across the run (QA preview only)",
    )
    parser.add_argument(
        "--write-previews",
        action="store_true",
        help="Write visible RGB overlays under qa_overlays/<view>/",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.data_root).expanduser().resolve() / "datasets"
    try:
        for run_id in args.run_id:
            run_root = dataset_root / run_id
            if args.stage == "native":
                summary = bake_native_run(
                    run_root,
                    output_name=args.native_output_name,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    max_frames=args.max_frames,
                    sample_frames=args.sample_frames,
                    write_previews=args.write_previews,
                    policy_path=args.policy,
                    curation_name=args.curation_name,
                )
            elif args.stage == "twinlite":
                if args.max_frames is not None or args.sample_frames is not None:
                    raise MaskBakeError(
                        "TwinLite stage consumes the completed native manifest; "
                        "select frames while baking the native stage"
                    )
                summary = build_model_cache(
                    run_root,
                    native_output_name=args.native_output_name,
                    output_name=args.output_name,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    write_previews=args.write_previews,
                )
            else:
                summary = bake_run(
                    run_root,
                    native_output_name=args.native_output_name,
                    output_name=args.output_name,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    max_frames=args.max_frames,
                    sample_frames=args.sample_frames,
                    write_previews=args.write_previews,
                    policy_path=args.policy,
                    curation_name=args.curation_name,
                )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    except (MaskBakeError, OSError, ValueError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
