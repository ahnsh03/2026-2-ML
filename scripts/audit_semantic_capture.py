#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `scripts/audit_semantic_capture.py` (원작성: 안승현)
"""Audit exact MORAI Semantic colors in synchronized Capture Mode runs."""

from __future__ import print_function

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "ros" / "src" / "data_collection" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from data_collection.mask_baker import OFFICIAL_PALETTE_RGB  # noqa: E402


SCHEMA_VERSION = "morai-semantic-audit-1.1.0"
DEFAULT_VIEWS = ("front", "left", "right")
ROAD_MARKING_CLASSES = ("white_lane", "yellow_lane", "blue_lane", "stopline")
ACTOR_CLASSES = (
    "vehicle",
    "sedan",
    "suv",
    "truck",
    "bus",
    "van",
    "wagon",
    "mpv",
    "pedestrian",
    "obstacle",
)
FOCUS_CLASSES = ROAD_MARKING_CLASSES + ACTOR_CLASSES + ("standing_object",)


def _rgb_code(rgb):
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


CODE_TO_CLASS = {
    _rgb_code(rgb): name for name, rgb in OFFICIAL_PALETTE_RGB.items()
}
CLASS_TO_CODE = {name: code for code, name in CODE_TO_CLASS.items()}
ACTOR_CODES = np.asarray([CLASS_TO_CODE[name] for name in ACTOR_CLASSES], dtype=np.uint32)


class AuditError(RuntimeError):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise AuditError("invalid JSON at {}:{}: {}".format(path, line_number, exc))
            if row.get("valid", True):
                rows.append(row)
    if not rows:
        raise AuditError("manifest has no valid rows: {}".format(path))
    return rows


def _read_color(path, label):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise AuditError("invalid {} image: {}".format(label, path))
    return image[:, :, :3]


def _packed_rgb(image_bgr):
    return (
        (image_bgr[:, :, 2].astype(np.uint32) << 16)
        | (image_bgr[:, :, 1].astype(np.uint32) << 8)
        | image_bgr[:, :, 0].astype(np.uint32)
    )


def _audit_sample(task):
    run_root, run_id, row, view = task
    frame_id = int(row["frame_id"])
    try:
        semantic_relative = row["paths"]["semantic"][view]
        intensity_relative = row["paths"]["intensity"][view]
        instance_relative = row["paths"]["instance"][view]
    except (KeyError, TypeError):
        raise AuditError(
            "missing path in manifest run={} frame={} view={}".format(
                run_id, frame_id, view
            )
        )
    semantic_path = run_root / semantic_relative
    semantic = _read_color(semantic_path, "semantic")
    codes = _packed_rgb(semantic)
    unique, counts = np.unique(codes, return_counts=True)
    class_pixels = Counter()
    unknown_pixels = Counter()
    for code, count in zip(unique.tolist(), counts.tolist()):
        name = CODE_TO_CLASS.get(int(code))
        if name is None:
            unknown_pixels[str(int(code))] += int(count)
        else:
            class_pixels[name] += int(count)

    actor_mask = np.isin(codes, ACTOR_CODES)
    actor_pixels = int(actor_mask.sum())
    instance_nonzero_actor_pixels = 0
    actor_instance_color_count = 0
    if actor_pixels:
        instance_path = run_root / instance_relative
        instance = _read_color(instance_path, "instance")
        if instance.shape[:2] != semantic.shape[:2]:
            raise AuditError(
                "Semantic/Instance shape mismatch run={} frame={} view={}".format(
                    run_id, frame_id, view
                )
            )
        instance_actor = instance[actor_mask]
        instance_nonzero_actor_pixels = int(np.any(instance_actor != 0, axis=1).sum())
        actor_instance_color_count = int(
            np.unique(instance_actor.reshape(-1, 3), axis=0).shape[0]
        )

    return {
        "run_id": run_id,
        "frame_id": frame_id,
        "view": view,
        "hw": [int(semantic.shape[0]), int(semantic.shape[1])],
        "pixels": int(semantic.shape[0] * semantic.shape[1]),
        "class_pixels": dict(class_pixels),
        "unknown_pixels_by_code": dict(unknown_pixels),
        "actor_pixels": actor_pixels,
        "instance_nonzero_actor_pixels": instance_nonzero_actor_pixels,
        "actor_instance_color_count": actor_instance_color_count,
        "paths": {
            "intensity": intensity_relative,
            "semantic": semantic_relative,
            "instance": instance_relative,
        },
    }


def _empty_stats():
    return {
        "samples": 0,
        "pixels": 0,
        "class_pixels": Counter(),
        "class_positive_samples": Counter(),
        "unknown_pixels_by_code": Counter(),
        "actor_pixels": 0,
        "actor_positive_samples": 0,
        "instance_nonzero_actor_pixels": 0,
        "actor_instance_color_count_max": 0,
    }


def _update(stats, sample):
    stats["samples"] += 1
    stats["pixels"] += sample["pixels"]
    stats["class_pixels"].update(sample["class_pixels"])
    stats["class_positive_samples"].update(
        name for name, count in sample["class_pixels"].items() if count > 0
    )
    stats["unknown_pixels_by_code"].update(sample["unknown_pixels_by_code"])
    stats["actor_pixels"] += sample["actor_pixels"]
    stats["actor_positive_samples"] += int(sample["actor_pixels"] > 0)
    stats["instance_nonzero_actor_pixels"] += sample["instance_nonzero_actor_pixels"]
    stats["actor_instance_color_count_max"] = max(
        stats["actor_instance_color_count_max"], sample["actor_instance_color_count"]
    )


def _finalize(stats):
    actor_pixels = int(stats["actor_pixels"])
    return {
        "samples": int(stats["samples"]),
        "pixels": int(stats["pixels"]),
        "class_pixels": dict(sorted(stats["class_pixels"].items())),
        "class_positive_samples": dict(
            sorted(stats["class_positive_samples"].items())
        ),
        "unknown_pixels_by_code": dict(sorted(stats["unknown_pixels_by_code"].items())),
        "actor_pixels": actor_pixels,
        "actor_positive_samples": int(stats["actor_positive_samples"]),
        "instance_nonzero_actor_pixels": int(stats["instance_nonzero_actor_pixels"]),
        "instance_nonzero_actor_ratio": (
            float(stats["instance_nonzero_actor_pixels"]) / actor_pixels
            if actor_pixels
            else None
        ),
        "actor_instance_color_count_max": int(stats["actor_instance_color_count_max"]),
    }


def _road_marking_summary(stats):
    class_pixels = stats["class_pixels"]
    positive_samples = stats["class_positive_samples"]
    return {
        name: {
            "pixels": int(class_pixels.get(name, 0)),
            "positive_samples": int(positive_samples.get(name, 0)),
        }
        for name in ROAD_MARKING_CLASSES
    }


def _render_preview(data_root, sample, selected_class, output_path):
    run_root = data_root / "datasets" / sample["run_id"]
    intensity = _read_color(run_root / sample["paths"]["intensity"], "intensity")
    semantic = _read_color(run_root / sample["paths"]["semantic"], "semantic")
    if intensity.shape[:2] != semantic.shape[:2]:
        raise AuditError("Intensity/Semantic preview shape mismatch")
    code = CLASS_TO_CODE[selected_class]
    mask = _packed_rgb(semantic) == code
    overlay = intensity.copy()
    if np.any(mask):
        color = np.asarray((0, 0, 255), dtype=np.float32)
        overlay[mask] = np.clip(
            overlay[mask].astype(np.float32) * 0.25 + color * 0.75, 0, 255
        ).astype(np.uint8)
    semantic_panel = semantic.copy()
    title = "{} {} frame={} {} pixels={}".format(
        sample["run_id"], sample["view"], sample["frame_id"], selected_class,
        sample["class_pixels"].get(selected_class, 0),
    )
    for panel in (overlay, semantic_panel):
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 34), (0, 0, 0), -1)
        cv2.putText(
            panel, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    preview = np.concatenate((overlay, semantic_panel), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise AuditError("failed to write preview: {}".format(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--data-root", default=os.environ.get("ML_DATA_ROOT", str(REPO_ROOT.parent / "data"))
    )
    parser.add_argument("--view", action="append", choices=DEFAULT_VIEWS)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument(
        "--require-zero-class",
        action="append",
        choices=tuple(sorted(OFFICIAL_PALETTE_RGB)),
        help="Write the report, then fail if this Semantic class has any pixels",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    views = tuple(args.view or DEFAULT_VIEWS)
    tasks = []
    run_frame_counts = {}
    for run_id in args.run_id:
        run_root = data_root / "datasets" / run_id
        rows = _load_jsonl(run_root / "manifest.jsonl")
        run_frame_counts[run_id] = len(rows)
        tasks.extend((run_root, run_id, row, view) for row in rows for view in views)

    total = _empty_stats()
    per_run = {run_id: _empty_stats() for run_id in args.run_id}
    per_view = {view: _empty_stats() for view in views}
    samples = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        for index, sample in enumerate(executor.map(_audit_sample, tasks), 1):
            samples.append(sample)
            _update(total, sample)
            _update(per_run[sample["run_id"]], sample)
            _update(per_view[sample["view"]], sample)
            if index % 1000 == 0 or index == len(tasks):
                print("[semantic-audit] {}/{} samples".format(index, len(tasks)), flush=True)

    top_samples = {}
    preview_root = Path(args.output).expanduser().resolve().with_suffix("")
    preview_root = preview_root.parent / (preview_root.name + "_qa")
    for class_name in FOCUS_CLASSES:
        selected = sorted(
            (
                sample for sample in samples
                if sample["class_pixels"].get(class_name, 0) > 0
            ),
            key=lambda item: item["class_pixels"][class_name],
            reverse=True,
        )[: max(0, args.preview_count)]
        top_samples[class_name] = [
            {
                "run_id": sample["run_id"],
                "frame_id": sample["frame_id"],
                "view": sample["view"],
                "pixels": sample["class_pixels"][class_name],
                "paths": sample["paths"],
            }
            for sample in selected
        ]
        for sample in selected:
            filename = "{}_{}_{}.png".format(
                sample["run_id"], sample["view"], str(sample["frame_id"]).zfill(6)
            )
            _render_preview(data_root, sample, class_name, preview_root / class_name / filename)

    total_final = _finalize(total)
    per_view_final = {view: _finalize(stats) for view, stats in per_view.items()}
    required_zero_classes = tuple(args.require_zero_class or ())
    zero_class_assertions = {
        name: {
            "pixels": int(total_final["class_pixels"].get(name, 0)),
            "passed": int(total_final["class_pixels"].get(name, 0)) == 0,
        }
        for name in required_zero_classes
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "data_root": str(data_root),
        "runs": list(args.run_id),
        "run_frame_counts": run_frame_counts,
        "views": list(views),
        "official_palette_rgb": {
            name: list(rgb) for name, rgb in OFFICIAL_PALETTE_RGB.items()
        },
        "actor_classes": list(ACTOR_CLASSES),
        "road_marking_classes": list(ROAD_MARKING_CLASSES),
        "road_marking_summary": {
            "total": _road_marking_summary(total_final),
            "per_view": {
                view: _road_marking_summary(stats)
                for view, stats in per_view_final.items()
            },
        },
        "assertions": {
            "zero_pixel_classes": zero_class_assertions,
            "passed": all(
                result["passed"] for result in zero_class_assertions.values()
            ),
        },
        "total": total_final,
        "per_run": {run_id: _finalize(stats) for run_id, stats in per_run.items()},
        "per_view": per_view_final,
        "top_samples": top_samples,
        "qa_preview_root": str(preview_root),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(output))
    print(
        json.dumps(
            {
                "output": str(output),
                "road_marking_summary": report["road_marking_summary"],
                "assertions": report["assertions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["assertions"]["passed"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditError, OSError, ValueError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        sys.exit(1)
