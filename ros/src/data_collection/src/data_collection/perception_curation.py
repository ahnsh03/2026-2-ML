# -*- coding: utf-8 -*-
# 이관: 인지 코드 레포 d0922bb `src/data_collection/src/data_collection/perception_curation.py` (원작성: 안승현)
"""Reproducible frame curation without mutating synchronized Silver data."""

from __future__ import print_function

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .mask_baker import OFFICIAL_PALETTE_RGB


CURATION_SCHEMA_VERSION = "perception-curation-1.0.0"
DEFAULT_CURATION_NAME = "curation_v2"
VIEWS = ("front", "left", "right")
FOCUS_CLASSES = (
    "vehicle", "sedan", "suv", "truck", "bus", "van", "wagon", "mpv",
    "pedestrian", "obstacle", "stopline",
)


class PerceptionCurationError(RuntimeError):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise PerceptionCurationError(
                    "invalid JSON at {}:{}: {}".format(path, line_number, exc)
                )
            if row.get("valid", True):
                rows.append(row)
    rows.sort(key=lambda item: int(item["frame_id"]))
    if not rows:
        raise PerceptionCurationError("manifest has no valid frames: {}".format(path))
    return rows


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _position(row):
    try:
        value = row["state_snapshot"]["ego"]["position"]
        return tuple(float(value[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return None


def _distance(left, right):
    if left is None or right is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def stationary_groups(rows, pose_threshold_m=0.01, min_group_frames=3):
    """Return index ranges for consecutive captures whose ego pose did not move."""
    if pose_threshold_m < 0 or min_group_frames < 2:
        raise PerceptionCurationError("invalid stationary-group thresholds")
    static_edges = []
    for left, right in zip(rows, rows[1:]):
        distance = _distance(_position(left), _position(right))
        consecutive = int(right["frame_id"]) == int(left["frame_id"]) + 1
        static_edges.append(
            bool(consecutive and distance is not None and distance < pose_threshold_m)
        )
    groups = []
    start = None
    for edge_index, is_static in enumerate(static_edges + [False]):
        if is_static and start is None:
            start = edge_index
        if not is_static and start is not None:
            end = edge_index
            if end - start + 1 >= min_group_frames:
                groups.append((start, end))
            start = None
    return groups


def _rgb_code(rgb):
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


FOCUS_CODES = np.asarray(
    [_rgb_code(OFFICIAL_PALETTE_RGB[name]) for name in FOCUS_CLASSES],
    dtype=np.uint32,
)


def _focus_signature(run_root, row, signature_hw):
    masks = []
    for view in VIEWS:
        try:
            relative = row["paths"]["semantic"][view]
        except (KeyError, TypeError):
            raise PerceptionCurationError(
                "missing semantic path frame={} view={}".format(row.get("frame_id"), view)
            )
        image = cv2.imread(str(run_root / relative), cv2.IMREAD_COLOR)
        if image is None:
            raise PerceptionCurationError("failed to read Semantic PNG: {}".format(relative))
        resized = cv2.resize(
            image, (int(signature_hw[1]), int(signature_hw[0])),
            interpolation=cv2.INTER_NEAREST,
        )
        codes = (
            (resized[:, :, 2].astype(np.uint32) << 16)
            | (resized[:, :, 1].astype(np.uint32) << 8)
            | resized[:, :, 0].astype(np.uint32)
        )
        masks.append(np.isin(codes, FOCUS_CODES))
    return np.stack(masks, axis=0)


def curate_run(
    run_root,
    output_name=DEFAULT_CURATION_NAME,
    pose_threshold_m=0.01,
    min_group_frames=3,
    max_stationary_gap_frames=5,
    focus_change_ratio=0.001,
    signature_hw=(45, 80),
    overwrite=False,
):
    """Select diverse frames and mark stationary near-duplicates as excluded."""
    run_root = Path(run_root).expanduser().resolve()
    source_manifest = run_root / "manifest.jsonl"
    rows = _load_jsonl(source_manifest)
    if max_stationary_gap_frames < 1 or not 0 <= focus_change_ratio <= 1:
        raise PerceptionCurationError("invalid curation thresholds")
    output_root = run_root / "derived" / str(output_name)
    temporary = output_root.with_name(".{}.{}.tmp".format(output_name, os.getpid()))
    if output_root.exists() and not overwrite:
        raise PerceptionCurationError("curation output exists: {}".format(output_root))
    if temporary.exists():
        shutil.rmtree(str(temporary))
    temporary.mkdir(parents=True)

    groups = stationary_groups(rows, pose_threshold_m, min_group_frames)
    group_by_index = {}
    for group_number, (start, end) in enumerate(groups, 1):
        for index in range(start, end + 1):
            group_by_index[index] = (group_number, start, end)

    decisions = []
    reason_counts = Counter()
    signature_cache = {}
    last_selected_by_group = {}
    try:
        for index, row in enumerate(rows):
            frame_id = int(row["frame_id"])
            group = group_by_index.get(index)
            pose_delta = None
            if index > 0:
                pose_delta = _distance(_position(rows[index - 1]), _position(row))
            selected = True
            reason = "moving_or_short_stop"
            scene_change = None
            group_id = None
            if group is not None:
                group_number, start, end = group
                group_id = "stationary_{:03d}".format(group_number)
                if index == start:
                    reason = "stationary_anchor_start"
                else:
                    previous_selected = last_selected_by_group[group_number]
                    for candidate in (previous_selected, index):
                        if candidate not in signature_cache:
                            signature_cache[candidate] = _focus_signature(
                                run_root, rows[candidate], signature_hw
                            )
                    scene_change = float(
                        np.not_equal(
                            signature_cache[previous_selected], signature_cache[index]
                        ).mean()
                    )
                    if scene_change >= focus_change_ratio:
                        reason = "stationary_focus_change"
                    elif index == end:
                        reason = "stationary_anchor_end"
                    elif index - previous_selected >= max_stationary_gap_frames:
                        reason = "stationary_periodic"
                    else:
                        selected = False
                        reason = "stationary_near_duplicate"
                if selected:
                    last_selected_by_group[group_number] = index
            reason_counts[reason] += 1
            decisions.append(
                {
                    "schema_version": CURATION_SCHEMA_VERSION,
                    "run_id": run_root.name,
                    "frame_id": frame_id,
                    "selected": selected,
                    "reason": reason,
                    "stationary_group": group_id,
                    "pose_delta_prev_m": pose_delta,
                    "focus_change_from_last_selected": scene_change,
                }
            )

        selected_count = sum(bool(row["selected"]) for row in decisions)
        policy = {
            "pose_threshold_m": pose_threshold_m,
            "min_group_frames": min_group_frames,
            "max_stationary_gap_frames": max_stationary_gap_frames,
            "focus_change_ratio": focus_change_ratio,
            "signature_hw": list(signature_hw),
            "focus_classes": list(FOCUS_CLASSES),
            "selection_unit": "capture frame; all three training views share one decision",
        }
        summary = {
            "schema_version": CURATION_SCHEMA_VERSION,
            "created_at_utc": _utc_now_iso(),
            "run_id": run_root.name,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _file_hash(source_manifest),
            "source_frame_count": len(rows),
            "selected_frame_count": selected_count,
            "excluded_frame_count": len(rows) - selected_count,
            "stationary_group_count": len(groups),
            "reason_counts": dict(sorted(reason_counts.items())),
            "policy": policy,
        }
        _write_jsonl(temporary / "manifest.jsonl", decisions)
        _write_json(temporary / "summary.json", summary)
        _write_json(
            temporary / "_SUCCESS",
            {
                "schema_version": CURATION_SCHEMA_VERSION,
                "completed_at_utc": _utc_now_iso(),
                "run_id": run_root.name,
                "selected_frame_count": selected_count,
                "source_manifest_sha256": summary["source_manifest_sha256"],
            },
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            shutil.rmtree(str(output_root))
        os.replace(str(temporary), str(output_root))
        summary["output_root"] = str(output_root)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise
