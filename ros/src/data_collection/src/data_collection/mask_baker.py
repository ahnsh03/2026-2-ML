# -*- coding: utf-8 -*-
# 이관: 인지 코드 레포 d0922bb `src/data_collection/src/data_collection/mask_baker.py` (원작성: 안승현)
"""Build model-independent MORAI masks and model-specific mask caches.

Synchronized RGB/Semantic PNG files are immutable Silver data. Native masks
are canonical Gold targets, while resized/letterboxed masks are disposable
model caches. The module deliberately has no ROS dependency.
"""

from __future__ import print_function

import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


NATIVE_SCHEMA_VERSION = "perception-targets-native-1.0.0"
MODEL_CACHE_SCHEMA_VERSION = "twinlite-mask-cache-1.0.0"
NATIVE_V2_SCHEMA_VERSION = "perception-targets-native-2.0.0"
MODEL_CACHE_V2_SCHEMA_VERSION = "twinlite-mask-cache-2.0.0"
NATIVE_V3_SCHEMA_VERSION = "perception-targets-native-3.0.0"
MODEL_CACHE_V3_SCHEMA_VERSION = "twinlite-mask-cache-3.0.0"
BAKE_SCHEMA_VERSION = MODEL_CACHE_SCHEMA_VERSION  # Backward-compatible import.
DEFAULT_NATIVE_OUTPUT_NAME = "perception_targets_native_v1"
DEFAULT_OUTPUT_NAME = "twinlite_384x640_v2"
TARGET_HW = (384, 640)
TWINLITE_VIEWS = ("front", "left", "right")

# Complete MORAI 26.R1 Semantic camera RGB palette. OpenCV loads PNG as
# BGR(A), so exact comparisons reverse these tuples at the boundary.
OFFICIAL_PALETTE_RGB = {
    "vehicle": (255, 2, 2),
    "sedan": (255, 60, 60),
    "suv": (255, 75, 75),
    "truck": (255, 90, 90),
    "bus": (255, 105, 105),
    "van": (255, 120, 120),
    "wagon": (255, 135, 135),
    "mpv": (255, 150, 150),
    "pedestrian": (98, 2, 255),
    "obstacle": (236, 255, 2),
    "asphalt": (127, 127, 127),
    "white_lane": (255, 255, 255),
    "yellow_lane": (255, 255, 0),
    "blue_lane": (0, 178, 255),
    "crosswalk": (76, 255, 76),
    "stopline": (255, 0, 0),
    "road_sign": (204, 127, 51),
    "traffic_light": (255, 74, 240),
    "traffic_sign": (99, 48, 250),
    "sidewalk": (255, 102, 30),
    "standing_object": (113, 178, 37),
    "building": (153, 255, 51),
    "road_edge": (178, 178, 178),
    "sky": (0, 255, 255),
    "etc": (23, 2, 6),
    "ego_vehicle": (0, 0, 0),
}

PALETTE_RGB = {
    name: OFFICIAL_PALETTE_RGB[name]
    for name in (
        "asphalt",
        "white_lane",
        "yellow_lane",
        "blue_lane",
        "crosswalk",
        "stopline",
        "road_sign",
    )
}
LANE_CLASSES = ("white_lane", "yellow_lane", "blue_lane")
DRIVABLE_CLASSES = (
    "asphalt",
    "white_lane",
    "yellow_lane",
    "blue_lane",
    "crosswalk",
    "stopline",
    "road_sign",
)
MASK_NAMES = ("lane", "drivable", "stopline")
V2_MASK_NAMES = (
    "lane",
    "drivable",
    "stopline",
    "drivable_ignore",
    "actor_occupancy_gt",
)
ROAD_MARKING_CLASS_VALUES = {
    "background": 0,
    "white_lane": 1,
    "yellow_lane": 2,
    "stopline": 3,
}
V3_MASK_NAMES = V2_MASK_NAMES + ("road_marking",)


class MaskBakeError(RuntimeError):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _stable_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_color_png(path, label):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MaskBakeError("failed to read {} PNG: {}".format(label, path))
    if image.ndim != 3 or image.shape[2] < 3:
        raise MaskBakeError(
            "{} PNG must have at least 3 channels: {} shape={}".format(
                label, path, image.shape
            )
        )
    return image[:, :, :3]


def _rgb_code(rgb):
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


OFFICIAL_COLOR_CODES = {
    _rgb_code(rgb): name for name, rgb in OFFICIAL_PALETTE_RGB.items()
}
OFFICIAL_CODE_ARRAY = np.asarray(sorted(OFFICIAL_COLOR_CODES), dtype=np.uint32)


def _semantic_codes(semantic_bgr):
    return (
        (semantic_bgr[:, :, 2].astype(np.uint32) << 16)
        | (semantic_bgr[:, :, 1].astype(np.uint32) << 8)
        | semantic_bgr[:, :, 0].astype(np.uint32)
    )


def semantic_unknown_colors(semantic_bgr):
    """Return non-MORAI RGB colors and pixel counts from one Semantic image."""
    if semantic_bgr.ndim != 3 or semantic_bgr.shape[2] != 3:
        raise MaskBakeError("semantic image must be HxWx3 BGR")
    codes = _semantic_codes(semantic_bgr)
    unknown_mask = ~np.isin(codes, OFFICIAL_CODE_ARRAY)
    if not np.any(unknown_mask):
        return []
    unique, counts = np.unique(codes[unknown_mask], return_counts=True)
    unknown = []
    for code, count in zip(unique.tolist(), counts.tolist()):
        if int(code) not in OFFICIAL_COLOR_CODES:
            unknown.append(
                {
                    "rgb": [
                        (int(code) >> 16) & 255,
                        (int(code) >> 8) & 255,
                        int(code) & 255,
                    ],
                    "pixels": int(count),
                }
            )
    return unknown


def semantic_class_masks(semantic_bgr):
    """Return exact-palette uint8 masks at native resolution."""
    if semantic_bgr.ndim != 3 or semantic_bgr.shape[2] != 3:
        raise MaskBakeError("semantic image must be HxWx3 BGR")

    class_masks = {}
    for name, rgb in PALETTE_RGB.items():
        bgr = np.asarray(rgb[::-1], dtype=np.uint8)
        class_masks[name] = np.all(semantic_bgr == bgr, axis=2)

    lane = np.zeros(semantic_bgr.shape[:2], dtype=bool)
    for name in LANE_CLASSES:
        lane |= class_masks[name]

    drivable = np.zeros(semantic_bgr.shape[:2], dtype=bool)
    for name in DRIVABLE_CLASSES:
        drivable |= class_masks[name]

    return {
        "lane": lane.astype(np.uint8),
        "drivable": drivable.astype(np.uint8),
        "stopline": class_masks["stopline"].astype(np.uint8),
    }


def semantic_policy_masks(semantic_bgr, policy):
    """Build masks from a frozen, explicit class policy."""
    if semantic_bgr.ndim != 3 or semantic_bgr.shape[2] != 3:
        raise MaskBakeError("semantic image must be HxWx3 BGR")
    targets = policy.get("targets") or {}
    required = ("lane", "drivable", "stopline", "actor_occupancy_gt")
    missing = [name for name in required if name not in targets]
    if missing:
        raise MaskBakeError("target policy is missing sections: {}".format(missing))

    codes = _semantic_codes(semantic_bgr)
    cache = {}

    def union(class_names):
        output = np.zeros(semantic_bgr.shape[:2], dtype=bool)
        for name in class_names:
            if name not in OFFICIAL_PALETTE_RGB:
                raise MaskBakeError("policy references unknown class: {}".format(name))
            if name not in cache:
                cache[name] = _rgb_code(OFFICIAL_PALETTE_RGB[name])
        if class_names:
            output = np.isin(
                codes,
                np.asarray([cache[name] for name in class_names], dtype=np.uint32),
            )
        return output.astype(np.uint8)

    def categorical(class_values):
        expected = dict(ROAD_MARKING_CLASS_VALUES)
        normalized = {str(name): int(value) for name, value in class_values.items()}
        if normalized != expected:
            raise MaskBakeError(
                "road_marking class_values must be {}: {}".format(
                    expected, normalized
                )
            )
        output = np.zeros(semantic_bgr.shape[:2], dtype=np.uint8)
        for name, value in normalized.items():
            if name == "background":
                continue
            output[codes == _rgb_code(OFFICIAL_PALETTE_RGB[name])] = value
        return output

    actor_classes = targets["actor_occupancy_gt"].get("positive_classes", [])
    ignore_classes = targets["drivable"].get("ignore_classes", [])
    actor = union(actor_classes)
    ignore = (
        actor.copy()
        if list(ignore_classes) == list(actor_classes)
        else union(ignore_classes)
    )
    masks = {
        "lane": union(targets["lane"].get("positive_classes", [])),
        "drivable": union(targets["drivable"].get("positive_classes", [])),
        "stopline": union(targets["stopline"].get("positive_classes", [])),
        "drivable_ignore": ignore,
        "actor_occupancy_gt": actor,
    }
    if "road_marking" in targets:
        masks["road_marking"] = categorical(
            targets["road_marking"].get("class_values", {})
        )
    return masks


def _validate_zero_pixel_constraints(semantic_bgr, policy, frame_id, view):
    constraints = policy.get("audit_constraints") or {}
    class_names = constraints.get("zero_pixel_classes") or []
    if not class_names:
        return
    codes = _semantic_codes(semantic_bgr)
    violations = {}
    for name in class_names:
        if name not in OFFICIAL_PALETTE_RGB:
            raise MaskBakeError("policy references unknown audit class: {}".format(name))
        count = int(np.count_nonzero(codes == _rgb_code(OFFICIAL_PALETTE_RGB[name])))
        if count:
            violations[name] = count
    if violations:
        raise MaskBakeError(
            "zero-pixel policy violation frame={} view={}: {}".format(
                frame_id, view, violations
            )
        )


def _task_valid_for_view(policy, target_name, view, default=True):
    if not policy:
        return bool(default)
    target = (policy.get("targets") or {}).get(target_name) or {}
    valid_views = target.get("valid_views")
    if valid_views is not None:
        unknown = set(valid_views) - set(TWINLITE_VIEWS)
        if unknown:
            raise MaskBakeError(
                "{} valid_views contains unsupported views: {}".format(
                    target_name, sorted(unknown)
                )
            )
        return view in valid_views
    valid_policy = str(target.get("valid_policy") or "")
    if "view_is_front" in valid_policy:
        return view == "front"
    return bool(default)


def letterbox_mask(mask, target_hw=TARGET_HW):
    """Letterbox a discrete mask and return it with reversible geometry."""
    if mask.ndim != 2:
        raise MaskBakeError("mask must be HxW, got {}".format(mask.shape))
    source_h, source_w = int(mask.shape[0]), int(mask.shape[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if source_h <= 0 or source_w <= 0 or target_h <= 0 or target_w <= 0:
        raise MaskBakeError("image and target dimensions must be positive")

    scale = min(float(target_w) / source_w, float(target_h) / source_h)
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    right = target_w - resized_w - left
    bottom = target_h - resized_h - top

    resized = cv2.resize(
        mask.astype(np.uint8),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    output[top : top + resized_h, left : left + resized_w] = resized
    geometry = {
        "original_hw": [source_h, source_w],
        "target_hw": [target_h, target_w],
        "scale": scale,
        "scale_xy": [float(resized_w) / source_w, float(resized_h) / source_h],
        "resized_hw": [resized_h, resized_w],
        "pad_ltrb": [left, top, right, bottom],
        "valid_roi_xyxy": [left, top, left + resized_w, top + resized_h],
    }
    return output, geometry


def letterbox_image(image_bgr, geometry, padding_value=114):
    """Apply recorded geometry to a BGR image for QA or model loading."""
    resized_h, resized_w = geometry["resized_hw"]
    target_h, target_w = geometry["target_hw"]
    left, top, _, _ = geometry["pad_ltrb"]
    resized = cv2.resize(
        image_bgr,
        (int(resized_w), int(resized_h)),
        interpolation=cv2.INTER_AREA,
    )
    output = np.full(
        (int(target_h), int(target_w), 3), int(padding_value), dtype=np.uint8
    )
    output[top : top + resized_h, left : left + resized_w] = resized
    return output


def _write_png_atomic(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise MaskBakeError("failed to write PNG: {}".format(temporary))
    os.replace(str(temporary), str(path))


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise MaskBakeError(
                    "invalid JSON at {}:{}: {}".format(path, line_number, exc)
                )
    return rows


def _load_manifest(path):
    rows = []
    frame_ids = set()
    for row in _load_jsonl(path):
        if not row.get("valid", True):
            continue
        frame_id = int(row["frame_id"])
        if frame_id in frame_ids:
            raise MaskBakeError("duplicate frame_id: {}".format(frame_id))
        frame_ids.add(frame_id)
        rows.append(row)
    rows.sort(key=lambda item: int(item["frame_id"]))
    if not rows:
        raise MaskBakeError("manifest has no valid frames: {}".format(path))
    return rows


def _select_rows(rows, max_frames=None, sample_frames=None):
    if max_frames is not None and sample_frames is not None:
        raise MaskBakeError("max_frames and sample_frames are mutually exclusive")
    if max_frames is not None:
        count = int(max_frames)
        if count <= 0:
            raise MaskBakeError("max_frames must be positive")
        return rows[:count]
    if sample_frames is not None:
        count = int(sample_frames)
        if count <= 0:
            raise MaskBakeError("sample_frames must be positive")
        if count >= len(rows):
            return rows
        indices = np.linspace(0, len(rows) - 1, count, dtype=int)
        return [rows[int(index)] for index in indices]
    return rows


def _source_path(run_root, row, modality, view):
    try:
        relative = row["paths"][modality][view]
    except (KeyError, TypeError):
        raise MaskBakeError(
            "missing {} path for frame={} view={}".format(
                modality, row.get("frame_id"), view
            )
        )
    path = run_root / relative
    if not path.is_file():
        raise MaskBakeError("source file not found: {}".format(path))
    return path, str(relative)


def _validate_output_name(name):
    name = str(name)
    if not name or name in (".", "..") or Path(name).name != name:
        raise MaskBakeError("output name must be one safe directory name")
    return name


def _prepare_run(run_root, max_frames=None, sample_frames=None, curation_name=None):
    run_root = Path(run_root).resolve()
    manifest_path = run_root / "manifest.jsonl"
    dataset_path = run_root / "dataset.json"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise MaskBakeError(
            "run is missing manifest.jsonl or dataset.json: {}".format(run_root)
        )
    with dataset_path.open("r", encoding="utf-8") as stream:
        dataset_meta = json.load(stream)
    all_rows = _load_manifest(manifest_path)
    rows = all_rows
    curation_info = None
    if curation_name:
        curation_name = _validate_output_name(curation_name)
        curation_root = run_root / "derived" / curation_name
        curation_manifest = curation_root / "manifest.jsonl"
        curation_summary = curation_root / "summary.json"
        if not (curation_root / "_SUCCESS").is_file() or not curation_manifest.is_file():
            raise MaskBakeError("completed curation not found: {}".format(curation_root))
        decisions = _load_jsonl(curation_manifest)
        by_frame = {int(row["frame_id"]): row for row in decisions}
        source_ids = {int(row["frame_id"]) for row in all_rows}
        if set(by_frame) != source_ids or len(by_frame) != len(decisions):
            raise MaskBakeError("curation frame IDs do not match source manifest")
        selected_ids = {
            frame_id for frame_id, decision in by_frame.items()
            if bool(decision.get("selected"))
        }
        if not selected_ids:
            raise MaskBakeError("curation selected no frames")
        rows = [row for row in all_rows if int(row["frame_id"]) in selected_ids]
        curation_info = {
            "name": curation_name,
            "manifest": str(curation_manifest),
            "manifest_sha256": _file_hash(curation_manifest),
            "summary_sha256": _file_hash(curation_summary)
            if curation_summary.is_file()
            else None,
            "selected_frame_count": len(rows),
            "excluded_frame_count": len(all_rows) - len(rows),
        }
    rows = _select_rows(rows, max_frames=max_frames, sample_frames=sample_frames)
    return run_root, manifest_path, dataset_meta, all_rows, rows, curation_info


def _temporary_output(run_root, output_name, overwrite, dry_run):
    output_name = _validate_output_name(output_name)
    output_root = run_root / "derived" / output_name
    temporary_root = output_root.with_name(".{}.{}.tmp".format(output_name, os.getpid()))
    if not dry_run:
        if output_root.exists() and not overwrite:
            raise MaskBakeError(
                "output already exists (use --overwrite explicitly): {}".format(
                    output_root
                )
            )
        if temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        temporary_root.mkdir(parents=True)
    return output_root, temporary_root


def _publish_output(output_root, temporary_root, overwrite):
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        if not overwrite:
            raise MaskBakeError("refusing to replace output: {}".format(output_root))
        shutil.rmtree(str(output_root))
    os.replace(str(temporary_root), str(output_root))


def _new_counters(mask_names=MASK_NAMES):
    return {
        view: Counter(
            {
                "samples": 0,
                "valid_pixels": 0,
                **{
                    key: 0
                    for name in mask_names
                    for key in (name + "_pixels", name + "_positive_samples")
                },
            }
        )
        for view in TWINLITE_VIEWS
    }


def _update_counters(counters, view, masks, valid_pixels, mask_names=MASK_NAMES):
    stats = counters[view]
    stats["samples"] += 1
    stats["valid_pixels"] += int(valid_pixels)
    for name in mask_names:
        if name == "road_marking":
            pixel_count = int(np.count_nonzero(masks[name]))
            for class_name, class_value in ROAD_MARKING_CLASS_VALUES.items():
                class_count = int(np.count_nonzero(masks[name] == class_value))
                stats["road_marking_{}_pixels".format(class_name)] += class_count
                stats[
                    "road_marking_{}_positive_samples".format(class_name)
                ] += int(class_count > 0)
        else:
            pixel_count = int(masks[name].sum())
        stats[name + "_pixels"] += pixel_count
        stats[name + "_positive_samples"] += int(pixel_count > 0)


def _summarize_counters(counters, mask_names=MASK_NAMES):
    result = {}
    for view, stats in counters.items():
        view_summary = dict(stats)
        valid_pixels = max(1, int(stats["valid_pixels"]))
        for name in mask_names:
            view_summary[name + "_pixel_ratio"] = float(
                stats[name + "_pixels"]
            ) / valid_pixels
        result[view] = view_summary
    return result


def render_mask_overlay(image_bgr, masks, title):
    """Render a visible QA overlay; canonical masks themselves stay {0,1}."""
    output = image_bgr[:, :, :3].copy()
    layers = [(masks["drivable"].astype(bool), (40, 180, 40), 0.38)]
    if "road_marking" in masks:
        marking_colors = {
            ROAD_MARKING_CLASS_VALUES["white_lane"]: (255, 255, 255),
            ROAD_MARKING_CLASS_VALUES["yellow_lane"]: (0, 230, 255),
            ROAD_MARKING_CLASS_VALUES["stopline"]: (0, 0, 255),
        }
        layers.extend(
            (masks["road_marking"] == value, color, 0.90)
            for value, color in marking_colors.items()
        )
        legend = "green=drivable white/yellow=lane red=stopline"
    else:
        layers.extend(
            (
                (masks["lane"].astype(bool), (0, 230, 255), 0.82),
                (masks["stopline"].astype(bool), (0, 0, 255), 0.90),
            )
        )
        legend = "green=drivable yellow=lane red=stopline"
    for selected, color, alpha in layers:
        if np.any(selected):
            color = np.asarray(color, dtype=np.float32)
            blended = (
                output[selected].astype(np.float32) * (1.0 - alpha) + color * alpha
            )
            output[selected] = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.rectangle(
        output, (0, 0), (min(output.shape[1] - 1, 760), 34), (0, 0, 0), -1
    )
    cv2.putText(
        output,
        title + " | " + legend,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def bake_native_run(
    run_root,
    output_name=DEFAULT_NATIVE_OUTPUT_NAME,
    overwrite=False,
    dry_run=False,
    max_frames=None,
    sample_frames=None,
    write_previews=False,
    policy_path=None,
    curation_name=None,
):
    """Create model-independent masks at each camera's native resolution."""
    run_root, manifest_path, dataset_meta, all_rows, rows, curation_info = _prepare_run(
        run_root, max_frames=max_frames, sample_frames=sample_frames,
        curation_name=curation_name,
    )
    output_root, temporary_root = _temporary_output(
        run_root, output_name, overwrite, dry_run
    )
    run_id = str(dataset_meta.get("run_id") or run_root.name)
    if policy_path:
        policy_path = Path(policy_path).expanduser().resolve()
        with policy_path.open("r", encoding="utf-8") as stream:
            class_policy = json.load(stream)
        if class_policy.get("status") not in ("frozen", "frozen_not_baked"):
            raise MaskBakeError("target policy is not frozen: {}".format(policy_path))
        if "road_marking" in (class_policy.get("targets") or {}):
            mask_names = V3_MASK_NAMES
            native_schema_version = NATIVE_V3_SCHEMA_VERSION
        else:
            mask_names = V2_MASK_NAMES
            native_schema_version = NATIVE_V2_SCHEMA_VERSION
        policy_sha256 = _file_hash(policy_path)
    else:
        class_policy = {
            "official_palette_rgb": {
                name: list(rgb) for name, rgb in OFFICIAL_PALETTE_RGB.items()
            },
            "lane_classes": list(LANE_CLASSES),
            "drivable_classes": list(DRIVABLE_CLASSES),
            "stopline_classes": ["stopline"],
            "mask_values": [0, 1],
        }
        mask_names = MASK_NAMES
        native_schema_version = NATIVE_SCHEMA_VERSION
        policy_sha256 = None
    summary = {
        "schema_version": native_schema_version,
        "run_id": run_id,
        "source_schema_version": dataset_meta.get("schema_version"),
        "source_manifest_sha256": _file_hash(manifest_path),
        "mapping_sha256": _stable_hash(class_policy),
        "policy_id": class_policy.get("policy_id"),
        "policy_sha256": policy_sha256,
        "curation": curation_info,
        "output_name": output_name,
        "dry_run": bool(dry_run),
        "complete_source_run": len(rows) == len(all_rows),
        "frame_count": len(rows),
        "sample_count": len(rows) * len(TWINLITE_VIEWS),
        "selected_frame_ids": [int(row["frame_id"]) for row in rows],
        "class_policy": class_policy,
        "road_marking_class_values": (
            dict(ROAD_MARKING_CLASS_VALUES)
            if "road_marking" in mask_names
            else None
        ),
        "unknown_semantic_pixels": 0,
        "views": {},
    }
    counters = _new_counters(mask_names)
    output_rows = []
    try:
        for row in rows:
            frame_id = int(row["frame_id"])
            stem = "{:06d}.png".format(frame_id)
            for view in TWINLITE_VIEWS:
                rgb_path, rgb_relative = _source_path(run_root, row, "intensity", view)
                semantic_path, semantic_relative = _source_path(
                    run_root, row, "semantic", view
                )
                rgb = _read_color_png(rgb_path, "intensity")
                semantic = _read_color_png(semantic_path, "semantic")
                if rgb.shape[:2] != semantic.shape[:2]:
                    raise MaskBakeError(
                        "RGB/Semantic shape mismatch frame={} view={} rgb={} semantic={}".format(
                            frame_id, view, rgb.shape[:2], semantic.shape[:2]
                        )
                    )
                unknown = semantic_unknown_colors(semantic)
                if unknown:
                    raise MaskBakeError(
                        "unknown Semantic RGB frame={} view={}: {}".format(
                            frame_id, view, unknown[:10]
                        )
                    )
                _validate_zero_pixel_constraints(
                    semantic, class_policy, frame_id, view
                )
                masks = (
                    semantic_policy_masks(semantic, class_policy)
                    if policy_path
                    else semantic_class_masks(semantic)
                )
                _update_counters(
                    counters, view, masks, masks["lane"].size, mask_names
                )
                output_paths = {
                    name: "{}_masks/{}/{}".format(name, view, stem)
                    for name in mask_names
                }
                if not dry_run:
                    for name, relative in output_paths.items():
                        _write_png_atomic(temporary_root / relative, masks[name])
                    if write_previews:
                        preview = render_mask_overlay(
                            rgb,
                            masks,
                            "{} frame={} view={} native".format(run_id, frame_id, view),
                        )
                        _write_png_atomic(
                            temporary_root / "qa_overlays" / view / stem, preview
                        )
                output_rows.append(
                    {
                        "schema_version": native_schema_version,
                        "run_id": run_id,
                        "frame_id": frame_id,
                        "view": view,
                        "source": {
                            "intensity": rgb_relative,
                            "semantic": semantic_relative,
                        },
                        "outputs": output_paths,
                        "native_hw": [int(semantic.shape[0]), int(semantic.shape[1])],
                        **(
                            {
                                "target_encodings": {
                                    "road_marking": {
                                        "type": "categorical_uint8",
                                        "class_values": dict(
                                            ROAD_MARKING_CLASS_VALUES
                                        ),
                                    }
                                }
                            }
                            if "road_marking" in masks
                            else {}
                        ),
                        "task_valid": {
                            "drivable": _task_valid_for_view(
                                class_policy, "drivable", view
                            ),
                            "lane": _task_valid_for_view(
                                class_policy, "lane", view
                            ),
                            "stopline": _task_valid_for_view(
                                class_policy,
                                "stopline",
                                view,
                                default=view == "front",
                            ),
                            **(
                                {
                                    "road_marking": _task_valid_for_view(
                                        class_policy, "road_marking", view
                                    )
                                }
                                if "road_marking" in masks
                                else {}
                            ),
                        },
                    }
                )
        summary["views"] = _summarize_counters(counters, mask_names)
        if not dry_run:
            _write_jsonl(temporary_root / "manifest.jsonl", output_rows)
            _write_json(temporary_root / "qa_summary.json", summary)
            _write_json(
                temporary_root / "_SUCCESS",
                {
                    "schema_version": native_schema_version,
                    "completed_at_utc": _utc_now_iso(),
                    "run_id": run_id,
                    "frame_count": len(rows),
                    "sample_count": len(output_rows),
                    "complete_source_run": summary["complete_source_run"],
                    "mapping_sha256": summary["mapping_sha256"],
                    "policy_id": summary["policy_id"],
                    "policy_sha256": summary["policy_sha256"],
                    "road_marking_class_values": summary[
                        "road_marking_class_values"
                    ],
                    "curation_manifest_sha256": (
                        curation_info["manifest_sha256"] if curation_info else None
                    ),
                },
            )
            _publish_output(output_root, temporary_root, overwrite)
            summary["output_root"] = str(output_root)
    except Exception:
        if not dry_run and temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        raise
    return summary


def _read_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise MaskBakeError("failed to read binary mask: {}".format(path))
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1}):
        raise MaskBakeError(
            "mask must contain only 0/1: {} values={}".format(path, values)
        )
    return mask.astype(np.uint8)


def _read_road_marking_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise MaskBakeError("failed to read road-marking mask: {}".format(path))
    values = set(np.unique(mask).tolist())
    allowed = set(ROAD_MARKING_CLASS_VALUES.values())
    if not values.issubset(allowed):
        raise MaskBakeError(
            "road-marking mask has unsupported values: {} values={} allowed={}".format(
                path, sorted(values), sorted(allowed)
            )
        )
    return mask.astype(np.uint8)


def build_model_cache(
    run_root,
    native_output_name=DEFAULT_NATIVE_OUTPUT_NAME,
    output_name=DEFAULT_OUTPUT_NAME,
    target_hw=TARGET_HW,
    overwrite=False,
    dry_run=False,
    write_previews=False,
):
    """Build a TwinLite cache exclusively from a completed native target set."""
    run_root = Path(run_root).resolve()
    native_output_name = _validate_output_name(native_output_name)
    native_root = run_root / "derived" / native_output_name
    native_manifest = native_root / "manifest.jsonl"
    if not (native_root / "_SUCCESS").is_file() or not native_manifest.is_file():
        raise MaskBakeError("completed native target set not found: {}".format(native_root))
    native_rows = _load_jsonl(native_manifest)
    if not native_rows:
        raise MaskBakeError("native manifest has no samples: {}".format(native_manifest))
    output_root, temporary_root = _temporary_output(
        run_root, output_name, overwrite, dry_run
    )
    run_id = str(native_rows[0]["run_id"])
    first_outputs = native_rows[0].get("outputs") or {}
    is_v3 = all(name in first_outputs for name in V3_MASK_NAMES)
    is_v2 = all(name in first_outputs for name in V2_MASK_NAMES)
    if is_v3:
        mask_names = V3_MASK_NAMES
        cached_mask_names = MASK_NAMES + ("drivable_ignore", "road_marking")
        cache_schema_version = MODEL_CACHE_V3_SCHEMA_VERSION
    elif is_v2:
        mask_names = V2_MASK_NAMES
        cached_mask_names = MASK_NAMES + ("drivable_ignore",)
        cache_schema_version = MODEL_CACHE_V2_SCHEMA_VERSION
    else:
        mask_names = MASK_NAMES
        cached_mask_names = MASK_NAMES
        cache_schema_version = MODEL_CACHE_SCHEMA_VERSION
    with (native_root / "_SUCCESS").open("r", encoding="utf-8") as stream:
        native_success = json.load(stream)
    transform_policy = {
        "native_manifest_sha256": _file_hash(native_manifest),
        "target_hw": list(target_hw),
        "mask_interpolation": "INTER_NEAREST",
        "rgb_downscale_interpolation": "INTER_AREA",
        "rgb_padding_value": 114,
        "mask_padding_value": 0,
        "valid_mask_padding_value": 0,
    }
    summary = {
        "schema_version": cache_schema_version,
        "run_id": run_id,
        "native_output_name": native_output_name,
        "native_manifest_sha256": transform_policy["native_manifest_sha256"],
        "transform_sha256": _stable_hash(transform_policy),
        "output_name": output_name,
        "dry_run": bool(dry_run),
        "sample_count": len(native_rows),
        "target_hw": list(target_hw),
        "transform_policy": transform_policy,
        "policy_id": native_success.get("policy_id"),
        "policy_sha256": native_success.get("policy_sha256"),
        "curation_manifest_sha256": native_success.get("curation_manifest_sha256"),
        "road_marking_class_values": (
            dict(ROAD_MARKING_CLASS_VALUES) if is_v3 else None
        ),
        "views": {},
    }
    counters = _new_counters(mask_names)
    geometry_rows = []
    try:
        for row in native_rows:
            frame_id = int(row["frame_id"])
            view = str(row["view"])
            stem = "{:06d}.png".format(frame_id)
            masks = {}
            geometry = None
            for name in mask_names:
                if name == "road_marking":
                    native_mask = _read_road_marking_mask(
                        native_root / row["outputs"][name]
                    )
                else:
                    native_mask = _read_binary_mask(
                        native_root / row["outputs"][name]
                    )
                baked, current_geometry = letterbox_mask(native_mask, target_hw)
                if geometry is None:
                    geometry = current_geometry
                elif geometry != current_geometry:
                    raise MaskBakeError("paired letterbox geometry diverged")
                masks[name] = baked
            valid_native = np.ones(tuple(geometry["original_hw"]), dtype=np.uint8)
            valid, valid_geometry = letterbox_mask(valid_native, target_hw)
            if geometry != valid_geometry:
                raise MaskBakeError("valid-mask geometry diverged")
            _update_counters(counters, view, masks, int(valid.sum()), mask_names)
            output_paths = {
                name: "{}_masks/{}/{}".format(name, view, stem)
                for name in cached_mask_names
            }
            if is_v2:
                output_paths["valid"] = "valid_masks/geometric/{}/{}".format(
                    view, stem
                )
                valid_masks = {
                    "lane": valid.copy(),
                    "drivable": valid & (1 - masks["drivable_ignore"]),
                    "stopline": valid.copy()
                    if row["task_valid"].get("stopline", False)
                    else np.zeros_like(valid),
                    **(
                        {
                            "road_marking": valid.copy()
                            if row["task_valid"].get("road_marking", False)
                            else np.zeros_like(valid)
                        }
                        if is_v3
                        else {}
                    ),
                }
                valid_paths = {
                    "lane": output_paths["valid"],
                    "drivable": "valid_masks/drivable/{}/{}".format(view, stem),
                    "stopline": output_paths["valid"],
                    **(
                        {"road_marking": output_paths["valid"]}
                        if is_v3
                        else {}
                    ),
                }
            else:
                output_paths["valid"] = "valid_masks/{}/{}".format(view, stem)
                valid_masks = {
                    name: valid for name in ("lane", "drivable", "stopline")
                }
                valid_paths = {
                    name: output_paths["valid"]
                    for name in ("lane", "drivable", "stopline")
                }
            if not dry_run:
                for name in cached_mask_names:
                    _write_png_atomic(temporary_root / output_paths[name], masks[name])
                _write_png_atomic(temporary_root / output_paths["valid"], valid)
                if is_v2:
                    _write_png_atomic(
                        temporary_root / valid_paths["drivable"],
                        valid_masks["drivable"],
                    )
                if write_previews:
                    rgb = _read_color_png(
                        run_root / row["source"]["intensity"], "intensity"
                    )
                    rgb_baked = letterbox_image(rgb, geometry)
                    preview = render_mask_overlay(
                        rgb_baked,
                        masks,
                        "{} frame={} view={} 384x640".format(run_id, frame_id, view),
                    )
                    _write_png_atomic(
                        temporary_root / "qa_overlays" / view / stem, preview
                    )
            geometry_rows.append(
                {
                    "schema_version": cache_schema_version,
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "view": view,
                    "source": row["source"],
                    "native_targets": row["outputs"],
                    "native_output_name": native_output_name,
                    "outputs": output_paths,
                    "valid_masks": valid_paths,
                    "geometry": geometry,
                    "task_valid": row["task_valid"],
                    **(
                        {
                            "target_encodings": {
                                "road_marking": {
                                    "type": "categorical_uint8",
                                    "class_values": dict(ROAD_MARKING_CLASS_VALUES),
                                }
                            }
                        }
                        if is_v3
                        else {}
                    ),
                }
            )
        summary["views"] = _summarize_counters(counters, mask_names)
        if not dry_run:
            _write_jsonl(temporary_root / "geometry.jsonl", geometry_rows)
            _write_json(temporary_root / "qa_summary.json", summary)
            _write_json(
                temporary_root / "_SUCCESS",
                {
                    "schema_version": cache_schema_version,
                    "completed_at_utc": _utc_now_iso(),
                    "run_id": run_id,
                    "sample_count": len(geometry_rows),
                    "native_manifest_sha256": summary["native_manifest_sha256"],
                    "transform_sha256": summary["transform_sha256"],
                    "policy_id": summary["policy_id"],
                    "policy_sha256": summary["policy_sha256"],
                    "curation_manifest_sha256": summary["curation_manifest_sha256"],
                    "native_output_name": native_output_name,
                    "road_marking_class_values": summary[
                        "road_marking_class_values"
                    ],
                },
            )
            _publish_output(output_root, temporary_root, overwrite)
            summary["output_root"] = str(output_root)
    except Exception:
        if not dry_run and temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        raise
    return summary


def bake_run(
    run_root,
    output_name=DEFAULT_OUTPUT_NAME,
    target_hw=TARGET_HW,
    overwrite=False,
    dry_run=False,
    max_frames=None,
    sample_frames=None,
    native_output_name=DEFAULT_NATIVE_OUTPUT_NAME,
    write_previews=False,
    policy_path=None,
    curation_name=None,
):
    """Compatibility wrapper: publish native targets, then TwinLite cache."""
    native_summary = bake_native_run(
        run_root,
        output_name=native_output_name,
        overwrite=overwrite,
        dry_run=dry_run,
        max_frames=max_frames,
        sample_frames=sample_frames,
        write_previews=write_previews,
        policy_path=policy_path,
        curation_name=curation_name,
    )
    if dry_run:
        return {"native": native_summary, "model_cache": None}
    cache_summary = build_model_cache(
        run_root,
        native_output_name=native_output_name,
        output_name=output_name,
        target_hw=target_hw,
        overwrite=overwrite,
        dry_run=False,
        write_previews=write_previews,
    )
    return {"native": native_summary, "model_cache": cache_summary}
