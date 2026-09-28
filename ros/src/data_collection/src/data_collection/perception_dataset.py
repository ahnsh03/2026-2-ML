# -*- coding: utf-8 -*-
# 이관: 인지 코드 레포 d0922bb `src/data_collection/src/data_collection/perception_dataset.py` (원작성: 안승현)
"""Create immutable, run-based perception dataset-version manifests."""

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


DATASET_SCHEMA_VERSION = "morai-perception-dataset-1.1.0"
SAMPLE_SCHEMA_VERSION = "morai-perception-sample-1.1.0"
DEFAULT_DATASET_VERSION = "twinlite_morai_v1"
DEFAULT_CACHE_NAME = "twinlite_384x640_v2"
DEFAULT_TRAINING_HEADS = ("lane", "drivable")
EXPECTED_VIEWS = ("front", "left", "right")
ROAD_MARKING_CLASS_VALUES = {
    "background": 0,
    "white_lane": 1,
    "yellow_lane": 2,
    "stopline": 3,
}


class PerceptionDatasetError(RuntimeError):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise PerceptionDatasetError("failed to read JSON {}: {}".format(path, exc))


def _load_jsonl(path):
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise PerceptionDatasetError(
                        "invalid JSON at {}:{}: {}".format(path, line_number, exc)
                    )
    except OSError as exc:
        raise PerceptionDatasetError("failed to read {}: {}".format(path, exc))
    return rows


def _safe_name(value, label):
    value = str(value)
    if not value or value in (".", "..") or Path(value).name != value:
        raise PerceptionDatasetError("{} must be one safe directory name".format(label))
    return value


def _data_relative(path, data_root):
    try:
        return Path(path).resolve().relative_to(data_root).as_posix()
    except ValueError:
        raise PerceptionDatasetError(
            "path escapes data root: {} (root={})".format(path, data_root)
        )


def _verify_binary_mask(path, expected_hw):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise PerceptionDatasetError("failed to read single-channel mask: {}".format(path))
    if list(mask.shape) != list(expected_hw):
        raise PerceptionDatasetError(
            "mask shape mismatch: {} expected={} actual={}".format(
                path, list(expected_hw), list(mask.shape)
            )
        )
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1}):
        raise PerceptionDatasetError(
            "mask values must be 0/1: {} values={}".format(path, sorted(values))
        )


def _verify_road_marking_mask(path, expected_hw):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise PerceptionDatasetError("failed to read road-marking mask: {}".format(path))
    if list(mask.shape) != list(expected_hw):
        raise PerceptionDatasetError(
            "mask shape mismatch: {} expected={} actual={}".format(
                path, list(expected_hw), list(mask.shape)
            )
        )
    values = set(np.unique(mask).tolist())
    allowed = set(ROAD_MARKING_CLASS_VALUES.values())
    if not values.issubset(allowed):
        raise PerceptionDatasetError(
            "road-marking values must be {}: {} values={}".format(
                sorted(allowed), path, sorted(values)
            )
        )


def _prepare_split_assignments(train_runs, val_runs, test_runs):
    assignments = {
        "train": [str(run_id) for run_id in train_runs],
        "val": [str(run_id) for run_id in val_runs],
        "test": [str(run_id) for run_id in test_runs],
    }
    owners = {}
    for split, run_ids in assignments.items():
        if len(run_ids) != len(set(run_ids)):
            raise PerceptionDatasetError("duplicate run in {} split".format(split))
        for run_id in run_ids:
            _safe_name(run_id, "run id")
            if run_id in owners:
                raise PerceptionDatasetError(
                    "run {} appears in both {} and {}".format(
                        run_id, owners[run_id], split
                    )
                )
            owners[run_id] = split
    if not assignments["train"]:
        raise PerceptionDatasetError("train split must contain at least one run")
    return assignments


def _normalize_excluded_frames(excluded_frames_by_run, assignments):
    normalized = {}
    assigned_runs = {
        run_id for run_ids in assignments.values() for run_id in run_ids
    }
    for run_id, frame_ids in (excluded_frames_by_run or {}).items():
        run_id = _safe_name(run_id, "excluded-frame run id")
        if run_id not in assigned_runs:
            raise PerceptionDatasetError(
                "excluded-frame run is not assigned to a split: {}".format(run_id)
            )
        values = set()
        for frame_id in frame_ids:
            frame_id = int(frame_id)
            if frame_id < 0:
                raise PerceptionDatasetError("excluded frame id must be non-negative")
            values.add(frame_id)
        if values:
            normalized[run_id] = values
    return normalized


def _build_run_rows(
    data_root,
    dataset_name,
    split,
    run_id,
    cache_name,
    verify_masks,
    excluded_frame_ids=(),
):
    run_root = data_root / "datasets" / run_id
    cache_root = run_root / "derived" / cache_name
    success_path = cache_root / "_SUCCESS"
    geometry_path = cache_root / "geometry.jsonl"
    if not success_path.is_file() or not geometry_path.is_file():
        raise PerceptionDatasetError(
            "completed model cache not found for {}: {}".format(run_id, cache_root)
        )

    success = _load_json(success_path)
    geometry_rows = _load_jsonl(geometry_path)
    if not geometry_rows:
        raise PerceptionDatasetError("empty geometry manifest: {}".format(geometry_path))
    if int(success.get("sample_count", -1)) != len(geometry_rows):
        raise PerceptionDatasetError(
            "cache sample count mismatch for {}: marker={} manifest={}".format(
                run_id, success.get("sample_count"), len(geometry_rows)
            )
        )
    cache_target_name_sets = {
        tuple(
            sorted(
                {"lane", "drivable", "stopline"}
                | (
                    {"road_marking"}
                    if "road_marking" in (row.get("outputs") or {})
                    else set()
                )
            )
        )
        for row in geometry_rows
    }
    if len(cache_target_name_sets) != 1:
        raise PerceptionDatasetError(
            "cache samples disagree on stored targets for {}: {}".format(
                run_id, sorted(cache_target_name_sets)
            )
        )
    cache_target_names = list(next(iter(cache_target_name_sets)))

    excluded_frame_ids = set(int(frame_id) for frame_id in excluded_frame_ids)
    cached_frame_ids = {int(row["frame_id"]) for row in geometry_rows}
    missing_exclusions = excluded_frame_ids - cached_frame_ids
    if missing_exclusions:
        raise PerceptionDatasetError(
            "excluded frames are absent from cache for {}: {}".format(
                run_id, sorted(missing_exclusions)
            )
        )

    output_rows = []
    seen = set()
    view_counts = Counter()
    target_hw = None
    for row in geometry_rows:
        row_run_id = str(row.get("run_id"))
        frame_id = int(row["frame_id"])
        view = str(row["view"])
        if row_run_id != run_id:
            raise PerceptionDatasetError(
                "run mismatch in cache: expected={} actual={}".format(run_id, row_run_id)
            )
        if view not in EXPECTED_VIEWS:
            raise PerceptionDatasetError("unsupported camera view: {}".format(view))
        if frame_id in excluded_frame_ids:
            continue
        key = (run_id, frame_id, view)
        if key in seen:
            raise PerceptionDatasetError("duplicate sample: {}".format(key))
        seen.add(key)

        geometry = row["geometry"]
        current_hw = [int(v) for v in geometry["target_hw"]]
        if target_hw is None:
            target_hw = current_hw
        elif target_hw != current_hw:
            raise PerceptionDatasetError(
                "mixed target dimensions in {}: {} vs {}".format(
                    run_id, target_hw, current_hw
                )
            )

        image_path = run_root / row["source"]["intensity"]
        if not image_path.is_file():
            raise PerceptionDatasetError("source image not found: {}".format(image_path))

        cache_outputs = row["outputs"]
        target_paths = {
            "lane": cache_root / cache_outputs["lane"],
            "drivable": cache_root / cache_outputs["drivable"],
            "stopline": cache_root / cache_outputs["stopline"],
        }
        if "road_marking" in cache_outputs:
            target_paths["road_marking"] = cache_root / cache_outputs["road_marking"]
        auxiliary_paths = {
            name: cache_root / cache_outputs[name]
            for name in ("drivable_ignore", "actor_occupancy_gt")
            if name in cache_outputs
        }
        native_targets = row.get("native_targets") or {}
        native_output_name = row.get("native_output_name") or success.get(
            "native_output_name"
        )
        if "actor_occupancy_gt" in native_targets and native_output_name:
            auxiliary_paths["actor_occupancy_gt"] = (
                cache_root.parent / native_output_name / native_targets["actor_occupancy_gt"]
            )
        valid_path = cache_root / cache_outputs["valid"]
        valid_paths = {
            name: cache_root / relative
            for name, relative in (row.get("valid_masks") or {}).items()
        }
        if not valid_paths:
            valid_paths = {
                name: valid_path for name in ("lane", "drivable", "stopline")
            }
        required_valid = {"lane", "drivable", "stopline"}
        if "road_marking" in target_paths:
            required_valid.add("road_marking")
        if not required_valid.issubset(valid_paths):
            raise PerceptionDatasetError(
                "cache is missing head-valid masks for {}".format(run_id)
            )
        binary_paths = [
            path for name, path in target_paths.items() if name != "road_marking"
        ] + [valid_path] + list(valid_paths.values())
        for path in binary_paths:
            if not path.is_file():
                raise PerceptionDatasetError("derived mask not found: {}".format(path))
            if verify_masks:
                _verify_binary_mask(path, current_hw)
        if "road_marking" in target_paths:
            marking_path = target_paths["road_marking"]
            if not marking_path.is_file():
                raise PerceptionDatasetError(
                    "derived mask not found: {}".format(marking_path)
                )
            if verify_masks:
                _verify_road_marking_mask(marking_path, current_hw)
        for path in auxiliary_paths.values():
            if not path.is_file():
                raise PerceptionDatasetError("auxiliary target not found: {}".format(path))

        sample_id = "{}/{}/{:06d}".format(run_id, view, frame_id)
        output_rows.append(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "dataset_version": dataset_name,
                "split": split,
                "sample_id": sample_id,
                "run_id": run_id,
                "frame_id": frame_id,
                "view": view,
                "image": _data_relative(image_path, data_root),
                "targets": {
                    name: _data_relative(path, data_root)
                    for name, path in target_paths.items()
                },
                "valid_mask": _data_relative(valid_path, data_root),
                "valid_masks": {
                    name: _data_relative(path, data_root)
                    for name, path in valid_paths.items()
                },
                "auxiliary_targets": {
                    name: _data_relative(path, data_root)
                    for name, path in auxiliary_paths.items()
                },
                "geometry": geometry,
                "task_valid": row["task_valid"],
                "target_encodings": row.get("target_encodings") or {},
            }
        )
        view_counts[view] += 1

    output_rows.sort(key=lambda item: (item["run_id"], item["frame_id"], item["view"]))
    return output_rows, {
        "run_id": run_id,
        "split": split,
        "sample_count": len(output_rows),
        "frame_count": len(set(row["frame_id"] for row in output_rows)),
        "excluded_frame_ids": sorted(excluded_frame_ids),
        "excluded_sample_count": sum(
            1 for row in geometry_rows if int(row["frame_id"]) in excluded_frame_ids
        ),
        "view_counts": dict(sorted(view_counts.items())),
        "target_hw": target_hw,
        "cache_geometry_sha256": _file_hash(geometry_path),
        "cache_transform_sha256": success.get("transform_sha256"),
        "policy_id": success.get("policy_id"),
        "policy_sha256": success.get("policy_sha256"),
        "curation_manifest_sha256": success.get("curation_manifest_sha256"),
        "target_names": cache_target_names,
    }


def build_dataset_version(
    data_root,
    train_runs,
    val_runs=(),
    test_runs=(),
    dataset_name=DEFAULT_DATASET_VERSION,
    cache_name=DEFAULT_CACHE_NAME,
    training_heads=DEFAULT_TRAINING_HEADS,
    overwrite=False,
    verify_masks=True,
    excluded_frames_by_run=None,
):
    """Publish train/val/test manifests without flattening run directories."""
    data_root = Path(data_root).expanduser().resolve()
    dataset_name = _safe_name(dataset_name, "dataset name")
    cache_name = _safe_name(cache_name, "cache name")
    assignments = _prepare_split_assignments(train_runs, val_runs, test_runs)
    excluded_frames_by_run = _normalize_excluded_frames(
        excluded_frames_by_run, assignments
    )
    training_heads = tuple(str(head) for head in training_heads)
    supported_heads = {"lane", "drivable", "road_marking", "stopline"}
    if not training_heads or not set(training_heads).issubset(supported_heads):
        raise PerceptionDatasetError(
            "training heads must be a non-empty subset of {}".format(
                ",".join(sorted(supported_heads))
            )
        )
    if len(training_heads) != len(set(training_heads)):
        raise PerceptionDatasetError("training heads must not contain duplicates")

    version_root = data_root / "dataset_versions" / dataset_name
    temporary_root = version_root.with_name(".{}.{}.tmp".format(dataset_name, os.getpid()))
    if version_root.exists() and not overwrite:
        raise PerceptionDatasetError(
            "dataset version already exists (use --overwrite): {}".format(version_root)
        )
    if temporary_root.exists():
        shutil.rmtree(str(temporary_root))
    temporary_root.mkdir(parents=True)

    split_rows = {"train": [], "val": [], "test": []}
    run_summaries = []
    try:
        for split in ("train", "val", "test"):
            for run_id in assignments[split]:
                rows, summary = _build_run_rows(
                    data_root,
                    dataset_name,
                    split,
                    run_id,
                    cache_name,
                    verify_masks,
                    excluded_frames_by_run.get(run_id, ()),
                )
                split_rows[split].extend(rows)
                run_summaries.append(summary)

        all_sample_ids = []
        split_summaries = {}
        for split in ("train", "val", "test"):
            rows = split_rows[split]
            sample_ids = [row["sample_id"] for row in rows]
            if len(sample_ids) != len(set(sample_ids)):
                raise PerceptionDatasetError("duplicate sample id in {}".format(split))
            all_sample_ids.extend(sample_ids)
            manifest_path = temporary_root / "{}.jsonl".format(split)
            _write_jsonl(manifest_path, rows)
            split_summaries[split] = {
                "runs": assignments[split],
                "sample_count": len(rows),
                "frame_count": sum(
                    item["frame_count"]
                    for item in run_summaries
                    if item["split"] == split
                ),
                "view_counts": dict(
                    sorted(Counter(row["view"] for row in rows).items())
                ),
                "manifest_sha256": _file_hash(manifest_path),
            }
        if len(all_sample_ids) != len(set(all_sample_ids)):
            raise PerceptionDatasetError("a sample appears in more than one split")

        target_hws = {
            tuple(summary["target_hw"])
            for summary in run_summaries
            if summary["target_hw"] is not None
        }
        if len(target_hws) != 1:
            raise PerceptionDatasetError(
                "dataset runs disagree on target dimensions: {}".format(target_hws)
            )
        target_hw = list(next(iter(target_hws)))
        target_policy_ids = sorted(
            {summary["policy_id"] for summary in run_summaries if summary.get("policy_id")}
        )
        stored_target_names = sorted(
            {
                name
                for summary in run_summaries
                for name in summary.get("target_names", [])
            }
        )
        target_name_sets = {
            tuple(summary.get("target_names", [])) for summary in run_summaries
        }
        if len(target_name_sets) != 1:
            raise PerceptionDatasetError(
                "dataset runs disagree on stored targets: {}".format(
                    sorted(target_name_sets)
                )
            )
        missing_training_heads = set(training_heads) - set(stored_target_names)
        if missing_training_heads:
            raise PerceptionDatasetError(
                "training heads are absent from the cache: {}".format(
                    sorted(missing_training_heads)
                )
            )
        metadata = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_version": dataset_name,
            "created_at_utc": _utc_now_iso(),
            "path_contract": "all manifest paths are relative to data_root",
            "cache_name": cache_name,
            "target_hw": target_hw,
            "views": list(EXPECTED_VIEWS),
            "training_heads": list(training_heads),
            "target_encodings": (
                {
                    "road_marking": {
                        "type": "categorical_uint8",
                        "class_values": ROAD_MARKING_CLASS_VALUES,
                    }
                }
                if "road_marking" in stored_target_names
                else {}
            ),
            "stored_future_targets": [
                name for name in stored_target_names if name not in training_heads
            ],
            "stored_auxiliary_targets": (
                ["drivable_ignore", "actor_occupancy_gt"] if target_policy_ids else []
            ),
            "target_policy_ids": target_policy_ids,
            "head_valid_policy": {
                "lane": "geometric_valid",
                "drivable": "per-sample valid_masks.drivable (policy-specific)",
                "stopline": "geometric_valid AND per-sample task_valid",
                **(
                    {"road_marking": "geometric_valid AND per-sample task_valid"}
                    if "road_marking" in stored_target_names
                    else {}
                ),
            },
            "rgb_policy": {
                "copied": False,
                "source": "datasets/<run_id>/frames/intensity/<view>/<frame>.png",
                "transform": "apply per-sample geometry at load time; padding value 114",
            },
            "split_policy": "exclusive run-level split; no frame-level random split",
            "explicit_frame_exclusions": {
                run_id: sorted(frame_ids)
                for run_id, frame_ids in sorted(excluded_frames_by_run.items())
            },
            "splits": split_summaries,
            "runs": run_summaries,
            "total_sample_count": len(all_sample_ids),
            "mask_verification": (
                "all files, shapes, binary values, and categorical values checked"
                if "road_marking" in stored_target_names
                else "all files, shapes, and binary values checked"
            )
            if verify_masks
            else "file existence checked; pixel audit skipped",
        }
        _write_json(temporary_root / "dataset.json", metadata)
        _write_json(
            temporary_root / "_SUCCESS",
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "dataset_version": dataset_name,
                "completed_at_utc": _utc_now_iso(),
                "total_sample_count": len(all_sample_ids),
                "split_manifest_sha256": {
                    split: split_summaries[split]["manifest_sha256"]
                    for split in ("train", "val", "test")
                },
            },
        )
        version_root.parent.mkdir(parents=True, exist_ok=True)
        if version_root.exists():
            shutil.rmtree(str(version_root))
        os.replace(str(temporary_root), str(version_root))
        metadata["output_root"] = str(version_root)
        return metadata
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        raise
