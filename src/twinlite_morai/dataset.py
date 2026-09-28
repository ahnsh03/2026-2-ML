# 이관: 인지 코드 레포 d0922bb `src/perception/twinlite_morai/dataset.py` (원작성: 안승현)
"""Manifest-backed MORAI dataset adapter for shared 3-view TwinLite training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .tasks import ROAD_MARKING_CLASS_VALUES, task_from_training_heads


SUPPORTED_DATASET_SCHEMAS = {
    "morai-perception-dataset-1.0.0",
    "morai-perception-dataset-1.1.0",
}
SUPPORTED_SAMPLE_SCHEMAS = {
    "morai-perception-sample-1.0.0",
    "morai-perception-sample-1.1.0",
}
DEFAULT_DATASET_VERSION = "twinlite_morai_v1"


class MoraiDatasetError(RuntimeError):
    """Raised when the version manifest and files violate the data contract."""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise MoraiDatasetError(f"failed to read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise MoraiDatasetError(
                        f"invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise MoraiDatasetError(f"failed to read manifest {path}: {exc}") from exc
    return rows


def _resolve_under_root(data_root: Path, relative: str) -> Path:
    candidate = (data_root / relative).resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise MoraiDatasetError(
            f"manifest path escapes data root: {relative}"
        ) from exc
    return candidate


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MoraiDatasetError(f"failed to read RGB input: {path}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise MoraiDatasetError(
            f"RGB input must be BGR or BGRA: {path} shape={image.shape}"
        )
    bgr = image[:, :, :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _letterbox_rgb(rgb: np.ndarray, geometry: Dict[str, Any]) -> np.ndarray:
    original_hw = [int(value) for value in geometry["original_hw"]]
    target_h, target_w = [int(value) for value in geometry["target_hw"]]
    resized_h, resized_w = [int(value) for value in geometry["resized_hw"]]
    left, top, right, bottom = [int(value) for value in geometry["pad_ltrb"]]

    if list(rgb.shape[:2]) != original_hw:
        raise MoraiDatasetError(
            f"RGB shape differs from recorded geometry: "
            f"expected={original_hw} actual={list(rgb.shape[:2])}"
        )
    if left + resized_w + right != target_w or top + resized_h + bottom != target_h:
        raise MoraiDatasetError(f"invalid letterbox geometry: {geometry}")

    shrinking = resized_h <= rgb.shape[0] and resized_w <= rgb.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=interpolation)
    output = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    output[top : top + resized_h, left : left + resized_w] = resized
    return output


def _read_binary_mask(path: Path, expected_hw: List[int], label: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise MoraiDatasetError(f"failed to read {label} mask: {path}")
    if list(mask.shape) != expected_hw:
        raise MoraiDatasetError(
            f"{label} mask shape mismatch: expected={expected_hw} "
            f"actual={list(mask.shape)} path={path}"
        )
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1}):
        raise MoraiDatasetError(
            f"{label} mask must contain only 0/1: path={path} values={values}"
        )
    return mask.astype(np.uint8, copy=False)


def _read_categorical_mask(
    path: Path, expected_hw: List[int], label: str, class_count: int
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise MoraiDatasetError(f"failed to read {label} mask: {path}")
    if list(mask.shape) != expected_hw:
        raise MoraiDatasetError(
            f"{label} mask shape mismatch: expected={expected_hw} "
            f"actual={list(mask.shape)} path={path}"
        )
    values = set(np.unique(mask).tolist())
    allowed = set(range(int(class_count)))
    if not values.issubset(allowed):
        raise MoraiDatasetError(
            f"{label} mask class ids must be in {sorted(allowed)}: "
            f"path={path} values={sorted(values)}"
        )
    return mask.astype(np.uint8, copy=False)


class MoraiTwinLiteDataset(Dataset):
    """Load RGB and paired two-head targets from a version manifest.

    RGB is read from the immutable run directory and letterboxed on demand. The
    pre-baked targets and valid masks already share the recorded geometry. The
    supported tasks are binary drivable+lane and binary drivable+4-class road
    marking. A single dataset instance pools front/left/right samples without
    flattening their physical directories or losing run/view metadata.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        dataset_version: str = DEFAULT_DATASET_VERSION,
        split: str = "train",
        joint_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        verify_files: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        configured_root = data_root or os.environ.get("PERCEPTION_DATA")
        if not configured_root:
            raise MoraiDatasetError(
                "data_root is required when PERCEPTION_DATA is not configured"
            )
        self.data_root = Path(configured_root).expanduser().resolve()
        self.version_root = (
            self.data_root / "dataset_versions" / str(dataset_version)
        ).resolve()
        self.split = str(split)
        self.joint_transform = joint_transform

        if self.split not in ("train", "val", "test"):
            raise MoraiDatasetError(f"unsupported split: {self.split}")
        if not (self.version_root / "_SUCCESS").is_file():
            raise MoraiDatasetError(
                f"completed dataset version not found: {self.version_root}"
            )

        self.dataset_meta = _load_json(self.version_root / "dataset.json")
        if self.dataset_meta.get("schema_version") not in SUPPORTED_DATASET_SCHEMAS:
            raise MoraiDatasetError(
                "unsupported dataset schema: "
                f"{self.dataset_meta.get('schema_version')}"
            )
        self.target_hw = [int(value) for value in self.dataset_meta["target_hw"]]
        self.training_heads = tuple(self.dataset_meta["training_heads"])
        try:
            self.task_spec = task_from_training_heads(self.training_heads)
        except ValueError as exc:
            raise MoraiDatasetError(str(exc)) from exc
        self.heads = self.task_spec.heads

        manifest_path = self.version_root / f"{self.split}.jsonl"
        self.rows = _load_jsonl(manifest_path)
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise MoraiDatasetError("max_samples must be positive")
            self.rows = self.rows[: int(max_samples)]
        if self.split != "test" and not self.rows:
            raise MoraiDatasetError(f"split is empty: {manifest_path}")

        seen = set()
        for row in self.rows:
            if row.get("schema_version") not in SUPPORTED_SAMPLE_SCHEMAS:
                raise MoraiDatasetError(
                    f"unsupported sample schema: {row.get('schema_version')}"
                )
            if row.get("split") != self.split:
                raise MoraiDatasetError(
                    f"sample {row.get('sample_id')} belongs to {row.get('split')}, "
                    f"not {self.split}"
                )
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise MoraiDatasetError(f"duplicate sample id: {sample_id}")
            seen.add(sample_id)
            if [int(value) for value in row["geometry"]["target_hw"]] != self.target_hw:
                raise MoraiDatasetError(
                    f"sample target shape differs from dataset: {sample_id}"
                )
            if self.task_spec.secondary_head == "road_marking":
                encoding = (row.get("target_encodings") or {}).get(
                    "road_marking", {}
                )
                if (
                    encoding.get("type") != "categorical_uint8"
                    or encoding.get("class_values") != ROAD_MARKING_CLASS_VALUES
                ):
                    raise MoraiDatasetError(
                        f"sample {sample_id} has an invalid road-marking encoding"
                    )
            if verify_files:
                relative_paths = [row["image"], row["valid_mask"]]
                relative_paths.extend(row["targets"][head] for head in self.heads)
                relative_paths.extend((row.get("valid_masks") or {}).values())
                missing = [
                    str(_resolve_under_root(self.data_root, relative))
                    for relative in relative_paths
                    if not _resolve_under_root(self.data_root, relative).is_file()
                ]
                if missing:
                    raise MoraiDatasetError(
                        f"sample {sample_id} references missing files: {missing}"
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        geometry = row["geometry"]
        rgb_path = _resolve_under_root(self.data_root, row["image"])
        rgb = _letterbox_rgb(_read_rgb(rgb_path), geometry)

        drivable_path = _resolve_under_root(
            self.data_root, row["targets"]["drivable"]
        )
        valid_path = _resolve_under_root(self.data_root, row["valid_mask"])
        drivable = _read_binary_mask(
            drivable_path, self.target_hw, "drivable"
        )
        secondary_head = self.task_spec.secondary_head
        secondary_path = _resolve_under_root(
            self.data_root, row["targets"][secondary_head]
        )
        if secondary_head == "road_marking":
            secondary = _read_categorical_mask(
                secondary_path,
                self.target_hw,
                secondary_head,
                self.task_spec.class_counts[secondary_head],
            )
        else:
            secondary = _read_binary_mask(
                secondary_path, self.target_hw, secondary_head
            )
        valid = _read_binary_mask(valid_path, self.target_hw, "valid")
        valid_masks = {}
        for head in self.heads:
            relative = (row.get("valid_masks") or {}).get(head, row["valid_mask"])
            valid_masks[head] = _read_binary_mask(
                _resolve_under_root(self.data_root, relative),
                self.target_hw,
                head + " valid",
            )

        arrays: Dict[str, Any] = {
            "image": rgb,
            "targets": {"drivable": drivable, secondary_head: secondary},
            "valid_mask": valid,
            "valid_masks": valid_masks,
        }
        if self.joint_transform is not None:
            arrays = self.joint_transform(arrays)

        image = np.ascontiguousarray(arrays["image"].transpose(2, 0, 1))
        drivable = np.ascontiguousarray(arrays["targets"]["drivable"])
        secondary = np.ascontiguousarray(arrays["targets"][secondary_head])
        valid = np.ascontiguousarray(arrays["valid_mask"])
        valid_masks = {
            head: np.ascontiguousarray(arrays["valid_masks"][head])
            for head in self.heads
        }

        return {
            "image": torch.from_numpy(image).float().div_(255.0),
            "targets": {
                "drivable": torch.from_numpy(drivable).long(),
                secondary_head: torch.from_numpy(secondary).long(),
            },
            "valid_mask": torch.from_numpy(valid).bool(),
            "valid_masks": {
                head: torch.from_numpy(mask).bool()
                for head, mask in valid_masks.items()
            },
            "meta": {
                "sample_id": row["sample_id"],
                "run_id": row["run_id"],
                "frame_id": int(row["frame_id"]),
                "view": row["view"],
                "geometry": geometry,
            },
        }
