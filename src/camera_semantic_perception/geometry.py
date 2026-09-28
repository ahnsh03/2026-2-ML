# 이관: 인지 코드 레포 d0922bb `src/perception/camera_semantic_perception/src/camera_semantic_perception/geometry.py` (원작성: 안승현)
"""Training-compatible letterbox geometry and reversible probability mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxGeometry:
    original_hw: Tuple[int, int]
    target_hw: Tuple[int, int]
    resized_hw: Tuple[int, int]
    pad_ltrb: Tuple[int, int, int, int]
    scale_xy: Tuple[float, float]


def compute_letterbox_geometry(
    original_hw: Tuple[int, int], target_hw: Tuple[int, int]
) -> LetterboxGeometry:
    source_h, source_w = (int(original_hw[0]), int(original_hw[1]))
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("image and target dimensions must be positive")

    scale = min(float(target_w) / source_w, float(target_h) / source_h)
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    right = target_w - resized_w - left
    bottom = target_h - resized_h - top
    return LetterboxGeometry(
        original_hw=(source_h, source_w),
        target_hw=(target_h, target_w),
        resized_hw=(resized_h, resized_w),
        pad_ltrb=(left, top, right, bottom),
        scale_xy=(float(resized_w) / source_w, float(resized_h) / source_h),
    )


def letterbox_bgr(
    image_bgr: np.ndarray,
    target_hw: Tuple[int, int] = (384, 640),
    padding_value: int = 114,
) -> Tuple[np.ndarray, LetterboxGeometry]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
        raise ValueError("input must be a BGR/BGRA image")
    geometry = compute_letterbox_geometry(
        (int(image_bgr.shape[0]), int(image_bgr.shape[1])), target_hw
    )
    resized_h, resized_w = geometry.resized_hw
    shrinking = resized_h <= image_bgr.shape[0] and resized_w <= image_bgr.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized_bgr = cv2.resize(
        image_bgr[:, :, :3], (resized_w, resized_h), interpolation=interpolation
    )
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    target_h, target_w = geometry.target_hw
    output = np.full(
        (target_h, target_w, 3), int(padding_value), dtype=np.uint8
    )
    left, top, _, _ = geometry.pad_ltrb
    output[top : top + resized_h, left : left + resized_w] = resized_rgb
    nchw = np.ascontiguousarray(output.transpose(2, 0, 1)).astype(np.float32)
    nchw *= 1.0 / 255.0
    return nchw, geometry


def restore_probability(
    probability: np.ndarray, geometry: LetterboxGeometry
) -> np.ndarray:
    if probability.ndim != 2 or tuple(probability.shape) != geometry.target_hw:
        raise ValueError(
            "probability shape must be {}, got {}".format(
                geometry.target_hw, probability.shape
            )
        )
    left, top, _, _ = geometry.pad_ltrb
    resized_h, resized_w = geometry.resized_hw
    cropped = probability[top : top + resized_h, left : left + resized_w]
    original_h, original_w = geometry.original_hw
    restored = cv2.resize(
        cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR
    )
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)


def restore_categorical_probabilities(
    probabilities: np.ndarray, geometry: LetterboxGeometry
) -> np.ndarray:
    """Restore a ``[C,H,W]`` softmax volume and renormalize each pixel."""
    value = np.asarray(probabilities)
    if value.ndim != 3 or tuple(value.shape[1:]) != geometry.target_hw:
        raise ValueError(
            "categorical probability shape must be [C,{},{}], got {}".format(
                geometry.target_hw[0], geometry.target_hw[1], value.shape
            )
        )
    if value.dtype != np.float32:
        raise ValueError("categorical probabilities must be float32")
    restored = np.stack(
        [restore_probability(channel, geometry) for channel in value], axis=0
    )
    total = restored.sum(axis=0, keepdims=True)
    restored /= np.maximum(total, np.finfo(np.float32).eps)
    return np.ascontiguousarray(restored, dtype=np.float32)


def restore_class_ids(
    class_ids: np.ndarray, geometry: LetterboxGeometry
) -> np.ndarray:
    """Remove letterbox padding and restore a categorical mask losslessly."""
    if class_ids.ndim != 2 or tuple(class_ids.shape) != geometry.target_hw:
        raise ValueError(
            "class ID shape must be {}, got {}".format(
                geometry.target_hw, class_ids.shape
            )
        )
    if class_ids.dtype != np.uint8:
        raise ValueError("class IDs must be uint8")
    left, top, _, _ = geometry.pad_ltrb
    resized_h, resized_w = geometry.resized_hw
    cropped = class_ids[top : top + resized_h, left : left + resized_w]
    original_h, original_w = geometry.original_hw
    return cv2.resize(
        cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8, copy=False)
