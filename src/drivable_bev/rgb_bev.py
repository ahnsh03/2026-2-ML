# 이관: 인지 코드 레포 d0922bb `src/perception/drivable_bev/src/drivable_bev/rgb_bev.py` (원작성: 안승현)
"""Raw RGB projection and map-overlay helpers for calibration review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .grid import BevGridSpec


@dataclass(frozen=True)
class StampedEgoPose:
    stamp_ns: int
    x_m: float
    y_m: float
    heading_deg: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.stamp_ns, self.x_m, self.y_m, self.heading_deg], dtype=np.float64
        )
        if self.stamp_ns <= 0 or not np.isfinite(values).all():
            raise ValueError("ego pose stamp and values must be finite and positive")


def nearest_ego_pose(
    poses: Sequence[StampedEgoPose], target_stamp_ns: int, max_age_ns: int
):
    """Return the nearest pose and its signed time delta, or ``(None, None)``."""
    if int(target_stamp_ns) <= 0 or int(max_age_ns) < 0:
        raise ValueError("target stamp must be positive and max age non-negative")
    if not poses:
        return None, None
    pose = min(poses, key=lambda value: abs(value.stamp_ns - int(target_stamp_ns)))
    delta_ns = int(pose.stamp_ns) - int(target_stamp_ns)
    if abs(delta_ns) > int(max_age_ns):
        return None, delta_ns
    return pose, delta_ns


def warp_rgb_to_bev(image: np.ndarray, bev_from_image: np.ndarray, shape) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("RGB source must be a uint8 HxWx3 BGR image")
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("BEV output shape must be positive")
    return np.ascontiguousarray(
        cv2.warpPerspective(
            value,
            np.asarray(bev_from_image, dtype=np.float64),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    )


def fuse_rgb_views(
    warped_by_view: Mapping[str, np.ndarray], quality_by_view: Mapping[str, np.ndarray]
):
    """Feather overlapping RGB observations using deterministic geometry weights."""
    if not warped_by_view or set(warped_by_view) != set(quality_by_view):
        raise ValueError("RGB images and quality maps must contain identical views")
    views = tuple(sorted(warped_by_view))
    images = []
    weights = []
    shape = None
    for view in views:
        image = np.asarray(warped_by_view[view])
        quality = np.asarray(quality_by_view[view], dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("{} RGB BEV must be uint8 HxWx3".format(view))
        if quality.shape != image.shape[:2] or not np.isfinite(quality).all():
            raise ValueError("{} quality map does not match RGB BEV".format(view))
        if np.any(quality < 0.0):
            raise ValueError("quality weights cannot be negative")
        shape = image.shape if shape is None else shape
        if image.shape != shape:
            raise ValueError("all RGB BEV images must have the same shape")
        images.append(image.astype(np.float32))
        weights.append(quality)
    image_stack = np.stack(images)
    weight_stack = np.stack(weights)
    weight_sum = np.sum(weight_stack, axis=0)
    valid = weight_sum > 0.0
    fused = np.zeros(shape, dtype=np.uint8)
    weighted = np.sum(image_stack * weight_stack[:, :, :, None], axis=0)
    fused[valid] = np.clip(
        np.rint(weighted[valid] / weight_sum[valid, None]), 0, 255
    ).astype(np.uint8)
    source_count = np.count_nonzero(weight_stack > 0.0, axis=0).astype(np.uint8)
    return np.ascontiguousarray(fused), np.ascontiguousarray(source_count)


def draw_metric_polyline(
    image: np.ndarray,
    points_xy: np.ndarray,
    grid: BevGridSpec,
    color,
    thickness: int = 2,
) -> None:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return
    pixels = grid.metric_to_pixel(points)
    finite = np.isfinite(pixels).all(axis=1)
    for index in range(len(points) - 1):
        # The map artifact is sampled at 0.5 m. Avoid drawing a fictitious
        # chord if a future artifact contains a discontinuity.
        if (
            finite[index]
            and finite[index + 1]
            and np.linalg.norm(points[index + 1] - points[index]) <= 2.0
        ):
            cv2.line(
                image,
                tuple(np.rint(pixels[index]).astype(int)),
                tuple(np.rint(pixels[index + 1]).astype(int)),
                tuple(int(value) for value in color),
                int(thickness),
                cv2.LINE_AA,
            )


def overlay_shadow_map_boundaries(
    fused_rgb: np.ndarray,
    boundaries_base: Mapping[str, np.ndarray],
    grid: BevGridSpec,
    alpha: float = 0.85,
) -> np.ndarray:
    """Overlay only the three map pseudo-boundaries on a fused raw RGB BEV."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("overlay alpha must lie in [0,1]")
    expected = {
        "white_negative_normal",
        "yellow_center",
        "white_positive_normal",
    }
    if set(boundaries_base) != expected:
        raise ValueError("pseudo-GT must contain exactly the three shadow boundaries")
    base = np.asarray(fused_rgb)
    if base.shape != (*grid.shape, 3) or base.dtype != np.uint8:
        raise ValueError("fused RGB does not match the configured BEV grid")
    layer = base.copy()
    # Magenta makes white-boundary map geometry visible without being confused
    # with the actual white paint. Yellow keeps the centre-line class explicit.
    styles = {
        "white_negative_normal": (255, 0, 255),
        "yellow_center": (0, 255, 255),
        "white_positive_normal": (255, 0, 255),
    }
    for name in (
        "white_negative_normal",
        "yellow_center",
        "white_positive_normal",
    ):
        draw_metric_polyline(layer, boundaries_base[name], grid, styles[name], 2)
    return cv2.addWeighted(layer, float(alpha), base, 1.0 - float(alpha), 0.0)
