# 이관: 인지 코드 레포 d0922bb `src/perception/drivable_bev/src/drivable_bev/homography.py` (원작성: 안승현)
"""Ground-plane homographies and geometric validity masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CameraCalibration
from .grid import BevGridSpec


@dataclass(frozen=True)
class HomographyModel:
    image_from_ground: np.ndarray
    image_from_bev: np.ndarray
    bev_from_image: np.ndarray
    coverage: np.ndarray
    source_pixels: np.ndarray
    ground_sampling_density: np.ndarray


def build_homography(
    camera: CameraCalibration,
    grid: BevGridSpec,
    max_ground_range_m: float,
    min_camera_depth_m: float,
) -> HomographyModel:
    if max_ground_range_m <= 0.0:
        raise ValueError("max_ground_range_m must be positive")
    if min_camera_depth_m <= 0.0:
        raise ValueError("min_camera_depth_m must be positive")

    image_from_ground = camera.ground_to_image_homography
    ground_from_bev = grid.bev_pixel_to_ground_homography()
    image_from_bev = image_from_ground @ ground_from_bev
    if abs(np.linalg.det(image_from_bev)) < 1e-12:
        raise ValueError("camera-to-BEV homography is singular")
    bev_from_image = np.linalg.inv(image_from_bev)
    bev_from_image /= bev_from_image[2, 2]

    rows, columns = np.indices(grid.shape, dtype=np.float64)
    bev_pixels = np.column_stack(
        [columns.reshape(-1), rows.reshape(-1), np.ones(rows.size)]
    )
    ground_h = (ground_from_bev @ bev_pixels.T).T
    ground_xy = ground_h[:, :2] / ground_h[:, 2:3]
    xyz = camera.ground_plane.xyz(ground_xy)
    optical = camera.ground_to_optical(xyz)
    image_h = (camera.intrinsic @ optical.T).T

    positive_depth = optical[:, 2] > min_camera_depth_m
    pixels = np.full((len(ground_xy), 2), np.nan, dtype=np.float64)
    pixels[positive_depth] = (
        image_h[positive_depth, :2] / image_h[positive_depth, 2:3]
    )
    in_image = (
        positive_depth
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= camera.width - 1.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= camera.height - 1.0)
    )
    in_range = np.linalg.norm(ground_xy, axis=1) <= max_ground_range_m
    coverage = ((in_image & in_range).reshape(grid.shape).astype(np.uint8) * 255)

    # Absolute image-pixel area per square metre on the ground plane.  It is a
    # geometric quality cue for overlap seams: a view that resolves the same
    # patch with more source pixels should normally own that BEV cell.
    h = image_from_ground
    x = ground_xy[:, 0]
    y = ground_xy[:, 1]
    denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    numerator_u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    numerator_v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    safe = np.abs(denominator) > 1e-9
    density = np.zeros(len(ground_xy), dtype=np.float64)
    if np.any(safe):
        d2 = denominator[safe] ** 2
        du_dx = (h[0, 0] * denominator[safe] - numerator_u[safe] * h[2, 0]) / d2
        du_dy = (h[0, 1] * denominator[safe] - numerator_u[safe] * h[2, 1]) / d2
        dv_dx = (h[1, 0] * denominator[safe] - numerator_v[safe] * h[2, 0]) / d2
        dv_dy = (h[1, 1] * denominator[safe] - numerator_v[safe] * h[2, 1]) / d2
        density[safe] = np.abs(du_dx * dv_dy - du_dy * dv_dx)
    density = density.reshape(grid.shape)
    density[coverage == 0] = 0.0

    return HomographyModel(
        image_from_ground=image_from_ground,
        image_from_bev=image_from_bev,
        bev_from_image=bev_from_image,
        coverage=coverage,
        source_pixels=pixels.reshape((*grid.shape, 2)),
        ground_sampling_density=density,
    )
