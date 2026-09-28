# 이관: 인지 코드 레포 d0922bb `src/perception/drivable_bev/src/drivable_bev/projector.py` (원작성: 안승현)
"""Cached camera semantic projection onto an ego-centric grid."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CameraCalibration
from .grid import BevGridSpec
from .homography import HomographyModel, build_homography


@dataclass(frozen=True)
class ProjectedSemantic:
    drivable_probability: np.ndarray
    road_marking_class_id: np.ndarray
    road_marking_confidence: np.ndarray
    coverage: np.ndarray


class CameraBevProjector:
    def __init__(
        self,
        camera: CameraCalibration,
        grid: BevGridSpec,
        max_ground_range_m: float = 60.0,
        min_camera_depth_m: float = 0.1,
    ) -> None:
        self.camera = camera
        self.grid = grid
        self.homography: HomographyModel = build_homography(
            camera, grid, max_ground_range_m, min_camera_depth_m
        )

    def project(
        self,
        drivable_probability: np.ndarray,
        road_marking_class_id: np.ndarray,
        road_marking_confidence: np.ndarray = None,
    ) -> ProjectedSemantic:
        drivable = self._validate_source(drivable_probability, "drivable")
        marking = self._validate_source(road_marking_class_id, "road marking")
        confidence = (
            np.full(marking.shape, 255, dtype=np.uint8)
            if road_marking_confidence is None
            else self._validate_source(road_marking_confidence, "road marking confidence")
        )
        if np.any(marking > 3):
            raise ValueError("road-marking class IDs must be in [0,3]")

        size = (self.grid.width_px, self.grid.height_px)
        projected_drivable = cv2.warpPerspective(
            drivable,
            self.homography.bev_from_image,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        projected_marking = cv2.warpPerspective(
            marking,
            self.homography.bev_from_image,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        projected_confidence = cv2.warpPerspective(
            confidence,
            self.homography.bev_from_image,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = self.homography.coverage > 0
        projected_drivable[~valid] = 0
        projected_marking[~valid] = 0
        projected_confidence[~valid] = 0
        return ProjectedSemantic(
            drivable_probability=np.ascontiguousarray(projected_drivable),
            road_marking_class_id=np.ascontiguousarray(projected_marking),
            road_marking_confidence=np.ascontiguousarray(projected_confidence),
            coverage=self.homography.coverage.copy(),
        )

    def _validate_source(self, value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value)
        expected = (self.camera.height, self.camera.width)
        if array.shape != expected:
            raise ValueError(
                "{} image must have shape {}, got {}".format(
                    name, expected, array.shape
                )
            )
        if array.dtype != np.uint8:
            raise ValueError("{} image must use uint8/mono8".format(name))
        return np.ascontiguousarray(array)
