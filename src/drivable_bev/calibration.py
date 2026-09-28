# 이관: 인지 코드 레포 d0922bb `src/perception/drivable_bev/src/drivable_bev/calibration.py` (원작성: 안승현)
"""MORAI camera calibration parsing and sensor-set drift validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import yaml


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


@dataclass(frozen=True)
class GroundPlane:
    """Planar road surface expressed in ``base_link`` coordinates.

    The MORAI vehicle reference is at the rear-axle centre, not on the road
    surface.  Keeping that vertical offset explicit prevents camera mounts
    from being silently reinterpreted as heights above the road.
    """

    frame_id: str
    z_at_origin_m: float
    dz_dx: float
    dz_dy: float
    source: str
    model: str = "plane"

    def __post_init__(self) -> None:
        if self.frame_id != "base_link":
            raise ValueError("ground plane frame_id must be base_link")
        if self.model != "plane":
            raise ValueError("ground plane model must be plane")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if not np.isfinite(coefficients).all():
            raise ValueError("ground plane coefficients must be finite")
        if not self.source:
            raise ValueError("ground plane source must not be empty")

    @property
    def coefficients(self) -> Tuple[float, float, float]:
        """Return ``(a, b, c)`` for ``z = a*x + b*y + c``."""
        return (float(self.dz_dx), float(self.dz_dy), float(self.z_at_origin_m))

    def xyz(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        a, b, c = self.coefficients
        z = a * points[:, 0] + b * points[:, 1] + c
        return np.column_stack([points, z])

    def to_mapping(self) -> Dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "model": self.model,
            "z_at_origin_m": float(self.z_at_origin_m),
            "dz_dx": float(self.dz_dx),
            "dz_dy": float(self.dz_dy),
            "source": self.source,
        }


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    sensor_id: int
    width: int
    height: int
    horizontal_fov_deg: float
    translation_m: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    image_topic: str
    drivable_topic: str
    road_marking_topic: str
    road_marking_confidence_topic: str
    ground_plane: GroundPlane

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name must not be empty")
        if self.sensor_id <= 0:
            raise ValueError("camera sensor_id must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("horizontal FOV must lie between 0 and 180 degrees")
        pose = np.asarray((*self.translation_m, *self.rotation_deg), dtype=np.float64)
        if not np.isfinite(pose).all():
            raise ValueError("camera pose must be finite")

    @property
    def intrinsic(self) -> np.ndarray:
        focal = (0.5 * self.width) / math.tan(
            math.radians(0.5 * self.horizontal_fov_deg)
        )
        return np.asarray(
            [
                [focal, 0.0, 0.5 * self.width],
                [0.0, focal, 0.5 * self.height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def body_from_mount_rotation(self) -> np.ndarray:
        """MORAI mount to body rotation; positive pitch points the lens down."""
        roll, pitch, yaw = np.radians(np.asarray(self.rotation_deg, dtype=np.float64))
        return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)

    @property
    def optical_from_body_rotation(self) -> np.ndarray:
        mount_to_optical = np.asarray(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        return mount_to_optical @ self.body_from_mount_rotation.T

    @property
    def optical_from_body_translation(self) -> np.ndarray:
        translation = np.asarray(self.translation_m, dtype=np.float64)
        return -self.optical_from_body_rotation @ translation

    @property
    def ground_to_image_homography(self) -> np.ndarray:
        """Map base_link ground [x,y,1] to image [u,v,w]."""
        rotation = self.optical_from_body_rotation
        translation = self.optical_from_body_translation
        dz_dx, dz_dy, z_at_origin = self.ground_plane.coefficients
        ground_extrinsic = np.column_stack(
            [
                rotation[:, 0] + dz_dx * rotation[:, 2],
                rotation[:, 1] + dz_dy * rotation[:, 2],
                translation + z_at_origin * rotation[:, 2],
            ]
        )
        return self.intrinsic @ ground_extrinsic

    def ground_to_optical(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        return (
            self.optical_from_body_rotation @ points.T
            + self.optical_from_body_translation.reshape(3, 1)
        ).T

    def project_ground(self, points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        xyz = self.ground_plane.xyz(points)
        optical = self.ground_to_optical(xyz)
        homogeneous = (self.intrinsic @ optical.T).T
        pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid_depth = optical[:, 2] > 0.0
        pixels[valid_depth] = (
            homogeneous[valid_depth, :2]
            / homogeneous[valid_depth, 2:3]
        )
        return pixels, optical[:, 2]


def ground_plane_from_dict(value: Mapping[str, object]) -> GroundPlane:
    if not isinstance(value, Mapping):
        raise ValueError("ground_plane must be a mapping")
    required = ("frame_id", "model", "z_at_origin_m", "dz_dx", "dz_dy", "source")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError("ground_plane is missing fields: {}".format(missing))
    return GroundPlane(
        frame_id=str(value["frame_id"]),
        model=str(value["model"]),
        z_at_origin_m=float(value["z_at_origin_m"]),
        dz_dx=float(value["dz_dx"]),
        dz_dy=float(value["dz_dy"]),
        source=str(value["source"]),
    )


def calibration_from_dict(
    name: str, value: Mapping[str, object], ground_plane: GroundPlane
) -> CameraCalibration:
    topics = value.get("topics", {})
    if not isinstance(topics, Mapping):
        raise ValueError("camera topics must be a mapping")
    return CameraCalibration(
        name=name,
        sensor_id=int(value["sensor_id"]),
        width=int(value["width"]),
        height=int(value["height"]),
        horizontal_fov_deg=float(value["horizontal_fov_deg"]),
        translation_m=tuple(float(x) for x in value["translation_m"]),
        rotation_deg=tuple(float(x) for x in value["rotation_deg"]),
        image_topic=str(topics["image"]),
        drivable_topic=str(topics["drivable"]),
        road_marking_topic=str(topics["road_marking"]),
        road_marking_confidence_topic=str(topics["road_marking_confidence"]),
        ground_plane=ground_plane,
    )


def calibrations_from_dict(document: Mapping[str, object]) -> Dict[str, CameraCalibration]:
    if not isinstance(document, Mapping) or not isinstance(document.get("cameras"), Mapping):
        raise ValueError("calibration snapshot must contain a cameras mapping")
    if "ground_plane" not in document:
        raise ValueError("calibration snapshot must contain ground_plane")
    ground_plane = ground_plane_from_dict(document["ground_plane"])
    cameras = {
        str(name): calibration_from_dict(str(name), value, ground_plane)
        for name, value in document["cameras"].items()
    }
    if not cameras:
        raise ValueError("calibration snapshot contains no cameras")
    return cameras


def load_calibration_snapshot(
    path: Path,
) -> Tuple[Mapping[str, object], Dict[str, CameraCalibration]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    cameras = calibrations_from_dict(document)
    return document, cameras


def _as_float(value: object) -> float:
    return float(str(value))


def validate_sensor_set_snapshot(
    snapshot: Mapping[str, object], cameras: Mapping[str, CameraCalibration], sensor_set_path: Path
) -> None:
    """Fail when the checked-in runtime snapshot drifts from sensor-set v2."""
    path = Path(sensor_set_path)
    payload = path.read_bytes()
    expected_hash = str(snapshot.get("source_sha256", ""))
    actual_hash = hashlib.sha256(payload).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "sensor-set SHA-256 mismatch: expected {} got {}".format(
                expected_hash, actual_hash
            )
        )
    document = json.loads(payload.decode("utf-8"))
    sensor_by_id = {
        int(item["m_SensorUniqueID"]): item for item in document.get("cameraList", [])
    }
    for camera in cameras.values():
        if camera.sensor_id not in sensor_by_id:
            raise ValueError("camera sensor ID {} is missing".format(camera.sensor_id))
        sensor = sensor_by_id[camera.sensor_id]
        cc = sensor["cc"]
        expected = np.asarray(
            [
                camera.width,
                camera.height,
                camera.horizontal_fov_deg,
                *camera.translation_m,
                *camera.rotation_deg,
            ],
            dtype=np.float64,
        )
        actual = np.asarray(
            [
                int(cc["cameraResWidth"]),
                int(cc["cameraResHeight"]),
                float(cc["cameraFOV"]),
                _as_float(sensor["pos"]["x"]),
                _as_float(sensor["pos"]["y"]),
                _as_float(sensor["pos"]["z"]),
                _as_float(sensor["rot"]["roll"]),
                _as_float(sensor["rot"]["pitch"]),
                _as_float(sensor["rot"]["yaw"]),
            ],
            dtype=np.float64,
        )
        # MORAI serializes right yaw as 290 degrees while the runtime snapshot
        # uses its equivalent -70 degrees.
        actual[-1] = ((actual[-1] + 180.0) % 360.0) - 180.0
        expected[-1] = ((expected[-1] + 180.0) % 360.0) - 180.0
        if not np.allclose(actual, expected, atol=1e-6):
            raise ValueError(
                "camera {} snapshot differs from sensor set: {} != {}".format(
                    camera.name, expected.tolist(), actual.tolist()
                )
            )
