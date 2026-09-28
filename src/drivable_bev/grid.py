# 이관: 인지 코드 레포 d0922bb `src/perception/drivable_bev/src/drivable_bev/grid.py` (원작성: 안승현)
"""Single-source metric and pixel convention for ego-centric BEV layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class BevGridSpec:
    """Rectangular base_link grid with vehicle forward at image top.

    Metric coordinates follow REP-103: x is forward and y is left. OpenCV pixel
    coordinates are (column, row), so positive y moves toward smaller columns
    and positive x moves toward smaller rows.
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    resolution_m: float
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.x_min_m,
                self.x_max_m,
                self.y_min_m,
                self.y_max_m,
                self.resolution_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("BEV grid values must be finite")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("BEV maximum bounds must exceed minimum bounds")
        if self.resolution_m <= 0.0:
            raise ValueError("BEV resolution must be positive")
        if not self.frame_id:
            raise ValueError("BEV frame_id must not be empty")
        for span in (self.x_span_m, self.y_span_m):
            cells = span / self.resolution_m
            if not np.isclose(cells, round(cells), atol=1e-8):
                raise ValueError("BEV spans must be divisible by resolution")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BevGridSpec":
        return cls(
            x_min_m=float(value["x_min_m"]),
            x_max_m=float(value["x_max_m"]),
            y_min_m=float(value["y_min_m"]),
            y_max_m=float(value["y_max_m"]),
            resolution_m=float(value["resolution_m"]),
            frame_id=str(value.get("frame_id", "base_link")),
        )

    @property
    def x_span_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def y_span_m(self) -> float:
        return self.y_max_m - self.y_min_m

    @property
    def height_px(self) -> int:
        return int(round(self.x_span_m / self.resolution_m))

    @property
    def width_px(self) -> int:
        return int(round(self.y_span_m / self.resolution_m))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height_px, self.width_px

    @property
    def ego_pixel(self) -> Tuple[float, float]:
        """Return the rear-axle origin as OpenCV (column, row)."""
        return (
            self.y_max_m / self.resolution_m,
            self.x_max_m / self.resolution_m,
        )

    def metric_to_pixel(self, points_xy: np.ndarray) -> np.ndarray:
        """Convert base_link points (x forward, y left) to (column, row)."""
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        columns = (self.y_max_m - points[:, 1]) / self.resolution_m
        rows = (self.x_max_m - points[:, 0]) / self.resolution_m
        return np.stack([columns, rows], axis=1)

    def pixel_to_metric(self, points_cr: np.ndarray) -> np.ndarray:
        """Convert OpenCV pixels (column, row) to base_link metric points."""
        points = np.asarray(points_cr, dtype=np.float64).reshape(-1, 2)
        x = self.x_max_m - points[:, 1] * self.resolution_m
        y = self.y_max_m - points[:, 0] * self.resolution_m
        return np.stack([x, y], axis=1)

    def bev_pixel_to_ground_homography(self) -> np.ndarray:
        """Return [col,row,1] -> [base x,base y,1]."""
        return np.asarray(
            [
                [0.0, -self.resolution_m, self.x_max_m],
                [-self.resolution_m, 0.0, self.y_max_m],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
