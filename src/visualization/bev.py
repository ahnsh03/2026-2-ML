"""인지 코드 레포 ``drivable_bev``의 지면 투영·3카메라 융합을 오프라인 프레임에 적용한다.

좌표계는 ``drivable_bev`` 규약을 따른다: base_link = 후륜축 중심, x 전방 +, y 좌측 +,
노면 z = -0.35 m. BEV 화면은 위가 전방, 왼쪽이 차량 좌측이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

from .camera import DRIVABLE_BGR, LANE_BGR, overlay_masks

GRID_FILES = {"pilot": "bev_grid.yaml", "dev": "bev_grid_dev.yaml"}

# 차량(Ioniq 5) 외곽 [m], base_link(후륜축 중심) 기준
VEHICLE_X_RANGE_M = (-0.79, 3.845)  # 리어 오버행 0.79 / 축거 3.0 + 프론트 오버행 0.845
VEHICLE_HALF_WIDTH_M = 0.946  # 전폭 1.892 / 2


@dataclass(frozen=True)
class BevLayers:
    """융합된 BEV 레이어 (모두 uint8, grid 해상도)."""

    drivable_probability: np.ndarray
    lane_mask: np.ndarray  # {0,255}
    lane_confidence: np.ndarray
    coverage: np.ndarray  # {0,255}
    source_count: np.ndarray


class BevScene:
    def __init__(self, perception_repo: Path, grid_name: str = "dev", views: Sequence[str] = ("front", "left", "right")) -> None:
        from drivable_bev.calibration import load_calibration_snapshot, validate_sensor_set_snapshot
        from drivable_bev.fusion import CameraBevFusion, build_quality_maps
        from drivable_bev.grid import BevGridSpec
        from drivable_bev.projector import CameraBevProjector

        if grid_name not in GRID_FILES:
            raise ValueError("grid must be one of {}".format(sorted(GRID_FILES)))
        config_dir = Path(perception_repo) / "src" / "perception" / "drivable_bev" / "config"
        self.calibration_path = config_dir / "cameras_v2.yaml"
        snapshot, cameras = load_calibration_snapshot(self.calibration_path)
        self.calibration_warning = None
        try:
            validate_sensor_set_snapshot(snapshot, cameras, Path(perception_repo) / str(snapshot["source_sensor_set"]))
        except ValueError as exc:  # 시각화는 계속하되 기록한다
            self.calibration_warning = str(exc)

        self.grid_name = grid_name
        self.grid_path = config_dir / GRID_FILES[grid_name]
        grid_doc = yaml.safe_load(self.grid_path.read_text(encoding="utf-8"))
        fusion_doc = yaml.safe_load((config_dir / "fusion.yaml").read_text(encoding="utf-8"))
        self.grid = BevGridSpec.from_mapping(grid_doc)
        self.views = tuple(views)
        self.projectors = {
            view: CameraBevProjector(
                cameras[view],
                self.grid,
                max_ground_range_m=float(grid_doc["max_ground_range_m"]),
                min_camera_depth_m=float(grid_doc["min_camera_depth_m"]),
            )
            for view in self.views
        }
        quality_keys = (
            "edge_taper_fraction",
            "forward_fade_start_m",
            "lateral_fade_start_m",
            "rear_fade_start_m",
            "boundary_weight",
        )
        self.quality = build_quality_maps(
            self.projectors, self.grid, **{key: float(fusion_doc[key]) for key in quality_keys}
        )
        self.fusion = CameraBevFusion(
            self.quality, view_ids={view: int(fusion_doc["view_ids"][view]) for view in self.views}
        )
        self.ground_z_m = float(snapshot["ground_plane"]["z_at_origin_m"])

    # ------------------------------------------------------------------ 투영·융합
    def fuse_rgb(self, bgr_by_view: Mapping[str, np.ndarray]) -> np.ndarray:
        from drivable_bev.rgb_bev import fuse_rgb_views, warp_rgb_to_bev

        warped = {
            view: warp_rgb_to_bev(
                np.ascontiguousarray(bgr_by_view[view][:, :, :3]),
                self.projectors[view].homography.bev_from_image,
                self.grid.shape,
            )
            for view in self.views
        }
        fused, _ = fuse_rgb_views(warped, self.quality)
        return fused

    def fuse_semantics(
        self,
        drivable_u8_by_view: Mapping[str, np.ndarray],
        lane_mask_by_view: Mapping[str, np.ndarray],
        lane_confidence_by_view: Optional[Mapping[str, np.ndarray]] = None,
    ) -> BevLayers:
        """카메라별 원본 해상도 마스크를 투영해 융합한다.

        차선은 road-marking class 1로 넣어 ``drivable_bev`` 융합 규칙(경계는 기하 품질이 큰 뷰가 소유)을 그대로 쓴다.
        """
        projected = {}
        for view in self.views:
            lane = (np.asarray(lane_mask_by_view[view]) > 0).astype(np.uint8)
            if lane_confidence_by_view is None:
                confidence = lane * np.uint8(255)
            else:
                confidence = np.where(lane > 0, lane_confidence_by_view[view], 0).astype(np.uint8)
            projected[view] = self.projectors[view].project(
                np.ascontiguousarray(drivable_u8_by_view[view], dtype=np.uint8),
                np.ascontiguousarray(lane),
                np.ascontiguousarray(confidence),
            )
        fused = self.fusion.fuse(projected)
        return BevLayers(
            drivable_probability=fused.drivable_probability,
            lane_mask=(fused.road_marking_class_id > 0).astype(np.uint8) * 255,
            lane_confidence=fused.road_marking_confidence,
            coverage=fused.coverage,
            source_count=fused.source_count,
        )

    # ------------------------------------------------------------------ 렌더링
    def render_semantic(self, layers: BevLayers, scale: int) -> np.ndarray:
        """검은 배경에 카메라 범위(회색), 주행가능영역(초록), 차선(노랑)."""
        canvas = np.zeros((*self.grid.shape, 3), dtype=np.uint8)
        covered = layers.coverage > 0
        canvas[covered] = (45, 45, 45)
        alpha = (layers.drivable_probability.astype(np.float32) / 255.0)[covered][:, None]
        green = np.asarray(DRIVABLE_BGR, dtype=np.float32)
        canvas[covered] = np.clip(
            canvas[covered].astype(np.float32) * (1.0 - alpha) + green * alpha, 0, 255
        ).astype(np.uint8)
        canvas[layers.lane_mask > 0] = LANE_BGR
        return self.decorate(self.upscale(canvas, scale, cv2.INTER_NEAREST), scale)

    def render_rgb(self, rgb_bev: np.ndarray, scale: int) -> np.ndarray:
        return self.decorate(self.upscale(rgb_bev, scale, cv2.INTER_LINEAR), scale)

    def render_overlay(self, rgb_bev: np.ndarray, layers: BevLayers, scale: int) -> np.ndarray:
        base = self.upscale(rgb_bev, scale, cv2.INTER_LINEAR)
        drivable = self.upscale((layers.drivable_probability >= 128).astype(np.uint8), scale, cv2.INTER_NEAREST)
        lane = self.upscale((layers.lane_mask > 0).astype(np.uint8), scale, cv2.INTER_NEAREST)
        return self.decorate(overlay_masks(base, drivable, lane), scale)

    @staticmethod
    def upscale(image: np.ndarray, scale: int, interpolation) -> np.ndarray:
        if scale == 1:
            return np.ascontiguousarray(image).copy()
        height, width = image.shape[:2]
        return cv2.resize(image, (width * scale, height * scale), interpolation=interpolation)

    def _pixel(self, x_m: float, y_m: float, scale: int):
        column = (self.grid.y_max_m - y_m) / self.grid.resolution_m * scale
        row = (self.grid.x_max_m - x_m) / self.grid.resolution_m * scale
        return int(round(column)), int(round(row))

    def decorate(self, canvas: np.ndarray, scale: int) -> np.ndarray:
        """5 m 격자, 거리 눈금, 차량 외곽선을 그린다."""
        height, width = canvas.shape[:2]
        grid_color = (110, 110, 110)
        step = 5.0
        x = np.ceil(self.grid.x_min_m / step) * step
        while x <= self.grid.x_max_m + 1e-6:
            _, row = self._pixel(x, 0.0, scale)
            if 0 <= row < height:
                cv2.line(canvas, (0, row), (width - 1, row), grid_color, 1, cv2.LINE_AA)
                if x > 0:
                    cv2.putText(canvas, "{:.0f}m".format(x), (4, max(12, row - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (230, 230, 230), 1, cv2.LINE_AA)
            x += step
        y = np.ceil(self.grid.y_min_m / step) * step
        while y <= self.grid.y_max_m + 1e-6:
            column, _ = self._pixel(0.0, y, scale)
            if 0 <= column < width:
                cv2.line(canvas, (column, 0), (column, height - 1), grid_color, 1, cv2.LINE_AA)
            y += step
        x0, x1 = VEHICLE_X_RANGE_M
        half = VEHICLE_HALF_WIDTH_M
        corners = np.asarray(
            [self._pixel(x0, half, scale), self._pixel(x1, half, scale),
             self._pixel(x1, -half, scale), self._pixel(x0, -half, scale)],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [corners], (30, 30, 30))
        cv2.polylines(canvas, [corners], True, (255, 255, 255), 2, cv2.LINE_AA)
        nose = np.asarray([self._pixel(x1, 0.0, scale), self._pixel(x1 - 1.0, half * 0.7, scale),
                           self._pixel(x1 - 1.0, -half * 0.7, scale)], dtype=np.int32)
        cv2.fillPoly(canvas, [nose], (255, 255, 255))
        return canvas

    def describe(self) -> Dict[str, object]:
        return {
            "grid": self.grid_name,
            "grid_file": str(self.grid_path),
            "x_range_m": [self.grid.x_min_m, self.grid.x_max_m],
            "y_range_m": [self.grid.y_min_m, self.grid.y_max_m],
            "resolution_m": self.grid.resolution_m,
            "shape_rows_cols": list(self.grid.shape),
            "ground_z_m": self.ground_z_m,
            "calibration_file": str(self.calibration_path),
            "calibration_warning": self.calibration_warning,
        }
