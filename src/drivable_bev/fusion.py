# 이관: 인지 코드 레포 d0922bb `src/perception/drivable_bev/src/drivable_bev/fusion.py` (원작성: 안승현)
"""Deterministic quality-weighted fusion for projected camera semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .grid import BevGridSpec
from .projector import CameraBevProjector, ProjectedSemantic


@dataclass(frozen=True)
class FusedSemantic:
    drivable_probability: np.ndarray
    road_marking_class_id: np.ndarray
    road_marking_confidence: np.ndarray
    coverage: np.ndarray
    source_count: np.ndarray
    source_view: np.ndarray


def timestamp_span_ns(stamps_ns) -> int:
    values = [int(value) for value in stamps_ns]
    if not values or any(value <= 0 for value in values):
        raise ValueError("source timestamps must be positive")
    return max(values) - min(values)


def timestamps_within_slop(stamps_ns, max_skew_ns: int) -> bool:
    if int(max_skew_ns) < 0:
        raise ValueError("max timestamp skew cannot be negative")
    return timestamp_span_ns(stamps_ns) <= int(max_skew_ns)


def build_quality_maps(
    projectors: Mapping[str, CameraBevProjector],
    grid: BevGridSpec,
    edge_taper_fraction: float = 0.12,
    forward_fade_start_m: float = 15.0,
    lateral_fade_start_m: float = 6.0,
    rear_fade_start_m: float = -4.0,
    boundary_weight: float = 0.20,
) -> Mapping[str, np.ndarray]:
    """Build static overlap weights from image support and ground resolution."""
    if not projectors:
        raise ValueError("at least one projector is required")
    if not 0.0 < edge_taper_fraction < 0.5:
        raise ValueError("edge_taper_fraction must lie in (0,0.5)")
    if not 0.0 <= boundary_weight <= 1.0:
        raise ValueError("boundary_weight must lie in [0,1]")

    positive_density = np.concatenate(
        [
            projector.homography.ground_sampling_density[
                projector.homography.ground_sampling_density > 0.0
            ]
            for projector in projectors.values()
        ]
    )
    density_scale = float(np.percentile(np.sqrt(positive_density), 95.0))
    density_scale = max(density_scale, np.finfo(np.float32).eps)

    rows, columns = np.indices(grid.shape, dtype=np.float64)
    ground = grid.pixel_to_metric(
        np.column_stack([columns.reshape(-1), rows.reshape(-1)])
    ).reshape((*grid.shape, 2))
    x = ground[:, :, 0]
    abs_y = np.abs(ground[:, :, 1])

    def fade(value, start, end):
        if end <= start:
            return np.ones_like(value, dtype=np.float64)
        unit = np.clip((end - value) / (end - start), 0.0, 1.0)
        return boundary_weight + (1.0 - boundary_weight) * unit

    range_weight = fade(x, forward_fade_start_m, grid.x_max_m)
    range_weight *= fade(abs_y, lateral_fade_start_m, max(abs(grid.y_min_m), grid.y_max_m))
    rear_unit = np.clip(
        (x - grid.x_min_m) / max(1e-6, rear_fade_start_m - grid.x_min_m),
        0.0,
        1.0,
    )
    range_weight *= np.where(
        x < rear_fade_start_m,
        boundary_weight + (1.0 - boundary_weight) * rear_unit,
        1.0,
    )

    result = {}
    for view, projector in projectors.items():
        model = projector.homography
        camera = projector.camera
        pixels = model.source_pixels
        u = pixels[:, :, 0]
        v = pixels[:, :, 1]
        edge_distance = np.minimum.reduce(
            [u, camera.width - 1.0 - u, v, camera.height - 1.0 - v]
        )
        edge_scale = edge_taper_fraction * min(camera.width, camera.height)
        edge_weight = np.clip(edge_distance / max(edge_scale, 1e-6), 0.0, 1.0)
        density_weight = np.clip(
            np.sqrt(model.ground_sampling_density) / density_scale,
            0.05,
            1.0,
        )
        quality = edge_weight * density_weight * range_weight
        quality[model.coverage == 0] = 0.0
        result[str(view)] = np.ascontiguousarray(quality, dtype=np.float32)
    return result


class CameraBevFusion:
    """Fuse aligned per-view rasters without ever averaging categorical IDs."""

    def __init__(
        self,
        quality_by_view: Mapping[str, np.ndarray],
        view_ids: Optional[Mapping[str, int]] = None,
    ) -> None:
        if not quality_by_view:
            raise ValueError("at least one quality map is required")
        self.views = tuple(str(view) for view in quality_by_view)
        self.quality_by_view = {
            str(view): self._quality(value, str(view))
            for view, value in quality_by_view.items()
        }
        shapes = {value.shape for value in self.quality_by_view.values()}
        if len(shapes) != 1:
            raise ValueError("all quality maps must have the same shape")
        self.shape = next(iter(shapes))
        configured_ids = view_ids or {view: index + 1 for index, view in enumerate(self.views)}
        if set(configured_ids) != set(self.views):
            raise ValueError("view_ids must cover every fusion view")
        ids = [int(configured_ids[view]) for view in self.views]
        if any(value <= 0 or value > 255 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("view IDs must be unique uint8 values greater than zero")
        self.view_ids = np.asarray(ids, dtype=np.uint8)

    def fuse(
        self,
        projected_by_view: Mapping[str, ProjectedSemantic],
        valid_by_view: Optional[Mapping[str, np.ndarray]] = None,
    ) -> FusedSemantic:
        if set(projected_by_view) != set(self.views):
            raise ValueError("projected views differ from configured fusion views")
        if valid_by_view is not None and set(valid_by_view) != set(self.views):
            raise ValueError("valid masks differ from configured fusion views")

        drivable = []
        marking = []
        confidence = []
        weights = []
        for view in self.views:
            value = projected_by_view[view]
            drivable.append(self._mono8(value.drivable_probability, "drivable"))
            classes = self._mono8(value.road_marking_class_id, "road marking")
            if np.any(classes > 3):
                raise ValueError("road-marking class IDs must lie in [0,3]")
            marking.append(classes)
            confidence.append(
                self._mono8(value.road_marking_confidence, "road marking confidence")
            )
            weight = self.quality_by_view[view].copy()
            if valid_by_view is not None:
                valid = np.asarray(valid_by_view[view])
                if valid.shape != self.shape:
                    raise ValueError("valid mask shape differs from fusion grid")
                weight[valid <= 0] = 0.0
            weights.append(weight)

        drivable_stack = np.stack(drivable).astype(np.float32)
        marking_stack = np.stack(marking)
        confidence_stack = np.stack(confidence).astype(np.float32)
        weight_stack = np.stack(weights)
        weight_sum = weight_stack.sum(axis=0)
        valid_union = weight_sum > 0.0

        fused_drivable = np.zeros(self.shape, dtype=np.uint8)
        weighted_drivable = (drivable_stack * weight_stack).sum(axis=0)
        fused_drivable[valid_union] = np.rint(
            weighted_drivable[valid_union] / weight_sum[valid_union]
        ).astype(np.uint8)

        # Geometry owns the seam; confidence refines ties without allowing an
        # overconfident, poorly resolved edge pixel to dominate a central view.
        owner_score = weight_stack * (0.5 + 0.5 * confidence_stack / 255.0)
        owner = np.argmax(owner_score, axis=0)
        gather = owner[None, :, :]
        fused_marking = np.take_along_axis(marking_stack, gather, axis=0)[0]
        fused_confidence = np.take_along_axis(
            confidence_stack.astype(np.uint8), gather, axis=0
        )[0]
        source_view = self.view_ids[owner]
        fused_marking[~valid_union] = 0
        fused_confidence[~valid_union] = 0
        source_view[~valid_union] = 0

        source_count = np.count_nonzero(weight_stack > 0.0, axis=0).astype(np.uint8)
        coverage = (valid_union.astype(np.uint8) * 255)
        return FusedSemantic(
            drivable_probability=np.ascontiguousarray(fused_drivable),
            road_marking_class_id=np.ascontiguousarray(fused_marking),
            road_marking_confidence=np.ascontiguousarray(fused_confidence),
            coverage=np.ascontiguousarray(coverage),
            source_count=np.ascontiguousarray(source_count),
            source_view=np.ascontiguousarray(source_view),
        )

    def _mono8(self, value, name):
        array = np.asarray(value)
        if array.shape != self.shape or array.dtype != np.uint8:
            raise ValueError("{} must be uint8 {}".format(name, self.shape))
        return array

    @staticmethod
    def _quality(value, view):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("{} quality map must be a finite non-negative raster".format(view))
        return np.ascontiguousarray(array)
