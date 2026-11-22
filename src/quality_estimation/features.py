"""Versioned 56-column feature vector; this module never consumes GT."""
from __future__ import annotations

import cv2
import numpy as np

from camera_semantic_perception.geometry import LetterboxGeometry, compute_letterbox_geometry

REGIONS = ("global",) + tuple(f"r{r}c{c}" for r in range(1, 4) for c in range(1, 3))
STATISTICS = (
    "lane_confidence", "marking_entropy", "lane_area_fraction", "lane_components",
    "brightness_mean", "dark_fraction", "brightness_std", "edge_density",
)
FEATURE_NAMES = tuple(f"{region}.{stat}" for region in REGIONS for stat in STATISTICS)


def grid_regions(height: int, width: int):
    """Global then row-major 3x2 half-open rectangles, floor(k*size/n)."""
    if height < 3 or width < 2:
        raise ValueError("image is too small for a nonempty 3x2 grid")
    boxes = [(0, 0, width, height)]
    boxes.extend((c * width // 2, r * height // 3,
                  (c + 1) * width // 2, (r + 1) * height // 3)
                 for r in range(3) for c in range(2))
    return tuple(zip(REGIONS, boxes))


def geometric_valid(geometry: LetterboxGeometry):
    """Validity derived solely from input dimensions, never annotation masks."""
    expected = compute_letterbox_geometry(geometry.original_hw, geometry.target_hw)
    if geometry != expected:
        raise ValueError("geometry differs from the canonical letterbox transform")
    mask = np.zeros(geometry.target_hw, dtype=bool)
    left, top, _, _ = geometry.pad_ltrb
    h, w = geometry.resized_hw
    mask[top:top+h, left:left+w] = True
    return mask


def feature_schema():
    return {
        "schema_version": "ml-quality-xy-2.0.0",
        "feature_count": len(FEATURE_NAMES), "ordered_features": list(FEATURE_NAMES),
        "regions": {"rows": 3, "cols": 2, "order": list(REGIONS),
                    "bounds": "half-open floor(k*size/n); no overlap; no padding",
                    "output_space": "resized image crop inside model letterbox",
                    "rgb_space": "original RGB; same normalized grid, independently rounded"},
        "statistics": {
            "lane_confidence": "mean(P(white)+P(yellow)) on argmax white/yellow pixels; 0 when none",
            "marking_entropy": "mean(-sum_c p_c*ln(p_c)/ln(4)) on all image pixels of region; 0*ln(0)=0",
            "lane_area_fraction": "argmax white/yellow count / region pixel count",
            "lane_components": "8-connected foreground components after cropping region; background excluded; no size filter",
            "brightness_mean": "mean(OpenCV RGB2GRAY uint8) / 255",
            "dark_fraction": "fraction of original grayscale uint8 pixels strictly below 40",
            "brightness_std": "population std(original grayscale uint8) / 255",
            "edge_density": "fraction nonzero in original full-image Canny(100,200,apertureSize=3,L2gradient=True); then crop",
        },
        "feature_dtype": "float32", "standardization": "none; fit future transforms on training groups only",
        "forbidden_inputs": ["GT", "IoU", "GT_valid", "GT_support", "weather", "time", "pose", "run_id"],
        "target": {"name": "lane_iou", "count": 1, "scope": "full front image in model space, padding excluded",
                   "prediction": "isin(argmax(4-class probabilities), [1,2])",
                   "ground_truth": "isin(road_marking class IDs, [1,2])",
                   "validity": "geometric_valid AND annotation_valid (labels only)",
                   "formula": "TP/(TP+FP+FN)", "empty_union": "JSON null / NPY NaN; label_defined=false",
                   "stopline": "not a lane; predicting lane on stopline counts as FP"},
    }


def extract_features(rgb: np.ndarray, probabilities: np.ndarray,
                     geometry: LetterboxGeometry) -> np.ndarray:
    """Return one raw X row, using only original RGB and frozen model output."""
    if rgb.dtype != np.uint8 or rgb.shape != (*geometry.original_hw, 3):
        raise ValueError("rgb must be original HxWx3 uint8 RGB")
    if probabilities.dtype != np.float32 or probabilities.shape != (4, *geometry.target_hw):
        raise ValueError("probabilities must be 4xHxW float32 in model space")
    if (not np.isfinite(probabilities).all() or probabilities.min() < 0
            or probabilities.max() > 1
            or not np.allclose(probabilities.sum(axis=0), 1, atol=2e-6, rtol=0)):
        raise ValueError("invalid categorical probabilities")
    geometric_valid(geometry)
    left, top, _, _ = geometry.pad_ltrb
    h, w = geometry.resized_hw
    p = probabilities[:, top:top+h, left:left+w]
    classes = p.argmax(axis=0)
    lane = (classes == 1) | (classes == 2)
    lane_probability = p[1] + p[2]
    # Clamp solely for log; zero-probability terms still have zero weight.
    entropy = -(p * np.log(np.maximum(p, np.finfo(np.float32).tiny))).sum(axis=0) / np.log(4)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3, L2gradient=True) != 0
    values = []
    for (_, (x0, y0, x1, y1)), (_, (a0, b0, a1, b1)) in zip(
            grid_regions(h, w), grid_regions(*geometry.original_hw)):
        foreground = lane[y0:y1, x0:x1]
        image = gray[b0:b1, a0:a1]
        n_components = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)[0] - 1
        values.extend((
            float(lane_probability[y0:y1, x0:x1][foreground].mean()) if foreground.any() else 0.0,
            float(entropy[y0:y1, x0:x1].mean()), float(foreground.mean()), float(n_components),
            float(image.mean() / 255), float((image < 40).mean()), float(image.std(ddof=0) / 255),
            float(edges[b0:b1, a0:a1].mean()),
        ))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("non-finite or incomplete features")
    return result
