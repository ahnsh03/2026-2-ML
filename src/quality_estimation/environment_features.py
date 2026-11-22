"""Experimental front-camera RGB descriptors, independent of labels and perception output."""
from __future__ import annotations

import cv2
import numpy as np


# Half-open pixel bounds reviewed on the fixed 1280x720 front-camera profile.
# The left and right context may contain objects, vegetation, sky, or pavement.
PROFILE = "front-1280x720-environment-patches-v1"
PATCHES = {
    "sky": (360, 45, 920, 170),
    "road_near": (425, 470, 820, 556),
    "road_far": (550, 364, 730, 396),
    "left_context": (40, 295, 260, 375),
    "right_context": (1020, 295, 1240, 375),
    "hood_left": (470, 609, 592, 637),
    "hood_right": (688, 609, 810, 637),
}
STATISTICS = (
    "mean", "dark_fraction", "std", "edge_density", "median", "p90_minus_p10",
    "bright_fraction", "saturation_median", "chroma_r_minus_b", "median_filter_residual",
)
CROSS_PATCH_NAMES = ("hood_minus_sky", "near_minus_far", "hood_left_minus_right")
NAMES = tuple(f"{region}.{stat}" for region in PATCHES for stat in STATISTICS) + CROSS_PATCH_NAMES


def extract_environment(rgb: np.ndarray) -> np.ndarray:
    """Return patch statistics on the canonical field of view; caller verifies camera profile."""
    if rgb.dtype != np.uint8 or rgb.shape != (720, 1280, 3):
        raise ValueError("expected original uint8 720x1280 front RGB")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200, L2gradient=True)
    median = cv2.medianBlur(gray, 3)
    medians = {}
    values = []
    for name, (x0, y0, x1, y1) in PATCHES.items():
        region = np.s_[y0:y1, x0:x1]
        g = gray[region]
        color = rgb[region]
        q10, q50, q90 = np.percentile(g, (10, 50, 90)) / 255.
        medians[name] = q50
        hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        rb = color[:, :, 0].astype(np.int16) - color[:, :, 2]
        values.extend((
            g.mean()/255., (g < 40).mean(), g.std()/255., (edges[region] > 0).mean(),
            q50, q90-q10, (g >= 245).mean(), np.median(hsv[:, :, 1])/255.,
            np.median(rb)/255., cv2.absdiff(g, median[region]).mean()/255.,
        ))
    hood = (medians["hood_left"] + medians["hood_right"])/2
    values.extend((hood-medians["sky"], medians["road_near"]-medians["road_far"],
                   medians["hood_left"]-medians["hood_right"]))
    return np.asarray(values, dtype=np.float32)
