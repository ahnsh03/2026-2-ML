"""Experimental descriptors on the entire calibrated front frame and the frozen Medium v6 heads."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .environment_features import extract_environment

PROFILE_DIR = Path(__file__).resolve().parents[2] / "config" / "roi10_front_v1"
REGIONS = tuple(f"{band}_{side}" for band in ("upper", "far", "near")
                for side in ("left", "center", "right")) + ("hood",)
NODE_FEATURES = ("intensity", "chroma", "intensity_std", "dark_channel", "local_contrast",
                 "sharpness", "glare_fraction", "median_residual")
HEAD_FEATURES = ("drivable_fraction", "lane_fraction", "lane_probability", "marking_entropy")
PAIRS = ((9, 1), (9, 4), (9, 7), (1, 4), (4, 7), (0, 3), (2, 5), (3, 6),
         (5, 8), (0, 2), (3, 5), (6, 8))
SKY_INPUTS = ("r", "g", "b", "saturation", "gray", "mean5", "std5", "residual3", "x/320", "y/180")


@dataclass(frozen=True)
class FrontProfile:
    """Fixed complete region map, block ownership and RGB-only pixel classifier."""
    native: np.ndarray
    small: np.ndarray
    blocks: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    sky_label: np.ndarray

    @classmethod
    def load(cls, directory=PROFILE_DIR):
        directory = Path(directory)
        spec = json.loads((directory / "profile.json").read_text())
        if spec["version"] != "roi10-front-v1":
            raise ValueError("unsupported front-camera profile version")
        for filename, key in (("regions.png", "regions_sha256"), ("sky_tree.json", "sky_tree_sha256")):
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != spec[key]:
                raise ValueError("profile checksum mismatch for " + filename)
        native = cv2.imread(str(directory / "regions.png"), cv2.IMREAD_UNCHANGED)
        if (native is None or native.shape != (720, 1280) or native.dtype != np.uint8
                or np.bincount(native.ravel()).tolist() != spec["native_region_pixels"]
                or len(spec["native_region_pixels"]) != len(REGIONS)):
            raise ValueError("invalid region mask")
        tree = json.loads((directory / "sky_tree.json").read_text())
        if tuple(tree["features"]) != SKY_INPUTS:
            raise ValueError("unexpected sky classifier inputs")
        small = cv2.resize(native, (320, 180), interpolation=cv2.INTER_NEAREST)
        blocks = cv2.resize(native, (640, 360), interpolation=cv2.INTER_NEAREST)[2::5, 2::5]
        nodes = tree["nodes"]
        fields = [np.asarray([n[k] for n in nodes]) for k in ("left", "right", "feature", "threshold", "sky")]
        # Every internal child follows its parent. This also bounds the traversal depth.
        for i, (left, right, feat, _, _) in enumerate(zip(*fields)):
            if not ((left == right == -1) or (i < left < len(nodes) and i < right < len(nodes)
                                                and 0 <= feat < len(SKY_INPUTS))):
                raise ValueError("invalid sky decision tree")
        return cls(native, small, blocks, *fields)


def _require_rgb(rgb):
    if rgb.dtype != np.uint8 or rgb.shape != (720, 1280, 3):
        raise ValueError("expected original fixed-profile 1280x720 front RGB")


def predicted_sky(thumb, profile):
    """Classify only pixels above the fixed horizon using the frozen depth-eight tree."""
    if thumb.dtype != np.uint8 or thumb.shape != (180, 320, 3):
        raise ValueError("expected 320x180 RGB working image")
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    mean = cv2.blur(gray, (5, 5))
    std = np.sqrt(np.maximum(0, cv2.blur(gray**2, (5, 5)) - mean**2))
    sat = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[:, :, 1]
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    fields = np.stack([*np.moveaxis(thumb / 255., 2, 0), sat / 255., gray, mean, std,
                       cv2.absdiff(gray, cv2.blur(gray, (3, 3))), xx / 320, yy / 180], axis=-1)
    upper = profile.small < 3
    x = fields[upper].astype(np.float32)
    cursor = np.zeros(len(x), dtype=np.int32)
    active = np.arange(len(x))
    while active.size:
        current = cursor[active]
        active = active[profile.left[current] >= 0]
        current = cursor[active]
        left = x[active, profile.feature[current]] <= profile.threshold[current]
        cursor[active] = np.where(left, profile.left[current], profile.right[current])
    mask = np.zeros_like(upper)
    mask[upper] = profile.sky_label[cursor]
    return mask


def _image_fields(thumb):
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    intensity = gray.astype(np.float32) / 255.
    color = thumb.astype(np.int16)
    chroma = (color.max(2) - color.min(2)) / 255.
    kernel = np.ones((3, 3), np.uint8)
    low = cv2.erode(gray, kernel).astype(np.float32)
    high = cv2.dilate(gray, kernel).astype(np.float32)
    contrast = (high - low) / np.maximum(1., high + low)
    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1)
    sharp = np.minimum(1., cv2.magnitude(gx, gy) / 4.)
    dark = cv2.erode(thumb.min(2), np.ones((2, 2), np.uint8)).astype(np.float32) / 255.
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3)).astype(np.float32) / 255.
    return intensity, chroma, dark, contrast, sharp, residual


def _summarize(fields, mask):
    if not mask.any():
        return np.zeros(len(NODE_FEATURES), np.float32)
    intensity, chroma, dark, contrast, sharp, residual = (v[mask] for v in fields)
    return np.asarray([intensity.mean(), chroma.mean(), intensity.std(), np.median(dark),
                       contrast.mean(), sharp.mean(), (intensity >= 245 / 255.).mean(),
                       residual.mean()], dtype=np.float32)


def _region_histograms(rgb, profile):
    half = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(half, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    hsv = cv2.cvtColor(half, cv2.COLOR_RGB2HSV)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    gray_blocks = gray.reshape(72, 5, 128, 5)
    low = gray_blocks.min((1, 3))
    high = gray_blocks.max((1, 3))
    saturation = (hsv[:, :, 1].astype(np.float32) / 255.).reshape(72, 5, 128, 5).mean((1, 3))
    values = (low, (high - low) / np.maximum(1 / 255., high + low),
              np.minimum(1., grad.reshape(72, 5, 128, 5).mean((1, 3)) / 4.),
              hsv[2::5, 2::5, 0].astype(np.float32) / 180., saturation)
    histograms = []
    for index, value in enumerate(values):
        bucket = 10 * profile.blocks.astype(int) + np.minimum(9, (value * 10).astype(int))
        weights = saturation.ravel() if index == 3 else None
        counts = np.bincount(bucket.ravel(), weights=weights, minlength=100).reshape(10, 10)
        histograms.append(counts / np.maximum(counts.sum(1, keepdims=True), 1e-12))
    return np.stack(histograms, axis=1).astype(np.float32)


def _hour_histograms(h):
    return np.concatenate([h[:3, 3:5].ravel(), h[3:6, 0].ravel(), h[3:6, 2].ravel(),
                           h[6:9, 1].ravel(), h[9, 0]])


def native_scene_histograms(rgb, profile):
    """Five native 10x10 block measurements, ten bins per measurement and physical region."""
    _require_rgb(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blocks = gray.reshape(72, 10, 128, 10)
    low = blocks.min((1, 3))
    high = blocks.max((1, 3))
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    channels = (low, (high - low) / np.maximum(1 / 255., high + low),
                np.minimum(1., magnitude.reshape(72, 10, 128, 10).mean((1, 3)) / 4.),
                hsv[5::10, 5::10, 0] / 180., hsv[:, :, 1].reshape(72, 10, 128, 10).mean((1, 3)) / 255.)
    owner = profile.native[5::10, 5::10].astype(int)
    histograms = []
    for values in channels:
        bucket = owner * 10 + np.minimum(9, (values * 10).astype(int))
        counts = np.bincount(bucket.ravel(), minlength=100).reshape(10, 10)
        histograms.append(counts / np.maximum(1, counts.sum(axis=1, keepdims=True)))
    return np.stack(histograms, axis=1).astype(np.float32)


def scene_histogram_views(histograms):
    """Frozen scene descriptors, preserving the original region/channel/bin order."""
    if histograms.ndim < 3 or histograms.shape[-3:] != (10, 5, 10):
        raise ValueError("expected native region, channel and bin axes")
    leading = histograms.shape[:-3]
    return {"weather150": histograms[..., :3, :, :].reshape(leading + (150,)),
            "hour400": np.take(histograms, [0, 1, 3, 4], axis=-2).reshape(leading + (400,))}


def extract_region_nodes(rgb, profile):
    """Compute only the 10-by-8 physical RGB statistics for the quality estimator."""
    _require_rgb(rgb)
    thumb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    fields = _image_fields(thumb)
    return np.stack([_summarize(fields, profile.small == i) for i in range(len(REGIONS))])


def extract_region_rgb(rgb, profile):
    """Return separate scene experiments. Zero sky statistics carry an explicit support fraction."""
    _require_rgb(rgb)
    thumb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    fields = _image_fields(thumb)
    masks = [profile.small == i for i in range(len(REGIONS))]
    nodes = np.stack([_summarize(fields, mask) for mask in masks])
    sky = predicted_sky(thumb, profile)
    support = np.asarray([sky[mask].mean() for mask in masks[:3]], np.float32)
    sky27 = np.concatenate([*[_summarize(fields, mask & sky) for mask in masks[:3]], support])
    return {
        "nodes": nodes,
        "signed_differences": np.stack([nodes[a] - nodes[b] for a, b in PAIRS]),
        "haze30": nodes[:, [1, 3, 4]].ravel(),
        "sky27": sky27,
        "weather100": np.concatenate([sky27, extract_environment(rgb)]),
        "hour187": np.concatenate([sky27, _hour_histograms(_region_histograms(rgb, profile))]),
    }


def weather_region_views(rgb, drivable_map, profile):
    """Ten 67-value views from RGB and the unpadded f0 drivable output, with no scene metadata."""
    _require_rgb(rgb)
    if drivable_map.shape != (180, 320) or drivable_map.dtype != np.uint8:
        raise ValueError("expected a 320x180 byte drivable map")
    thumb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    fields = _image_fields(thumb)
    sky = predicted_sky(thumb, profile)
    road = drivable_map >= 128
    histograms = _region_histograms(rgb, profile)
    views = []
    for i in range(len(REGIONS)):
        mask = profile.small == i
        material = mask & sky if i < 3 else (mask & road if i < 9 else mask)
        views.append(np.concatenate([_summarize(fields, mask), _summarize(fields, material),
                                     [material.sum() / mask.sum()], histograms[i].ravel()]))
    return np.asarray(views, np.float32)


def summarize_head_maps(maps, profile):
    """Pool unpadded, 2x2 area-reduced and byte-quantized outputs in published region order."""
    if maps.dtype != np.uint8 or maps.shape != (4, 180, 320):
        raise ValueError("expected four 320x180 quantized head maps")
    values = maps.astype(np.float32) / 255.
    return np.asarray([[plane[profile.small == i].mean() for plane in values]
                       for i in range(len(REGIONS))], dtype=np.float32)


def reduce_heads(drivable_class, marking_probability, geometry):
    """Convert native f0 class IDs and four-class probabilities, including letterbox, to working maps."""
    if (tuple(geometry.original_hw) != (720, 1280) or tuple(geometry.target_hw) != (384, 640)
            or tuple(geometry.resized_hw) != (360, 640) or tuple(geometry.pad_ltrb) != (0, 12, 0, 12)):
        raise ValueError("unexpected front-camera geometry")
    if (drivable_class.shape != (384, 640) or not np.all((drivable_class == 0) | (drivable_class == 1))
            or marking_probability.shape != (4, 384, 640) or marking_probability.dtype != np.float32):
        raise ValueError("expected binary drivable class and four float32 marking probabilities")
    if (not np.isfinite(marking_probability).all() or marking_probability.min() < 0
            or marking_probability.max() > 1 or not np.allclose(marking_probability.sum(0), 1., atol=2e-6, rtol=0)):
        raise ValueError("invalid marking probabilities")
    q = marking_probability[:, 12:372]
    cls = q.argmax(0)
    entropy = -(q * np.log(np.maximum(q, np.float32(1e-35)))).sum(0) / np.float32(np.log(4))
    planes = np.stack([drivable_class[12:372] == 1, (cls == 1) | (cls == 2), q[1] + q[2], entropy])
    pooled = planes.reshape(4, 180, 2, 320, 2).mean(axis=(2, 4), dtype=np.float32)
    return np.rint(np.clip(pooled, 0., 1.) * 255.).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Original front camera PNG/JPEG")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    parser.add_argument("--head-maps", type=Path, help="Optional uint8 NPY, four 320x180 planes")
    parser.add_argument("--native-histograms", action="store_true", help="Include the two native scene descriptors")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error("could not decode front image")
    profile = FrontProfile.load()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = extract_region_rgb(rgb, profile)
    if args.native_histograms:
        result.update(scene_histogram_views(native_scene_histograms(rgb, profile)))
    if args.head_maps is not None:
        result["output40"] = summarize_head_maps(np.load(args.head_maps, allow_pickle=False), profile).ravel()
    with args.output.open("x", encoding="utf-8") as dest:
        json.dump({k: v.tolist() for k, v in result.items()}, dest, allow_nan=False)


if __name__ == "__main__":
    main()
