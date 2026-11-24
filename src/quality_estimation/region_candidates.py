"""Named RGB, perception and relation candidates for the fixed nine-region study.

Extraction depends only on front RGB, frozen perception maps and camera geometry.
Scene labels and IoU targets belong in the study's separate sample table.
"""
from __future__ import annotations

import cv2
import numpy as np

from .roi10 import HEAD_FEATURES, NODE_FEATURES, _image_fields, _require_rgb, _summarize
from .spatial_features import EXTRA_RGB_FEATURES


COLOR_FEATURES = ("blue_minus_red", "saturation_iqr", "hue_cos", "hue_sin", "edge_density")
RGB_FEATURES = NODE_FEATURES + EXTRA_RGB_FEATURES + COLOR_FEATURES
OUTPUT_FEATURES = HEAD_FEATURES + (
    "lane_weighted_entropy", "lane_weighted_probability",
    "lane_drivable_disagreement", "lane_present",
)
HISTOGRAM_CHANNELS = ("minimum_intensity", "contrast", "sharpness", "hue", "saturation")
RELATION_PAIRS = (
    ("near_minus_far", "near", "far"),
    ("upper_bottom_minus_top", "upper_bottom", "upper_top"),
    ("hood_minus_upper_top", "hood", "upper_top"),
    ("hood_minus_near", "hood", "near"),
    ("upper_top_left_minus_right", "upper_top_left", "upper_top_right"),
    ("upper_bottom_left_minus_right", "upper_bottom_left", "upper_bottom_right"),
    ("far_left_minus_right", "far_left", "far_right"),
    ("near_left_minus_right", "near_left", "near_right"),
)
RELATION_STATISTICS = (
    "intensity", "dark_channel", "local_contrast", "sharpness",
    "intensity_p90_minus_p10", "saturation_median",
    "lane_fraction", "lane_probability", "marking_entropy",
    "lane_weighted_entropy", "lane_drivable_disagreement",
)


def candidate_schema(profile):
    if profile.version != "roi9-front-v1":
        raise ValueError("nine-region profile required")
    regions = profile.region_names + ("global",)
    return {
        "version": "region-candidates-v1",
        "profile": profile.version,
        "regions_sha256": profile.regions_sha256,
        "working_hw": [180, 320],
        "rgb": [f"{r}.{s}" for r in regions for s in RGB_FEATURES],
        "output": [f"{r}.{s}" for r in regions for s in OUTPUT_FEATURES],
        "relations": [f"{pair}.{s}" for pair, _, _ in RELATION_PAIRS
                      for s in RELATION_STATISTICS],
        "histograms": [f"{r}.{s}_bin{b}" for r in regions
                       for s in HISTOGRAM_CHANNELS for b in range(10)],
        "head_source": "fixed four-plane byte maps; conditional statistics use pooled lane weights",
        "histogram_source": "native 10x10 blocks assigned by center pixel; Roser-inspired adaptation",
    }


def _rgb_values(fields, gray, saturation, extra, mask):
    q10, median, q90 = np.quantile(gray[mask], [.1, .5, .9]) / 255.
    sat25, sat50, sat75 = np.quantile(saturation[mask], [.25, .5, .75]) / 255.
    robust = [(gray[mask] < 40).mean(), q10, median, q90 - q10, sat50]
    blue_red, hue_cos, hue_sin, edges = extra
    color = [blue_red[mask].mean(), sat75-sat25,
             hue_cos[mask].mean(), hue_sin[mask].mean(), edges[mask].mean()]
    return np.r_[_summarize(fields, mask), robust, color].astype(np.float32)


def _output_values(planes, mask):
    drive, lane, probability, entropy = (p[mask] for p in planes)
    mass = lane.sum(dtype=np.float64)
    conditional = [0., 0.]
    if mass > 0:
        conditional = [(lane*entropy).sum(dtype=np.float64)/mass,
                       (lane*probability).sum(dtype=np.float64)/mass]
    return np.asarray([
        drive.mean(), lane.mean(), probability.mean(), entropy.mean(),
        *conditional, (lane*(1-drive)).mean(), float(mass > 0),
    ], dtype=np.float32)


def native_histograms(rgb, profile, ordered_features=None):
    """Ten-bin block distributions, adapted to the complete fixed camera partition.

    Minimum and maximum are darkest/brightest decile means. Sharpness uses the
    existing normalized Sobel magnitude. Hue uses saturation-weighted circular
    means. These choices are explicit adaptations of the weather descriptor.
    """
    _require_rgb(rgb)
    all_names = candidate_schema(profile)["histograms"]
    columns = all_names if ordered_features is None else ordered_features
    allowed_names = set(all_names)
    if not columns or any(name not in allowed_names for name in columns):
        raise ValueError("expected native histogram feature names")
    needed = {name.split(".")[1].split("_bin")[0] for name in columns}
    def block_mean(a):
        return a.reshape(72, 10, 128, 10).mean((1, 3))
    channels = {}
    if needed.intersection(("minimum_intensity", "contrast", "sharpness")):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        if needed.intersection(("minimum_intensity", "contrast")):
            blocks = gray.reshape(72, 10, 128, 10).transpose(0, 2, 1, 3).reshape(72, 128, 100)
            ordered = np.partition(blocks, (9, 90), axis=-1)
            low, high = ordered[..., :10].mean(-1), ordered[..., 90:].mean(-1)
            channels["minimum_intensity"] = low
            channels["contrast"] = (high-low)/np.maximum(1/255., high+low)
        if "sharpness" in needed:
            magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                                     cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            channels["sharpness"] = np.minimum(1., block_mean(magnitude)/4.)
    if needed.intersection(("hue", "saturation")):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[..., 1] / 255.
        channels["saturation"] = block_mean(saturation)
        if "hue" in needed:
            angle = hsv[..., 0] * (2*np.pi/180.)
            cos = block_mean(np.cos(angle)*saturation)
            sin = block_mean(np.sin(angle)*saturation)
            channels["hue"] = np.mod(np.arctan2(sin, cos), 2*np.pi)/(2*np.pi)
    owners = profile.native[5::10, 5::10]
    masks = {name: owners == i for i, name in enumerate(profile.region_names)}
    masks["global"] = np.ones(owners.shape, bool)
    histograms, result = {}, []
    for name in columns:
        region, statistic = name.split(".")
        channel, index = statistic.split("_bin")
        key = (region, channel)
        if key not in histograms:
            mask = masks[region]
            bins = np.minimum(9, (channels[channel][mask]*10).astype(int))
            counts = np.bincount(bins, minlength=10)
            histograms[key] = counts/len(bins)
        result.append(histograms[key][int(index)])
    return np.asarray(result, np.float32)


def extract_candidates(rgb, head_maps, profile, *, include_histograms=True):
    """Return separately named candidate arrays for quality and environment studies."""
    _require_rgb(rgb)
    schema = candidate_schema(profile)
    if head_maps.dtype != np.uint8 or head_maps.shape != (4, 180, 320):
        raise ValueError("expected four uint8 320x180 frozen perception maps")
    thumb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    fields = _image_fields(thumb)
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    angle = hsv[..., 0].astype(np.float32)*(2*np.pi/180.)
    weight = saturation.astype(np.float32)/255.
    extra = ((thumb[..., 2].astype(np.float32)-thumb[..., 0])/255.,
             np.cos(angle)*weight, np.sin(angle)*weight,
             cv2.Canny(gray, 100, 200, L2gradient=True)>0)
    masks = {name: profile.small == i for i, name in enumerate(profile.region_names)}
    masks["global"] = np.ones(profile.small.shape, bool)
    planes = head_maps.astype(np.float32)/255.
    rgb_nodes, head_nodes = {}, {}
    for name, mask in masks.items():
        rgb_nodes[name] = _rgb_values(fields, gray, saturation, extra, mask)
        head_nodes[name] = _output_values(planes, mask)
    rgb_vector = np.concatenate(list(rgb_nodes.values()))
    head_vector = np.concatenate(list(head_nodes.values()))
    for name in ("upper_top", "upper_bottom", "far", "near"):
        mask = masks[name+"_left"] | masks[name+"_right"]
        rgb_nodes[name] = _rgb_values(fields, gray, saturation, extra, mask)
        head_nodes[name] = _output_values(planes, mask)
    nodes = {name: dict(zip(RGB_FEATURES+OUTPUT_FEATURES,
                           np.r_[rgb_nodes[name], head_nodes[name]])) for name in rgb_nodes}
    relations = np.asarray([nodes[a][s]-nodes[b][s] for _, a, b in RELATION_PAIRS
                            for s in RELATION_STATISTICS], np.float32)
    result = {"rgb": rgb_vector, "output": head_vector, "relations": relations}
    if include_histograms:
        result["histograms"] = native_histograms(rgb, profile)
    for name, values in result.items():
        if values.shape != (len(schema[name]),) or not np.isfinite(values).all():
            raise ValueError("invalid candidate array for "+name)
    return result
