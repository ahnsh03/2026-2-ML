"""Extract a frozen ordered subset while sharing the study's exact numerical definitions."""
from __future__ import annotations

import cv2
import numpy as np

from .region_candidates import (
    OUTPUT_FEATURES, RELATION_PAIRS, RGB_FEATURES, _output_values, _rgb_values,
    candidate_schema, native_histograms,
)
from .roi10 import _image_fields, _require_rgb
from .spatial_features import UNCERTAINTY_FEATURES, extract_global_uncertainty


def extract_selected_features(rgb, head_maps, profile, ordered_features, *, marking_probability=None):
    """Compute only needed region summaries. RGB environment recipes accept head_maps=None."""
    _require_rgb(rgb)
    schema = candidate_schema(profile)
    locations = {name: kind for kind in ("rgb", "output", "relations", "histograms")
                 for name in schema[kind]}
    locations.update({name: "uncertainty" for name in UNCERTAINTY_FEATURES})
    if not ordered_features or len(set(ordered_features)) != len(ordered_features):
        raise ValueError("expected a nonempty unique ordered feature list")
    if any(name not in locations for name in ordered_features):
        raise ValueError("feature list does not match the nine-region schema")
    pairs = {name: (a, b) for name, a, b in RELATION_PAIRS}
    needed_rgb, needed_output = set(), set()
    for name in ordered_features:
        region, statistic = name.split(".")
        kind = locations[name]
        if kind == "rgb":
            needed_rgb.add(region)
        elif kind == "output":
            needed_output.add(region)
        elif kind == "relations":
            dependencies = needed_rgb if statistic in RGB_FEATURES else needed_output
            dependencies.update(pairs[region])
    if needed_output and (head_maps is None or head_maps.dtype != np.uint8
                          or head_maps.shape != (4, 180, 320)):
        raise ValueError("this recipe requires four uint8 perception maps")
    masks = {name: profile.small == i for i, name in enumerate(profile.region_names)}
    masks["global"] = np.ones(profile.small.shape, bool)
    for band in ("upper_top", "upper_bottom", "far", "near"):
        masks[band] = masks[band+"_left"] | masks[band+"_right"]
    nodes = {}
    if needed_rgb:
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
        for region in needed_rgb:
            values = _rgb_values(fields, gray, saturation, extra, masks[region])
            nodes.update({region+"."+stat: v for stat, v in zip(RGB_FEATURES, values)})
    if needed_output:
        planes = head_maps.astype(np.float32)/255.
        for region in needed_output:
            values = _output_values(planes, masks[region])
            nodes.update({region+"."+stat: v for stat, v in zip(OUTPUT_FEATURES, values)})
    histogram_columns = [name for name in ordered_features if locations[name] == "histograms"]
    if histogram_columns:
        nodes.update(zip(histogram_columns, native_histograms(rgb, profile, histogram_columns)))
    if any(locations[name] == "uncertainty" for name in ordered_features):
        if marking_probability is None or marking_probability.shape != (4, 384, 640):
            raise ValueError("raw uncertainty features require float32 four-class 384x640 softmax")
        values = extract_global_uncertainty(marking_probability[:, 12:372])
        nodes.update(zip(UNCERTAINTY_FEATURES, values))
    result = []
    for name in ordered_features:
        if locations[name] == "relations":
            pair, statistic = name.split(".")
            a, b = pairs[pair]
            result.append(nodes[a+"."+statistic]-nodes[b+"."+statistic])
        else:
            result.append(nodes[name])
    vector = np.asarray(result, np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("nonfinite selected feature values")
    return vector
