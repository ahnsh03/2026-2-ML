"""Intention-based feature groups for paired quality and RGB environment experiments."""
from __future__ import annotations

import numpy as np

from .region_candidates import RGB_FEATURES, OUTPUT_FEATURES


ROLE_RGB = {
    "upper_top": ("intensity", "intensity_std", "glare_fraction", "dark_fraction",
                  "saturation_median", "saturation_iqr", "blue_minus_red", "hue_cos", "hue_sin"),
    "upper_bottom": ("intensity", "dark_channel", "local_contrast", "sharpness",
                     "median_residual", "intensity_p90_minus_p10", "saturation_median"),
    "far": ("intensity", "dark_channel", "local_contrast", "sharpness", "median_residual",
            "dark_fraction", "intensity_p90_minus_p10", "saturation_median", "edge_density"),
    "near": ("intensity", "dark_channel", "local_contrast", "sharpness", "median_residual",
             "dark_fraction", "intensity_p90_minus_p10", "saturation_median", "edge_density"),
    "hood": ("intensity", "intensity_std", "dark_fraction", "glare_fraction", "median_residual",
             "saturation_median", "blue_minus_red"),
    "global": ("intensity", "intensity_std", "dark_fraction", "intensity_p90_minus_p10", "saturation_median"),
}


def quality_views(schema):
    """Each listed arm changes one interpretable group relative to its stated parent."""
    rgb, heads = schema["rgb"], schema["output"]
    core = [n for n in rgb if n.split(".")[1] in RGB_FEATURES[:8]]
    robust = [n for n in rgb if n.split(".")[1] in RGB_FEATURES[:13]]
    base_heads = [n for n in heads if n.split(".")[1] in OUTPUT_FEATURES[:4]]
    road_heads = [n for n in base_heads if n.startswith(("far_", "near_", "global."))]
    offroad_lane = [n for n in base_heads if n not in road_heads and n.endswith(".lane_fraction")]
    role_rgb = []
    for n in rgb:
        region, stat = n.split(".")
        role = region.rsplit("_", 1)[0] if region.endswith(("_left", "_right")) else region
        if stat in ROLE_RGB[role]:
            role_rgb.append(n)
    views = {
        "base120": core+base_heads,
        "robust170": robust+base_heads,
        "color220": rgb+base_heads,
        "conditional260": rgb+heads,
        "rgb_only180": rgb,
        "output_only80": heads,
        "color_road_outputs": rgb+road_heads,
        "color_offroad_lane": rgb+road_heads+offroad_lane,
        "role_all_outputs": role_rgb+base_heads,
        "role_road_outputs": role_rgb+road_heads,
        "role_offroad_lane": role_rgb+road_heads+offroad_lane,
        "role_conditional_outputs": role_rgb+heads,
    }
    # Group removal controls quantify whether a physically motivated role is useful.
    for label, prefixes in (("no_top", ("upper_top_",)), ("no_bottom", ("upper_bottom_",)),
                            ("no_hood", ("hood.",)), ("no_global", ("global.",))):
        views["role_"+label] = [n for n in views["role_all_outputs"] if not n.startswith(prefixes)]
    relation_groups = {
        "distance": ("near_minus_far.",),
        "light": ("upper_bottom_minus_top.", "hood_minus_upper_top.", "hood_minus_near."),
        "sides": ("upper_top_left_minus_right.", "upper_bottom_left_minus_right.",
                  "far_left_minus_right.", "near_left_minus_right."),
    }
    for name, prefixes in relation_groups.items():
        related = [n for n in schema["relations"] if n.startswith(prefixes)]
        if name == "light":
            related = [n for n in related if n.split(".")[1] in RGB_FEATURES]
        views["role_rel_"+name] = views["role_all_outputs"]+related
    views["role_rel_all"] = views["role_all_outputs"]+schema["relations"]
    return views


def environment_views(schema):
    """RGB-only views. Class names, clock, position and perception are excluded."""
    rgb, hist = schema["rgb"], schema["histograms"]
    bands = {
        "top_hood": ("upper_top_", "hood."),
        "upper_hood": ("upper_top_", "upper_bottom_", "hood."),
        "road": ("far_", "near_"),
        "all": (),
    }
    result = {}
    for label, prefixes in bands.items():
        result[label+"_rgb"] = [n for n in rgb if not prefixes or n.startswith(prefixes)]
        result[label+"_hist"] = [n for n in hist if not prefixes or n.startswith(prefixes)]
    result["role_rgb"] = [n for n in quality_views(schema)["role_all_outputs"] if n in rgb]
    light = [n for n in schema["relations"] if n.split(".")[1] in RGB_FEATURES
             and n.startswith(("upper_bottom_minus_top.", "hood_minus_upper_top.", "hood_minus_near."))]
    result["top_hood_rgb_light"] = result["top_hood_rgb"]+light
    result["upper_hood_rgb_light"] = result["upper_hood_rgb"]+light
    result["top_only_rgb"] = [n for n in rgb if n.startswith("upper_top_")]
    result["hood_only_rgb"] = [n for n in rgb if n.startswith("hood.")]
    photometric = ("intensity", "intensity_std", "dark_fraction", "glare_fraction",
                   "intensity_p90_minus_p10", "saturation_median", "blue_minus_red", "hue_cos", "hue_sin")
    result["top_hood_photometric"] = [n for n in result["top_hood_rgb"] if n.split(".")[1] in photometric]
    weather_context = []
    hour_context = []
    for name in hist:
        region, statistic = name.split(".")
        channel = statistic.split("_bin")[0]
        if ((region.startswith("upper_top_") and channel in ("hue", "saturation"))
                or (region.startswith("upper_bottom_") and channel in ("minimum_intensity", "contrast", "sharpness"))
                or (region == "hood" and channel in ("minimum_intensity", "contrast", "saturation"))):
            weather_context.append(name)
        if ((region.startswith("upper_top_") and channel in ("minimum_intensity", "hue", "saturation"))
                or (region.startswith("far_") and channel in ("minimum_intensity", "sharpness"))
                or (region.startswith("near_") and channel == "contrast")
                or (region == "hood" and channel in ("minimum_intensity", "saturation"))):
            hour_context.append(name)
    result["weather_context_hist"] = weather_context
    result["hour_context_hist"] = hour_context
    far_haze = [n for n in hist if n.startswith("far_")
                and n.split(".")[1].split("_bin")[0] in ("minimum_intensity", "contrast")]
    result["weather_context_far_hist"] = weather_context+far_haze
    for channel in ("minimum_intensity", "contrast", "sharpness", "hue", "saturation"):
        result["all_hist_no_"+channel] = [n for n in hist if not n.split(".")[1].startswith(channel+"_bin")]
    return result


def feature_matrix(pack, schema, ordered_features):
    """Materialize exactly the requested ordered columns from a reusable candidate pack."""
    kinds = ("rgb", "output", "relations", "histograms", "uncertainty")
    locations = {name: (kind, i) for kind in kinds if kind in schema
                 for i, name in enumerate(schema[kind])}
    if len(set(ordered_features)) != len(ordered_features):
        raise ValueError("duplicate requested feature columns")
    return np.column_stack([pack[kind][:, i] for kind, i in
                            (locations[name] for name in ordered_features)]).astype(np.float32)
