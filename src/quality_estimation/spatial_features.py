"""GT-free whole-frame regression features on the fixed seven- or ten-region camera profile."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .roi10 import HEAD_FEATURES, NODE_FEATURES, REGIONS, _image_fields, _require_rgb, _summarize


DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[2] / "config/roi7_front_v1"


EXTRA_RGB_FEATURES = (
    "dark_fraction", "intensity_p10", "intensity_median",
    "intensity_p90_minus_p10", "saturation_median",
)


@dataclass(frozen=True)
class SpatialProfile:
    version: str
    region_names: tuple[str, ...]
    native: np.ndarray
    small: np.ndarray
    regions_sha256: str

    @classmethod
    def load(cls, directory=None):
        directory = Path(directory) if directory is not None else DEFAULT_PROFILE_DIR
        spec = json.loads((directory / "profile.json").read_text())
        names = tuple(spec.get("region_names", REGIONS))
        if spec["version"] not in ("roi7-front-v1", "roi10-front-v1"):
            raise ValueError("unsupported spatial profile")
        expected_regions = 7 if spec["version"] == "roi7-front-v1" else 10
        if len(names) != expected_regions or len(set(names)) != expected_regions:
            raise ValueError("profile region names differ")
        path = directory / "regions.png"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        native = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if (digest != spec["regions_sha256"] or native is None
                or native.dtype != np.uint8 or native.shape != (720, 1280)
                or np.bincount(native.ravel()).tolist() != spec["native_region_pixels"]
                or np.unique(native).tolist() != list(range(expected_regions))
                or (spec["camera"]["height"], spec["camera"]["width"]) != native.shape):
            raise ValueError("spatial profile mask or geometry differs")
        small = cv2.resize(native, (320, 180), interpolation=cv2.INTER_NEAREST)
        if np.unique(small).size != expected_regions:
            raise ValueError("region has no working-resolution support")
        return cls(spec["version"], names, native, small, digest)


def spatial_feature_schema(profile, *, feature_set="base12", include_global=False):
    if feature_set not in ("base12", "robust17"):
        raise ValueError("unknown spatial feature set")
    statistics = NODE_FEATURES + HEAD_FEATURES
    if feature_set == "robust17":
        statistics += EXTRA_RGB_FEATURES
    regions = profile.region_names + (("global",) if include_global else ())
    return {
        "version": "spatial-quality-features-v1",
        "profile": profile.version,
        "regions_sha256": profile.regions_sha256,
        "feature_set": feature_set,
        "include_global": include_global,
        "ordered_features": [f"{region}.{stat}" for region in regions for stat in statistics],
        "working_hw": [180, 320],
        "head_maps": "uint8 four-plane 2x2-averaged model output, padding excluded",
        "extra_rgb": "320x180 area-resized RGB; grayscale <40, quantiles .1/.5/.9, HSV saturation median",
    }


def extract_spatial_features(rgb, head_maps, profile, *, feature_set="base12", include_global=False):
    """Return one region-major vector without opening any labels or scene metadata."""
    _require_rgb(rgb)
    if head_maps.dtype != np.uint8 or head_maps.shape != (4, 180, 320):
        raise ValueError("expected four 320x180 byte head maps")
    schema = spatial_feature_schema(profile, feature_set=feature_set, include_global=include_global)
    thumb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    fields = _image_fields(thumb)
    planes = head_maps.astype(np.float32) / 255.
    masks = [profile.small == i for i in range(len(profile.region_names))]
    if include_global:
        masks.append(np.ones((180, 320), dtype=bool))
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[:, :, 1]
    rows = []
    for mask in masks:
        values = [
            _summarize(fields, mask),
            np.asarray([plane[mask].mean() for plane in planes], dtype=np.float32),
        ]
        if feature_set == "robust17":
            q10, median, q90 = np.quantile(gray[mask], [.1, .5, .9]) / 255.
            values.append(np.asarray([
                (gray[mask] < 40).mean(), q10, median, q90 - q10,
                np.median(saturation[mask]) / 255.,
            ], dtype=np.float32))
        rows.append(np.concatenate(values))
    vector = np.concatenate(rows).astype(np.float32)
    if vector.shape != (len(schema["ordered_features"]),) or not np.isfinite(vector).all():
        raise ValueError("invalid spatial feature vector")
    return vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--head-maps", required=True, type=Path)
    parser.add_argument("--roi-profile", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--features", choices=("base12", "robust17"), default="base12")
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error("could not decode RGB")
    profile = SpatialProfile.load(args.roi_profile)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    values = extract_spatial_features(
        rgb, np.load(args.head_maps, allow_pickle=False), profile,
        feature_set=args.features, include_global=args.include_global,
    )
    record = dict(
        schema=spatial_feature_schema(profile, feature_set=args.features, include_global=args.include_global),
        values=values.tolist(),
    )
    with args.output.open("x", encoding="utf-8") as dest:
        json.dump(record, dest, allow_nan=False, ensure_ascii=False, indent=2)
        dest.write("\n")


if __name__ == "__main__":
    main()
