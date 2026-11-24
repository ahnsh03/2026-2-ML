import json
from pathlib import Path
import unittest

import numpy as np

from quality_estimation.roi10 import FrontProfile, extract_region_nodes, summarize_head_maps
from quality_estimation.spatial_features import (
    SpatialProfile, extract_roi7_iou_features, extract_spatial_features,
    roi7_iou_feature_schema, spatial_feature_schema,
)


class SpatialFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.seven = SpatialProfile.load(cls.root / "config/roi7_front_v1")
        cls.ten = SpatialProfile.load(cls.root / "config/roi10_front_v1")

    def test_default_profile_is_selected_seven_region_baseline(self):
        profile = SpatialProfile.load()
        self.assertEqual(profile.version, "roi7-front-v1")
        np.testing.assert_array_equal(profile.native, self.seven.native)

    def test_ten_region_baseline_matches_existing_extractor(self):
        rng = np.random.default_rng(1701)
        rgb = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        maps = rng.integers(0, 256, (4, 180, 320), dtype=np.uint8)
        original = FrontProfile.load()
        expected = np.concatenate([
            extract_region_nodes(rgb, original), summarize_head_maps(maps, original),
        ], axis=1).ravel()
        np.testing.assert_allclose(
            extract_spatial_features(rgb, maps, self.ten), expected, atol=1e-6,
        )

    def test_seven_regions_preserve_bands_and_split_center_at_half_width(self):
        seven, ten = self.seven.native, self.ten.native
        np.testing.assert_array_equal(seven == 6, ten == 9)
        for band in range(3):
            np.testing.assert_array_equal(seven // 2 == band, ten // 3 == band)
        self.assertEqual(seven[100, 639], 0)
        self.assertEqual(seven[100, 640], 1)
        self.assertEqual(ten[100, 639], ten[100, 640])
        rgb = np.zeros((720, 1280, 3), np.uint8)
        rgb[:, 640:] = 255
        maps = np.zeros((4, 180, 320), np.uint8)
        maps[:, :, 160:] = 255
        values = extract_spatial_features(rgb, maps, self.seven).reshape(7, 12)
        np.testing.assert_allclose(values[:2, 0], [0., 1.])
        np.testing.assert_allclose(values[:2, 8:], [[0.] * 4, [1.] * 4])

    def test_global_statistics_and_robust_quantiles_use_all_pixels(self):
        rgb = np.zeros((720, 1280, 3), np.uint8)
        rgb[:, 640:] = 255
        maps = np.zeros((4, 180, 320), np.uint8)
        maps[:, :, 160:] = 255
        for profile in (self.seven, self.ten):
            vector = extract_spatial_features(
                rgb, maps, profile, feature_set="robust17", include_global=True,
            )
            schema = spatial_feature_schema(
                profile, feature_set="robust17", include_global=True,
            )
            values = dict(zip(schema["ordered_features"], vector))
            self.assertAlmostEqual(values["global.intensity"], .5)
            self.assertAlmostEqual(values["global.drivable_fraction"], .5)
            self.assertAlmostEqual(values["global.dark_fraction"], .5)
            self.assertAlmostEqual(values["global.intensity_p10"], 0.)
            self.assertAlmostEqual(values["global.intensity_median"], .5)
            self.assertAlmostEqual(values["global.intensity_p90_minus_p10"], 1.)
            self.assertEqual(len(values), len(vector))

    def test_no_predictions_and_input_validation(self):
        rgb = np.zeros((720, 1280, 3), np.uint8)
        maps = np.zeros((4, 180, 320), np.uint8)
        for profile in (self.seven, self.ten):
            result = extract_spatial_features(rgb, maps, profile, feature_set="robust17")
            rows = result.reshape(len(profile.region_names), 17)
            np.testing.assert_array_equal(rows[:, :12], 0.)
            np.testing.assert_array_equal(rows[:, 12], 1.)
            np.testing.assert_array_equal(rows[:, 13:], 0.)
        with self.assertRaises(ValueError):
            extract_spatial_features(rgb, maps.astype(np.float32), self.seven)
        with self.assertRaises(ValueError):
            extract_spatial_features(rgb[::2], maps, self.seven)
        with self.assertRaises(ValueError):
            spatial_feature_schema(self.ten, feature_set="missing")


    def test_selected_schema_and_padding(self):
        rgb = np.zeros((720, 1280, 3), np.uint8)
        maps = np.zeros((4, 180, 320), np.uint8)
        p = np.zeros((4, 384, 640), np.float32)
        p[1] = .75
        p[0] = .25
        p[:, :12] = .25
        p[:, 372:] = .25
        selected = json.loads((self.root / "docs/baselines/roi7_final_20261124/selected_features.json").read_text())
        self.assertEqual(roi7_iou_feature_schema(self.seven)["ordered_features"], selected["ordered_features"])
        values = extract_roi7_iou_features(rgb, maps, p, self.seven)
        expected_entropy = -(.75 * np.log(.75) + .25 * np.log(.25)) / np.log(4)
        np.testing.assert_array_equal(values[:96], extract_spatial_features(rgb, maps, self.seven, include_global=True))
        np.testing.assert_allclose(values[-3:], [expected_entropy, expected_entropy, .5], atol=1e-6)
        p[[0, 1]] = p[[1, 0]]
        self.assertEqual(extract_roi7_iou_features(rgb, maps, p, self.seven)[-1], 0.)
        with self.assertRaises(ValueError):
            extract_roi7_iou_features(rgb, maps, p, self.ten)
