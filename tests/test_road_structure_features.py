import unittest
import numpy as np
from quality_estimation.road_structure_features import (
    morphology,
    extract_road_structure,
    ordered_features,
)
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.selected_region_features import extract_selected_features


class RoadStructureTests(unittest.TestCase):
    def test_gap_is_inside_first_and_last_supported_rows(self):
        lane = np.zeros((8, 5), bool)
        lane[[1, 2, 5, 6], :2] = True
        mask = np.ones_like(lane)
        np.testing.assert_allclose(
            morphology(lane, np.zeros_like(lane), mask), [0.5, 0.25, 0]
        )

    def test_weak_support_is_distinct_from_argmax_lane(self):
        profile = SpatialProfile.load()
        p = np.zeros((4, 360, 640), np.float32)
        p[0] = 0.7
        p[1] = 0.15
        p[2] = 0.15
        actual = extract_road_structure(p, profile).reshape(7, 3)
        np.testing.assert_array_equal(actual[:, :2], 0)
        np.testing.assert_allclose(actual[:, 2], 1)
        self.assertEqual(len(ordered_features()), 21)
        padded = np.pad(p, ((0, 0), (12, 12), (0, 0)), constant_values=0)
        padded[0, :12] = 1
        padded[0, 372:] = 1
        rgb = np.zeros((720, 1280, 3), np.uint8)
        subset = ["near.soft_outside", "global.row_coverage", "far.internal_gap"]
        actual = extract_selected_features(
            rgb, None, profile, subset, marking_probability=padded
        )
        np.testing.assert_allclose(actual, [1, 0, 0])

    def test_malformed_probabilities_rejected(self):
        p = np.ones((4, 360, 640), np.float32)
        with self.assertRaises(ValueError):
            extract_road_structure(p, SpatialProfile.load())


if __name__ == "__main__":
    unittest.main()
