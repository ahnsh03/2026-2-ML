"""Meaningful invariants: GT separation, padding, ROI geometry and label edge cases."""
import unittest

import numpy as np

from camera_semantic_perception.geometry import compute_letterbox_geometry
from quality_estimation import FEATURE_NAMES, build_label, extract_features
from quality_estimation.features import geometric_valid, grid_regions


class FeatureTests(unittest.TestCase):
    def test_uniform_uncertainty_and_no_lane_are_finite(self):
        geometry = compute_letterbox_geometry((6, 4), (6, 4))
        x = extract_features(np.full((6, 4, 3), 39, np.uint8), np.full((4, 6, 4), .25, np.float32), geometry)
        self.assertEqual(x.shape, (56,))
        self.assertEqual(len(set(FEATURE_NAMES)), 56)
        np.testing.assert_allclose(x.reshape(7, 8)[:, :4], [[0, 1, 0, 0]] * 7)
        np.testing.assert_allclose(x.reshape(7, 8)[:, 4:], [[39/255, 1, 0, 0]] * 7)

    def test_padding_cannot_change_features(self):
        geometry = compute_letterbox_geometry((6, 4), (8, 8))
        rgb = np.full((6, 4, 3), 128, np.uint8)
        p = np.zeros((4, 8, 8), np.float32); p[1] = 1
        q = p.copy(); padding = ~geometric_valid(geometry); q[:, padding] = 0; q[3, padding] = 1
        np.testing.assert_array_equal(extract_features(rgb, p, geometry), extract_features(rgb, q, geometry))

    def test_odd_grid_tiles_without_gaps_or_overlap(self):
        hits = np.zeros((7, 5), np.uint8)
        for _, (x0, y0, x1, y1) in grid_regions(7, 5)[1:]:
            hits[y0:y1, x0:x1] += 1
        np.testing.assert_array_equal(hits, np.ones_like(hits))
        self.assertEqual(grid_regions(7, 5)[1], ("r1c1", (0, 0, 2, 2)))
        self.assertEqual(grid_regions(7, 5)[-1], ("r3c2", (2, 4, 5, 7)))

    def test_native_rgb_statistics_not_downsampled_input(self):
        rgb = np.repeat(((np.indices((12, 8)).sum(0) % 2) * 255).astype(np.uint8)[..., None], 3, axis=2)
        geometry = compute_letterbox_geometry((12, 8), (6, 4))
        p = np.zeros((4, 6, 4), np.float32); p[0] = 1
        x = dict(zip(FEATURE_NAMES, extract_features(rgb, p, geometry)))
        self.assertEqual(x["global.brightness_mean"], .5)
        self.assertEqual(x["global.brightness_std"], .5)
        self.assertEqual(x["global.dark_fraction"], .5)

    def test_argmax_lane_and_eight_connected_components(self):
        geometry = compute_letterbox_geometry((6, 4), (6, 4))
        p = np.zeros((4, 6, 4), np.float32); p[0] = 1
        # Diagonal pixels are one 8-connected component.
        for yy, xx in [(0, 0), (1, 1)]:
            p[:, yy, xx] = [.30, .40, .05, .25]
        # Summed lane probability > .5, but stopline wins: not a lane.
        p[:, 5, 3] = [.05, .31, .29, .35]
        x = dict(zip(FEATURE_NAMES, extract_features(np.zeros((6, 4, 3), np.uint8), p, geometry)))
        self.assertEqual(x["global.lane_components"], 1)
        self.assertAlmostEqual(x["global.lane_area_fraction"], 2/24)
        self.assertAlmostEqual(x["global.lane_confidence"], .45)

    def test_dark_threshold_is_strict(self):
        geometry = compute_letterbox_geometry((6, 4), (6, 4))
        p = np.zeros((4, 6, 4), np.float32); p[0] = 1
        x = extract_features(np.full((6, 4, 3), 40, np.uint8), p, geometry)
        self.assertEqual(x[5], 0)

    def test_malformed_probabilities_fail(self):
        geometry = compute_letterbox_geometry((6, 4), (6, 4))
        with self.assertRaises(ValueError):
            extract_features(np.zeros((6, 4, 3), np.uint8), np.ones((4, 6, 4), np.float32), geometry)

    def test_legacy_rgba_does_not_change_semantic_classes(self):
        from quality_estimation.build_dataset import color_channels, native_marking
        bgr = np.array([[[255, 255, 255], [0, 255, 255], [0, 0, 255], [127, 127, 127]]], np.uint8)
        bgra = np.concatenate([bgr, np.zeros((1, 4, 1), np.uint8)], axis=2)
        np.testing.assert_array_equal(native_marking(color_channels(bgra, "GT")), [[1, 2, 3, 0]])


class LabelTests(unittest.TestCase):
    def label(self, pred, gt, valid=None):
        pred, gt = np.array([pred], np.uint8), np.array([gt], np.uint8)
        return build_label(pred, gt, np.ones_like(pred, bool),
                           np.ones_like(pred, bool) if valid is None else np.array([valid], bool))

    def test_white_yellow_union_stopline_false_positive_and_miss(self):
        result = self.label([2, 1, 0, 3], [1, 3, 2, 3])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["lane_iou"], 1/3)

    def test_empty_union_is_missing_not_perfect(self):
        result = self.label([0, 3], [0, 3])
        self.assertIsNone(result["lane_iou"])
        self.assertFalse(result["label_defined"])
        self.assertEqual(result["undefined_reason"], "empty_union")

    def test_no_gt_with_false_positive_is_defined_zero(self):
        result = self.label([1, 3], [0, 3])
        self.assertTrue(result["label_defined"])
        self.assertEqual(result["lane_iou"], 0)

    def test_annotation_valid_changes_y_without_changing_feature_api(self):
        first = self.label([1, 1], [1, 0])
        masked = self.label([1, 1], [1, 0], [True, False])
        self.assertEqual(first["lane_iou"], .5)
        self.assertEqual(masked["lane_iou"], 1)
        import inspect
        self.assertEqual(list(inspect.signature(extract_features).parameters), ["rgb", "probabilities", "geometry"])


if __name__ == "__main__":
    unittest.main()
