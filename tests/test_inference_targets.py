"""Regression checks for v6 lane union semantics and the legacy binary path."""
import unittest

import numpy as np

from visualization.visualize_inference import summarize_prediction


class InferenceTargetTests(unittest.TestCase):
    def summarize(self, marking, target, valid=None):
        shape = target.shape
        return summarize_prediction(
            {"road_marking": np.asarray(marking, dtype=np.float32).reshape(4, *shape),
             "drivable": np.stack([np.zeros(shape), np.ones(shape)])},
            {"road_marking": target, "drivable": np.ones(shape, dtype=np.uint8)},
            {"road_marking": np.ones(shape, dtype=bool) if valid is None else valid,
             "drivable": np.ones(shape, dtype=bool)},
            "road_marking",
        )

    def test_argmax_union_differs_from_summed_probability_threshold(self):
        # Pixel 0: stopline wins although P(white)+P(yellow) > .5.
        # Pixel 1: white wins although P(white)+P(yellow) < .5.
        result = self.summarize(
            [[.05, .30], [.31, .40], [.29, .05], [.35, .25]],
            np.array([[3, 1]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(result["marking_pred"], [[3, 1]])
        self.assertGreater(float(result["lane_prob"][0, 0]), .5)
        self.assertLess(float(result["lane_prob"][0, 1]), .5)
        self.assertEqual(result["lane_iou"], 1.0)
        self.assertEqual(result["lane_pred_px"], 1)

    def test_lane_on_stopline_counts_as_false_positive_and_valid_is_respected(self):
        marking = [[0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0]]
        target = np.array([[1, 3, 0]], dtype=np.uint8)
        result = self.summarize(marking, target, np.array([[True, True, False]]))
        self.assertEqual(result["lane_iou"], .5)
        self.assertEqual(result["lane_target_px"], 1)
        self.assertEqual(result["lane_pred_px"], 2)

    def test_empty_lane_union_is_undefined_even_with_stopline(self):
        result = self.summarize([[0], [0], [0], [1]], np.array([[3]], dtype=np.uint8))
        self.assertTrue(np.isnan(result["lane_iou"]))

    def test_legacy_binary_masks_and_thresholds(self):
        result = summarize_prediction(
            {"lane": np.array([[[.4, .2]], [[.6, .8]]]),
             "drivable": np.array([[[.1, .1]], [[.9, .9]]])},
            {"lane": np.array([[0, 1]]), "drivable": np.array([[1, 1]])},
            {"lane": np.ones((1, 2), dtype=bool), "drivable": np.ones((1, 2), dtype=bool)},
            "lane", lane_threshold=.7,
        )
        self.assertEqual(result["lane_iou"], 1.0)
        self.assertNotIn("marking_pred", result)


if __name__ == "__main__":
    unittest.main()
