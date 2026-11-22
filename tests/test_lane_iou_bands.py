import unittest

import numpy as np

from quality_estimation.labels import lane_iou_bands, lane_iou_by_depth_code


class LaneBandTests(unittest.TestCase):
    def test_band_counts_reconstruct_full_image_and_exclude_padding(self):
        target = np.ones((9, 4), np.uint8)
        pred = target.copy()
        pred[2:5, :2] = 0
        mask = np.ones_like(target, dtype=bool)
        mask[[0, 1, 8]] = False
        labels = lane_iou_bands(pred, target, mask, mask, split_fraction=.5)
        self.assertEqual(labels["upper"]["fn"], 6)
        self.assertEqual(labels["lower"]["fn"], 0)
        for field in ("tp", "fp", "fn", "valid_pixels"):
            self.assertEqual(labels["full"][field], labels["upper"][field] + labels["lower"][field])

    def test_depth_code_boundaries_and_missing_measurements_reconstruct_full_target(self):
        depth = np.array([[0, 1, 8, 9], [15, 16, 254, 255]], dtype=np.uint8)
        expected = np.ones_like(depth)
        predicted = expected.copy()
        predicted[:, 1] = 0
        valid = np.ones_like(depth, dtype=bool)
        annotated = valid.copy()
        annotated[1, 2] = False
        labels = lane_iou_by_depth_code(predicted, expected, valid, annotated, depth, cutoffs=(8, 15))
        self.assertEqual([labels[name]["valid_pixels"] for name in
                          ("zero", "low", "middle", "high", "saturated")], [1, 2, 2, 1, 1])
        self.assertEqual(labels["low"]["fn"], 1)
        self.assertEqual(labels["high"]["fn"], 1)
        for field in ("tp", "fp", "fn", "valid_pixels"):
            self.assertEqual(labels["full"][field], sum(v[field] for k, v in labels.items() if k != "full"))
        with self.assertRaises(ValueError):
            lane_iou_by_depth_code(predicted, expected, valid, annotated, depth, cutoffs=(15, 8))
