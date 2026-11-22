import unittest

import numpy as np

from quality_estimation.labels import lane_iou_bands


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
