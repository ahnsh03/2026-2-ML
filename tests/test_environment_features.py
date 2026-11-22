import unittest

import numpy as np

from quality_estimation.environment_features import extract_environment, NAMES, PATCHES


class PatchFeatureTests(unittest.TestCase):
    def test_uniform_patch_statistics_and_signed_contrast(self):
        image = np.full((720, 1280, 3), 128, dtype=np.uint8)
        x0, y0, x1, y1 = PATCHES["sky"]
        image[y0:y1, x0:x1] = 255
        values = dict(zip(NAMES, extract_environment(image)))
        self.assertEqual(len(values), 73)
        self.assertEqual(values["sky.bright_fraction"], 1)
        self.assertEqual(values["hood_left.std"], 0)
        self.assertEqual(values["road_near.edge_density"], 0)
        self.assertAlmostEqual(values["hood_minus_sky"], 128/255-1, places=6)

    def test_rejects_letterbox(self):
        with self.assertRaises(ValueError):
            extract_environment(np.zeros((384, 640, 3), np.uint8))
