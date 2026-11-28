import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "extract_rgb_features", Path(__file__).resolve().parents[1] / "scripts/extract_rgb_features.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RGBFeatureTest(unittest.TestCase):
    def test_constant_colors_and_channel_order(self):
        for color, expected in [([0, 0, 0], 0), ([255, 255, 255], 1),
                                ([0, 0, 255], 0.299), ([255, 0, 0], 0.114)]:
            stats, _, _, _, _ = MODULE.extract(np.full((30, 40, 3), color, np.uint8))
            self.assertEqual(sum(len(s) for s in stats.values()), 28)
            for values in stats.values():
                self.assertAlmostEqual(values["mean_brightness"], expected)
                self.assertAlmostEqual(values["brightness_std"], 0)
                self.assertEqual(values["dark_ratio"], float(expected < 0.2))
                self.assertEqual(values["edge_density"], 0)

    def test_dark_threshold_and_known_contrast(self):
        image = np.zeros((30, 40, 3), np.uint8)
        image[:, 20:] = 255
        stats, _, _, _, _ = MODULE.extract(image)
        self.assertAlmostEqual(stats["full"]["mean_brightness"], 0.5)
        self.assertAlmostEqual(stats["full"]["brightness_std"], 0.5)
        self.assertEqual(stats["full"]["dark_ratio"], 0.5)
        self.assertGreater(stats["full"]["edge_density"], 0)
        for level, ratio in ((50, 1), (52, 0)):
            stats, *_ = MODULE.extract(np.full((30, 40, 3), level, np.uint8))
            self.assertEqual(stats["full"]["dark_ratio"], ratio)

    def test_odd_dimensions_partition_and_global_edges(self):
        image = np.random.default_rng(17).integers(0, 256, (31, 43, 3), dtype=np.uint8)
        stats, regions, _, _, edges = MODULE.extract(image)
        visits = np.zeros(image.shape[:2], np.uint8)
        weighted = {k: 0 for k in ("mean_brightness", "dark_ratio", "edge_density")}
        for name, (x0, y0, x1, y1) in regions.items():
            if name == "full":
                continue
            visits[y0:y1, x0:x1] += 1
            size = (x1-x0) * (y1-y0)
            self.assertEqual(stats[name]["edge_density"], edges[y0:y1, x0:x1].mean())
            for key in weighted:
                weighted[key] += size * stats[name][key]
        self.assertTrue(np.all(visits == 1))
        for key in weighted:
            self.assertAlmostEqual(weighted[key] / visits.size, stats["full"][key])


if __name__ == "__main__":
    unittest.main()
