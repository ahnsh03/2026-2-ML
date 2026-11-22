import unittest

import numpy as np

from quality_estimation.gaussian_process import fit_quality_gp


class GaussianQualityTests(unittest.TestCase):
    def test_posterior_is_in_iou_units_with_positive_observation_width(self):
        x = np.array([[0, 5], [1, 5], [2, 5], [2, 5]], dtype=float)
        y = np.array([.3, .5, .76, .78])
        model = fit_quality_gp(x, y)
        estimate, sigma = model.predict(np.array([[0, 5], [2, 5], [100, 5]]), return_std=True)
        self.assertLess(estimate[0], estimate[1])
        self.assertTrue(np.all(sigma > 0))
        self.assertGreater(sigma[-1], sigma[1])
        self.assertLess(sigma[-1], .4)

    def test_rejects_row_budget_and_undefined_labels(self):
        with self.assertRaises(ValueError):
            fit_quality_gp([[0], [1], [2]], [.2, .3, .5], max_rows=2)
        with self.assertRaises(ValueError):
            fit_quality_gp([[0], [1]], [np.nan, .5])
