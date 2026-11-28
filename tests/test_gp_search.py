import unittest

import numpy as np
from scipy.spatial.distance import cdist

from quality_estimation.gp_search import (evaluate_cv, expected_improvement, metrics,
                                         next_point, purged_folds, search_space, shared_start)


class GPSearchTests(unittest.TestCase):
    def test_spatial_folds_exclude_same_parent_and_neighboring_locations(self):
        groups = np.repeat(["a", "b", "c"], 9)
        station = np.tile(np.arange(9)*50., 3)
        xy = np.column_stack([station, np.repeat([0., 1., 2.], 9)])
        folds = purged_folds(groups, xy, station)
        validation = []
        for _, train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            self.assertGreaterEqual(cdist(xy[train], xy[valid]).min(), 20)
            validation.extend(valid)
        self.assertEqual(sorted(validation), list(range(len(groups))))

    def test_invalid_fold_geometry_fails_instead_of_silently_weakening_purge(self):
        with self.assertRaises(ValueError):
            purged_folds(["a", "a", "b", "b"], np.zeros((4, 2)), np.arange(4))

    def test_intervals_use_raw_mean_and_report_false_positive_and_negative(self):
        result = metrics([.9, .2], [.8, .8], [.1, .1])
        self.assertAlmostEqual(result["clipped_mae"], .35)
        self.assertEqual(result["coverage90_raw"], .5)
        self.assertEqual(result["fn_at_fixed_07"], 1)
        self.assertIsNone(result["sigma_absolute_error_spearman"])
        self.assertEqual(metrics([1.], [1.2], [.01])["coverage90_raw"], 0)
        with self.assertRaises(ValueError):
            metrics([.3], [.3], [-.1])

    def test_expected_improvement_minimizes_loss_and_handles_zero_variance(self):
        values = expected_improvement(np.array([.1, .5]), np.zeros(2), .3)
        np.testing.assert_allclose(values, [.2, 0])
        self.assertGreater(expected_improvement([.2], [.1], .3)[0],
                           expected_improvement([.4], [.1], .3)[0])

    def test_search_methods_share_initial_trials_without_repeats(self):
        spec = dict(baseline=dict(length_multiplier=1., noise_variance=.1),
                    bounds=dict(length_multiplier=[.1, 10.], noise_variance=[.001, .3]),
                    grid_points_per_axis=5)
        points, coords, base = search_space(spec)
        initial = shared_start(len(points), base, 7, 4)
        bo = dict(length_scale=.25, alpha=1e-6, xi=0)
        arms = {}
        for method in ("random", "bo"):
            history = []
            for _ in range(8):
                index = next_point(coords, history, initial, method, 7, bo)
                self.assertNotIn(index, [r["index"] for r in history])
                history.append(dict(index=index, cv_mae=float(np.sum((coords[index]-.4)**2))))
            arms[method] = history
        self.assertEqual([h["index"] for h in arms["random"][:4]], initial)
        self.assertEqual([h["index"] for h in arms["bo"][:4]], initial)
        np.testing.assert_array_equal(points[base], [1., .1])

    def test_cv_returns_complete_out_of_fold_observation_predictions(self):
        groups = np.repeat(["a", "b", "c"], 9)
        station = np.tile(np.arange(9)*50., 3)
        xy = np.column_stack([station, np.repeat([0., 1., 2.], 9)])
        x = np.column_stack([station/400, np.ones(27)])
        y = .2+.5*x[:, 0]
        folds = purged_folds(groups, xy, station)
        score, mean, std = evaluate_cv(x, y, folds, dict(length_multiplier=1., noise_variance=.1))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue((std > 0).all())
        self.assertAlmostEqual(score["cv_mae"], np.mean([r["mae"] for r in score["folds"]]))


if __name__ == "__main__":
    unittest.main()
