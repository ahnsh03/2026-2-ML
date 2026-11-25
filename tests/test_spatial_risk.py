import unittest
import numpy as np
from quality_estimation.spatial_risk import (
    spatial_scores,
    conditional_batch,
    calibrate_moments,
    apply_calibration,
    gaussian_lower_probability,
)
from scipy.stats import multivariate_normal


class SpatialRiskTests(unittest.TestCase):
    def test_independent_zero_centered_thresholds_have_known_joint_probability(self):
        mean = np.array([[0.7, 0.8, 0.5]])
        cov = np.eye(3)[None] * 0.01
        score = spatial_scores(mean, cov)
        np.testing.assert_allclose(score["global_low"], 0.5)
        np.testing.assert_allclose(score["far_only"], 0.25)
        np.testing.assert_allclose(score["both_weak"], 0.25)
        np.testing.assert_allclose(score["normal_far_only"], 0.125)

    def test_batch_avoids_a_nearly_duplicate_measurement(self):
        covariance = np.array([[1.0, 0.999, 0], [0.999, 1.0, 0], [0, 0, 1.0]])
        selected = conditional_batch(
            np.ones(3), covariance, 2, np.array(["a", "b", "c"])
        )
        np.testing.assert_array_equal(selected, [0, 2])

    def test_calibration_preserves_positive_semidefinite_covariance(self):
        rng = np.random.default_rng(25)
        mean = np.full((100, 3), 0.5)
        cov = np.broadcast_to(
            np.array([[0.04, 0.01, 0], [0.01, 0.04, 0], [0, 0, 0.01]]), (100, 3, 3)
        ).copy()
        truth = np.clip(mean + rng.normal(0, 0.1, mean.shape) + 0.1, 0, 1)
        fit = calibrate_moments(mean, cov, truth)
        adjusted, newcov = apply_calibration(mean, cov, fit)
        self.assertTrue((np.linalg.eigvalsh(newcov) >= 0).all())
        np.testing.assert_allclose(np.median(truth - adjusted, axis=0), 0, atol=1e-12)

    def test_deterministic_quadrature_matches_reference_gaussian_probability(self):
        rng = np.random.default_rng(23)
        for dimensions in [2, 3]:
            for _ in range(8):
                a = rng.normal(size=(dimensions, dimensions))
                cov = a @ a.T + np.eye(dimensions) * 2
                mean = rng.normal(size=dimensions)
                bound = rng.normal(size=dimensions)
                expected = multivariate_normal.cdf(
                    bound, mean=mean, cov=cov, maxpts=500000, abseps=1e-7, releps=1e-7
                )
                actual = gaussian_lower_probability(bound, mean, cov)
                self.assertLess(abs(actual - expected), 5e-4)
                self.assertEqual(actual, gaussian_lower_probability(bound, mean, cov))


if __name__ == "__main__":
    unittest.main()
