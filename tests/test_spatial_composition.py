import unittest
import numpy as np
from quality_estimation.spatial_composition import independent_far_posterior


class SpatialCompositionTests(unittest.TestCase):
    def test_replacing_far_preserves_primary_and_cannot_reuse_its_cross_covariance(
        self,
    ):
        mean = np.array([[0.8, 0.9, 0.2]])
        covariance = np.array([[[0.2, 0.1, 0.08], [0.1, 0.3, 0.2], [0.08, 0.2, 0.4]]])
        # A new far process can be much more certain. Retaining the old .2
        # cross-covariance here would violate PSD.
        result, cov = independent_far_posterior(
            mean, covariance, np.array([[0.6]]), np.array([[[0.001]]])
        )
        np.testing.assert_array_equal(result[:, :2], mean[:, :2])
        np.testing.assert_array_equal(cov[:, :2, :2], covariance[:, :2, :2])
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), 0)
        self.assertEqual(cov[0, 1, 2], 0)
        self.assertEqual(mean[0, 2], 0.2)

    def test_negative_far_variance_or_invalid_primary_covariance_is_rejected(self):
        for old, far in [
            (np.eye(3), -0.01),
            (np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]]), 0.1),
        ]:
            with self.assertRaises(ValueError):
                independent_far_posterior(
                    np.ones((1, 3)), old[None], np.ones((1, 1)), np.array([[[far]]])
                )


if __name__ == "__main__":
    unittest.main()
