import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from quality_estimation.spatial_gp import (
    SpatialQualityGP,
    low_quality_probability,
    expected_shortfall,
)


class SpatialGPTests(unittest.TestCase):
    def test_posterior_matches_dense_multitask_gaussian_conditioning(self):
        rng = np.random.default_rng(25)
        x = rng.normal(size=(14, 3))
        y = rng.uniform(size=(14, 3))
        test = rng.normal(size=(5, 3))
        m = SpatialQualityGP(1.4, 0.08, 0.6).fit(x, y)
        mean, cov = m.predict(test, observation=False, batch_size=2)
        k = m.kernel(m.X_, m.X_)
        cross = m.kernel((test[:, m.selected_] - m.x_mean_) / m.x_scale_, m.X_)
        dense = np.kron(k, m.B_) + 0.08 * np.eye(42)
        target = ((y - m.y_mean_) / m.y_scale_).ravel()
        for i in range(len(test)):
            a = np.kron(cross[i : i + 1], m.B_)
            expected = (
                a @ solve(dense, target, assume_a="pos")
            ) * m.y_scale_ + m.y_mean_
            expected_cov = m.B_ - a @ solve(dense, a.T, assume_a="pos")
            expected_cov *= m.y_scale_[:, None] * m.y_scale_[None, :]
            np.testing.assert_allclose(mean[i], expected, atol=1e-10)
            np.testing.assert_allclose(cov[i], expected_cov, atol=1e-10)
        _, observation = m.predict(test)
        np.testing.assert_allclose(
            observation - cov,
            np.broadcast_to(np.diag(0.08 * m.y_scale_**2), cov.shape),
            atol=1e-12,
        )
        mu, std = m.contrast(mean, cov)
        np.testing.assert_allclose(mu, mean[:, 1] - mean[:, 2])
        self.assertTrue((std >= 0).all())
        window = m.posterior_covariance(test, [0, 1, -1])
        np.testing.assert_allclose(np.diag(window), std**2, atol=1e-11)
        transformed = (test[:, m.selected_] - m.x_mean_) / m.x_scale_
        physical = np.array([0, 1, -1]) * m.y_scale_
        dense_cross = np.kron(cross, m.B_)
        dense_posterior = np.kron(
            m.kernel(transformed, transformed), m.B_
        ) - dense_cross @ solve(dense, dense_cross.T, assume_a="pos")
        projection = np.kron(np.eye(len(test)), physical[None, :])
        np.testing.assert_allclose(
            window, projection @ dense_posterior @ projection.T, atol=1e-10
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.npz"
            m.save(p)
            other = SpatialQualityGP.load(p)
            np.testing.assert_allclose(
                other.predict(test)[0], m.predict(test)[0], atol=0
            )
            np.testing.assert_allclose(
                other.predict(test)[1], m.predict(test)[1], atol=0
            )

    def test_zero_uncertainty_is_deterministic_and_shortfall_has_analytic_midpoint(
        self,
    ):
        np.testing.assert_array_equal(
            low_quality_probability([0.6, 0.7, 0.8], [0, 0, 0]), [1, 0, 0]
        )
        np.testing.assert_allclose(expected_shortfall([0.6, 0.8], [0, 0]), [0.1, 0])
        self.assertAlmostEqual(
            float(expected_shortfall(0.7, 0.2)), 0.2 / np.sqrt(2 * np.pi)
        )

    def test_group_balancing_is_invariant_to_duplicating_an_entire_feature_group(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(18, 3))
        y = rng.uniform(size=(18, 2))
        test = rng.normal(size=(4, 3))
        a = SpatialQualityGP(feature_groups=["appearance", "appearance", "output"]).fit(
            x, y
        )
        b = SpatialQualityGP(feature_groups=["appearance"] * 4 + ["output"]).fit(
            x[:, [0, 1, 0, 1, 2]], y
        )
        ma, ca = a.predict(test)
        mb, cb = b.predict(test[:, [0, 1, 0, 1, 2]])
        np.testing.assert_allclose(ma, mb, atol=1e-10)
        np.testing.assert_allclose(ca, cb, atol=1e-10)

    def test_additive_kernel_is_weighted_sum_and_matches_dense_conditioning(self):
        rng = np.random.default_rng(51)
        x = rng.normal(size=(13, 4))
        y = rng.uniform(size=(13, 3))
        test = rng.normal(size=(3, 4))
        model = SpatialQualityGP(
            1.1,
            0.07,
            0.4,
            feature_groups=["appearance"] * 2 + ["prediction"] * 2,
            kernel_mode="additive",
            kernel_group_weights=[0.3, 0.7],
        ).fit(x, y)
        transformed = (test - model.x_mean_) / model.x_scale_
        k = model.kernel(model.X_, model.X_)
        cross = model.kernel(transformed, model.X_)
        from scipy.spatial.distance import cdist

        expected = np.zeros_like(cross)
        for ix, weight in [(slice(0, 2), 0.3), (slice(2, 4), 0.7)]:
            radius = (
                np.sqrt(3)
                * cdist(transformed[:, ix], model.X_[:, ix])
                / (1.1 * np.sqrt(2))
            )
            expected += weight * (1 + radius) * np.exp(-radius)
        np.testing.assert_allclose(cross, expected, atol=1e-12)
        dense = np.kron(k, model.B_) + 0.07 * np.eye(39)
        target = ((y - model.y_mean_) / model.y_scale_).ravel()
        prediction = (
            np.kron(cross, model.B_) @ solve(dense, target, assume_a="pos")
        ).reshape(3, 3)
        prediction = prediction * model.y_scale_ + model.y_mean_
        np.testing.assert_allclose(model.predict(test)[0], prediction, atol=1e-10)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "additive.npz"
            model.save(path)
            np.testing.assert_allclose(
                SpatialQualityGP.load(path).predict(test)[0],
                model.predict(test)[0],
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
