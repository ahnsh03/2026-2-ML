from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import solve
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.conditioned_spatial_gp import ConditionedSpatialGP


class ConditionedSpatialGPTests(unittest.TestCase):
    def test_joint_covariance_and_real_label_update_match_dense_fixed_parameter_refit(
        self,
    ):
        rng = np.random.default_rng(94)
        x = rng.normal(size=(12, 3))
        y = rng.uniform(size=(12, 3))
        acquired = rng.normal(size=(2, 3))
        labels = rng.uniform(size=(2, 3))
        query = rng.normal(size=(4, 3))
        model = SpatialQualityGP(1.4, 0.08, 0.4).fit(x, y)
        covariance = model.posterior_cross_covariance(query, query)
        _, expected = model.predict(query, observation=False)
        np.testing.assert_allclose(
            np.stack([covariance[i, :, i, :] for i in range(4)]), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            model.posterior_cross_covariance(query, acquired),
            model.posterior_cross_covariance(acquired, query).transpose(2, 3, 0, 1),
            atol=1e-12,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.npz"
            model.save(path)
            updated = ConditionedSpatialGP.fit(path, acquired, labels)
            mean, cov = updated.predict(query, observation=False)
            extended = np.vstack(
                [model.X_, (acquired - model.x_mean_) / model.x_scale_]
            )
            target = ((np.vstack([y, labels]) - model.y_mean_) / model.y_scale_).ravel()
            query_x = (query - model.x_mean_) / model.x_scale_
            dense = np.kron(model.kernel(extended, extended), model.B_) + 0.08 * np.eye(
                42
            )
            cross = np.kron(model.kernel(query_x, extended), model.B_)
            dense_mean = (cross @ solve(dense, target, assume_a="pos")).reshape(
                4, 3
            ) * model.y_scale_ + model.y_mean_
            dense_cov = np.kron(
                model.kernel(query_x, query_x), model.B_
            ) - cross @ solve(dense, cross.T, assume_a="pos")
            np.testing.assert_allclose(mean, dense_mean, atol=1e-10)
            for i in range(4):
                expected_cov = (
                    dense_cov[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3]
                    * model.y_scale_[:, None]
                    * model.y_scale_[None, :]
                )
                np.testing.assert_allclose(cov[i], expected_cov, atol=1e-10)
                self.assertGreater(np.linalg.eigvalsh(cov[i]).min(), -1e-10)

    def test_undefined_labels_are_not_zero_and_saved_update_is_parent_bound(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(8, 3))
        y = rng.uniform(size=(8, 3))
        observed = rng.normal(size=(2, 3))
        query = rng.normal(size=(2, 3))
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.npz"
            SpatialQualityGP().fit(x, y).save(parent)
            a = ConditionedSpatialGP.fit(parent, observed[:1], [[0.9, np.nan, np.nan]])
            b = ConditionedSpatialGP.fit(
                parent, observed, [[0.9, np.nan, np.nan], [np.nan] * 3]
            )
            np.testing.assert_allclose(
                a.predict(query)[0], b.predict(query)[0], atol=1e-12
            )
            np.testing.assert_allclose(
                a.predict(query)[1], b.predict(query)[1], atol=1e-12
            )
            saved = Path(directory) / "update.npz"
            b.save(saved)
            loaded = ConditionedSpatialGP.load(saved, parent)
            np.testing.assert_allclose(
                loaded.predict(query)[0], b.predict(query)[0], atol=0
            )
            with parent.open("ab") as stream:
                stream.write(b"changed parent bytes")
            with self.assertRaisesRegex(ValueError, "parent model"):
                ConditionedSpatialGP.load(saved, parent)

    def test_invalid_or_missing_acquired_truth_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.npz"
            SpatialQualityGP().fit(
                np.arange(24).reshape(8, 3), np.full((8, 3), 0.5)
            ).save(path)
            for x, y in [
                (np.zeros((1, 3)), [[np.nan] * 3]),
                (np.full((1, 3), np.nan), [[0.5] * 3]),
                (np.zeros((1, 3)), [[1.1] * 3]),
                (np.zeros((33, 3)), np.full((33, 3), 0.5)),
            ]:
                with self.assertRaises(ValueError):
                    ConditionedSpatialGP.fit(path, x, y)


if __name__ == "__main__":
    unittest.main()
