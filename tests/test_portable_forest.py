import tempfile, unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from quality_estimation.portable_forest import PortableForest


class PortableForestTests(unittest.TestCase):
    def test_regression_export_reload_and_float32_boundary_routing(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(301, 3))
        y = np.sin(x[:, 0]) + x[:, 1] ** 2
        forest = RandomForestRegressor(n_estimators=9, max_depth=5, random_state=1).fit(
            x, y
        )
        portable = PortableForest.from_sklearn(forest, ["a", "b", "c"])
        threshold = forest.estimators_[0].tree_.threshold[0]
        feature = forest.estimators_[0].tree_.feature[0]
        probes = x.copy()
        probes[:2, feature] = [
            np.nextafter(threshold, -np.inf),
            np.nextafter(threshold, np.inf),
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "forest.npz"
            portable.save(p)
            loaded = PortableForest.load(p)
            np.testing.assert_allclose(
                loaded.predict(probes), forest.predict(probes), atol=1e-12, rtol=0
            )

    def test_classifier_probabilities_and_nonconsecutive_labels_match_sklearn(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(150, 3))
        y = np.asarray([5, 8, 20])[
            (x[:, 0] > 0.3).astype(int) + (x[:, 1] > 0.4).astype(int)
        ]
        forest = RandomForestClassifier(
            n_estimators=11, min_samples_leaf=2, random_state=8
        ).fit(x, y)
        model = PortableForest.from_sklearn(forest, ["a", "b", "c"])
        np.testing.assert_array_equal(model.predict(x), forest.predict(x))
        np.testing.assert_allclose(
            model.predict_proba(x), forest.predict_proba(x), atol=1e-12, rtol=0
        )

    def test_wrong_order_nonfinite_input_and_cyclic_graph_are_rejected(self):
        x = np.arange(30).reshape(10, 3)
        model = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=1).fit(
            x, np.arange(10)
        )
        portable = PortableForest.from_sklearn(
            model,
            ["a", "b", "c"],
            dict(profile="roi9", regions_sha256="mask", checkpoint_sha256="cp"),
        )
        with self.assertRaises(ValueError):
            portable.validate_context(["b", "a", "c"], "roi9", "mask", "cp")
        with self.assertRaises(ValueError):
            portable.predict(np.full((1, 3), np.nan))
        portable.left[0] = 0
        with self.assertRaises(ValueError):
            portable._validate()


if __name__ == "__main__":
    unittest.main()
