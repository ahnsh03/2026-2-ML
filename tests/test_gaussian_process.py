import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from quality_estimation.features import FEATURE_NAMES, feature_schema
from quality_estimation.gaussian_process import fit_quality_gp
from quality_estimation.model_io import write_prediction_csv
from quality_estimation.models import make_estimator, predict_array


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

    def test_optional_model_round_trip_keeps_iou_units_and_point_columns(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(30, len(FEATURE_NAMES)))
        y = np.clip(x[:, 0] * .2 + .65, 0, 1)
        est = make_estimator("GaussianProcess", {"length_multiplier": 2.0}, seed=0).fit(x, y)
        bundle = dict(model="GaussianProcess", estimator=est, ordered_features=list(FEATURE_NAMES),
                      feature_schema=feature_schema(), column_indices=list(range(x.shape[1])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gp.joblib"
            joblib.dump(bundle, path)
            pred = predict_array(joblib.load(path), x[:3])
            mean, sigma = fit_quality_gp(x, y, length_multiplier=2).predict(x[:3], return_std=True)
            np.testing.assert_allclose(mean, pred["raw_estimate"])
            np.testing.assert_allclose(sigma, pred["prediction_std"])
            csv_path = Path(tmp) / "predictions.csv"
            write_prediction_csv(csv_path, ["a", "b", "c"], np.array([y[0], np.nan, y[2]]),
                                 {"GaussianProcess": pred})
            with csv_path.open() as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(saved[1]["actual_iou"], "")
            self.assertEqual(float(saved[2]["GaussianProcess_std"]), sigma[2])
