"""Leakage guards and selection accounting, not fitted-score expectations."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from quality_estimation.dataset_io import QualityDataset
from quality_estimation.features import FEATURE_NAMES, feature_schema
from quality_estimation.metrics import regression_metrics, selection_metrics
from quality_estimation.models import RankStableOLS, make_estimator, predict_array
from quality_estimation.splits import build_split


class MetricTests(unittest.TestCase):
    def test_strict_failure_threshold_does_not_change_after_float32_storage(self):
        for dtype in (np.float32, np.float64):
            for threshold in (.3, .7, .9):
                y=np.array([threshold,threshold-.01,threshold+.01,np.nan],dtype=dtype)
                result=selection_metrics(y,['at','below','above','undefined'],{'ML':[4,3,2,1]},
                                         budgets=[1.],threshold=threshold,repeats=5)
                for method in ('ML','Oracle'):
                    self.assertEqual(result['methods'][method][0]['found_failures'],1)
                    self.assertEqual(result['methods'][method][0]['selected_undefined'],1)
                self.assertEqual(result['methods']['Random'][0]['total_failures'],1)

    def test_undefined_selection_consumes_budget(self):
        result = selection_metrics(np.array([.1, np.nan, .9, .2]), ["a", "b", "c", "d"],
                                   {"ML": [3, 4, 1, 0]}, budgets=[.5], repeats=10)
        row = result["methods"]["ML"][0]
        self.assertEqual((row["selected"], row["found_failures"], row["selected_undefined"]), (2, 1, 1))
        self.assertEqual(row["recall"], .5)
        self.assertEqual(row["precision"], .5)

    def test_ties_use_identifiers_and_budget_rounds_up(self):
        result = selection_metrics(np.array([.1, .9, .2]), ["c", "a", "b"],
                                   {"ML": [1, 1, 1]}, budgets=[.34], repeats=5)
        row = result["methods"]["ML"][0]
        self.assertEqual(row["selected"], 2)
        self.assertEqual(row["found_failures"], 1)

    def test_no_failures_is_undefined_recall(self):
        result = selection_metrics(np.array([.9, .8]), ["a", "b"], {"ML": [0, 1]}, repeats=5)
        self.assertIsNone(result["methods"]["ML"][0]["recall"])
        self.assertIsNone(result["methods"]["Random"][0]["recall_mean"])

    def test_regression_clips_prediction_and_preserves_missing_truth(self):
        result = regression_metrics([0., 1., np.nan], [-1., 2., .5])
        self.assertEqual(result["clipped_MAE"], 0)
        self.assertEqual(result["raw_MAE"], 1)
        self.assertEqual(result["undefined"], 1)


class LearningTests(unittest.TestCase):
    def test_redundant_feature_roundoff_does_not_create_unbounded_ols_coefficients(self):
        rng = np.random.RandomState(32)
        base = rng.normal(size=(100,2))
        X = np.column_stack([base, base.sum(axis=1)+rng.normal(scale=1e-8,size=100)])
        y = .2*base[:,0]-.1*base[:,1]+rng.normal(scale=.001,size=100)
        model = RankStableOLS(rcond=1e-6).fit(X,y)
        self.assertEqual(model.rank_,2)
        self.assertLess(np.max(np.abs(model.coef_)),1)
        test = X.copy(); test[:,2] += .001
        self.assertLess(np.max(np.abs(model.predict(test)-model.predict(X))),.001)

    def test_scaler_only_sees_training_rows_and_inference_needs_no_gt(self):
        X = np.zeros((4, 56), np.float32)
        X[:, 0] = [0, 1, 2, 3]
        y = np.array([0., .2, .4, .6])
        model = make_estimator("Ridge", {"alpha": 1.}, 0)
        model.fit(X, y)
        before = model.named_steps["scale"].mean_.copy()
        bundle = {"estimator": model, "column_indices": list(range(56)),
                  "ordered_features": list(FEATURE_NAMES), "feature_schema": feature_schema()}
        test = X.copy(); test[:, 0] = 100
        result = predict_array(bundle, test)
        np.testing.assert_array_equal(before, model.named_steps["scale"].mean_)
        self.assertTrue(np.all(result["estimated_iou"] <= 1))
        self.assertTrue(np.all(result["estimated_iou"] >= 0))
        with self.assertRaisesRegex(ValueError, "order"):
            predict_array(bundle, test, list(reversed(FEATURE_NAMES)))

    def test_source_pixel_overlap_cannot_cross_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); (p / "dataset.json").write_text("{}")
            samples = [{"sample_id": str(i), "run_id": str(i), "parent_session_id": str(i),
                        "source_role": "after_f0_training", "input_sha256": {"rgb": "same"}}
                       for i in range(2)]
            data = QualityDataset(p, np.zeros((2,56)), np.zeros(2), np.ones(2,bool), ["0","1"], samples, {}, list(FEATURE_NAMES))
            with self.assertRaisesRegex(ValueError, "cross roles"):
                build_split(data, {"0": "meta_train", "1": "meta_dev"})

    def test_reserved_final_group_cannot_be_reassigned_for_training_or_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'dataset.json').write_text('{}')
            sample=dict(sample_id='one',run_id='run',parent_session_id='reserved',
                        source_role='after_f0_training',input_sha256={'rgb':'unique'},preregistered_quality_role='final_test')
            data=QualityDataset(p,np.zeros((1,56)),np.zeros(1),np.ones(1,bool),['one'],[sample],{},list(FEATURE_NAMES))
            for role in ('meta_train','meta_dev','diagnostic_only'):
                with self.assertRaisesRegex(ValueError,'preregistered'):
                    build_split(data,{'reserved':role})
            self.assertEqual(build_split(data,{'reserved':'final_test'})['counts'],{'final_test':1})

    def test_exposure_override_never_allows_final_test(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'dataset.json').write_text('{}')
            sample=dict(sample_id='one',run_id='run',parent_session_id='seen',
                        source_role='f0_train',input_sha256={'rgb':'one'})
            data=QualityDataset(p,np.zeros((1,56)),np.zeros(1),np.ones(1,bool),['one'],[sample],{},list(FEATURE_NAMES))
            with self.assertRaisesRegex(ValueError,'final_test'):
                build_split(data,{'seen':'final_test'},allow_exposed=True)

    def test_thinning_uses_motion_without_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); (p / "dataset.json").write_text("{}")
            samples = [{"sample_id": str(i), "run_id": "r", "parent_session_id": "p", "frame_id": i,
                        "source_role": "after_f0_training", "input_sha256": {"rgb": str(i)},
                        "ego": {"position": {"x": x, "y": 0}, "heading_deg": 0}}
                       for i,x in enumerate([0,0,2,6])]
            data = QualityDataset(p, np.zeros((4,56)), np.full(4,np.nan), np.zeros(4,bool), [str(i) for i in range(4)], samples, {}, list(FEATURE_NAMES))
            split = build_split(data, {"p": "meta_train"}, min_distance=5.)
            self.assertEqual([a["eligible"] for a in split["assignments"]], [True,False,False,True])


if __name__ == "__main__":
    unittest.main()
