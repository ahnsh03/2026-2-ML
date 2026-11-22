import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from quality_estimation.evaluate_scene_conditions import check_holdout, condition_label, descriptors, scores


class SceneConditionTests(unittest.TestCase):
    def test_metadata_and_missing_hour_zero(self):
        self.assertEqual(condition_label("WEATHER_TYPE_CLOUDY", "weather"), 1)
        self.assertEqual(condition_label("00", "hour"), 0)
        self.assertIsNone(condition_label("unknown", "weather"))
        for value in (True, -1, 24, 1.0):
            with self.assertRaises(ValueError):
                condition_label(value, "hour")

    def test_protected_roles_are_rejected_before_reading_a_training_rgb(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text('{"preregistered_quality_role":"final_test"}\n')
            with self.assertRaisesRegex(ValueError, "protected"):
                descriptors(path, Path(tmp), training=True)

    def test_cross_split_leakage_is_rejected_for_each_identity(self):
        train = dict(sample_id="a", parent_session_id="first", pixel_sha256="abc")
        test = dict(sample_id="b", parent_session_id="second", pixel_sha256="def")
        check_holdout([train], [test])
        for key in train:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                check_holdout([train], [dict(test, **{key: train[key]})])

    def test_unseen_classes_and_unknown_conditions(self):
        rows = [dict(hour=h) for h in (13, 13, 20, None)]
        metrics = scores(rows, "hour", [13, 8, 13, 8], [8, 13])
        self.assertEqual(metrics["labels"], [8, 13, 20])
        self.assertEqual(metrics["per_class_recall"], [None, .5, 0.])
        self.assertEqual(metrics["confusion_true_by_predicted"], [[0, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertEqual(metrics["balanced_accuracy"], .25)
        self.assertEqual(metrics["unseen_in_training"], 1)
        self.assertEqual(metrics["unknown"], 1)
        self.assertIsNone(scores([dict(hour=None)], "hour", [13], [8, 13])["accuracy"])
