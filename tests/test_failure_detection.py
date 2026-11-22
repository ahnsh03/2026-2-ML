"""Threshold, missing-target and one-class detection semantics."""
import unittest
import numpy as np
from quality_estimation.metrics import failure_detection_metrics

class FailureDetectionTests(unittest.TestCase):
    def test_threshold_precision_and_missing_labels(self):
        y=np.array([.7,.2,.9,np.nan],np.float32)
        prediction=np.array([.7,.8,.4,.1],np.float32)
        m=failure_detection_metrics(y,prediction)
        self.assertEqual((m['n'],m['undefined'],m['tp'],m['fp'],m['fn'],m['tn']),(3,1,0,1,1,1))
        self.assertEqual(m['recall'],0)
        self.assertEqual(m['auroc'],0)

    def test_single_class_metrics_are_explicit(self):
        m=failure_detection_metrics(np.array([.1,.2]),np.array([.3,.4]))
        self.assertIsNone(m['auroc'])
        self.assertEqual(m['average_precision'],1)
        self.assertEqual(m['recall'],1)
        m=failure_detection_metrics(np.array([.8,.9]),np.array([.8,.9]))
        self.assertIsNone(m['average_precision'])
        self.assertIsNone(m['recall'])
        self.assertIsNone(m['precision'])

    def test_invalid_estimate_does_not_silently_drop_rows(self):
        with self.assertRaises(ValueError):
            failure_detection_metrics(np.array([.2,np.nan]),np.array([.4,np.nan]))
        with self.assertRaises(ValueError):
            failure_detection_metrics(np.array([np.inf]),np.array([.4]))
