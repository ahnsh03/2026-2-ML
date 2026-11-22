import json
import unittest

import numpy as np

from quality_estimation.calibration import fit_group_cutoffs, group_alarm_thresholds
from quality_estimation.environment_clusters import assign_environment_clusters, fit_environment_clusters
from quality_estimation.environment_features import NAMES


class EnvironmentPolicyTests(unittest.TestCase):
    def test_unseen_and_sparse_groups_use_global(self):
        y = np.array([.8, .8, .9, .85, .8, .2, np.nan], np.float32)
        pred = [.8, .9, .9, .6, .7, .3, .1]
        policy = fit_group_cutoffs(y, pred, [0, 0, 0, 1, 1, 1, 2], min_normals=3)
        np.testing.assert_allclose(group_alarm_thresholds(policy, [0, 1, 2, 19]), [.8, .6, .6, .6])

    def test_center_assignments_roundtrip_with_constant_columns(self):
        x = np.zeros((6, len(NAMES)))
        x[:3, 0] = 1
        spec = json.loads(json.dumps(fit_environment_clusters(x, 2)))
        labels = assign_environment_clusters(spec, x)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertNotEqual(labels[0], labels[-1])
        spec["features"].reverse()
        with self.assertRaises(ValueError):
            assign_environment_clusters(spec, x)

    def test_group_cutoffs_respect_the_total_budget_when_sparse_normals_trigger_fallback(self):
        scores = np.r_[np.linspace(.50, .98, 25), [.1, .2, .3, .4], np.full(11, .95)]
        groups = np.r_[np.zeros(25, int), np.ones(15, int)]
        y = np.full(40, .8, dtype=np.float32)
        policy = fit_group_cutoffs(y, scores, groups, min_normals=20)
        self.assertEqual(policy["combined_false_positives_before_cap"], 6)
        self.assertEqual(policy["combined_false_positives"], 4)
        self.assertEqual(int((scores < group_alarm_thresholds(policy, groups)).sum()), 4)
        self.assertEqual(group_alarm_thresholds(policy, [1, 12]).tolist(), [.5, .5])
