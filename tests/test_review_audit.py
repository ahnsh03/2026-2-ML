import itertools
import unittest

import numpy as np

from quality_estimation.review_audit import estimate_window, summarize


class ReviewAuditTests(unittest.TestCase):
    def test_enumerating_every_uniform_draw_recovers_actual_population(self):
        fixed = [True, False]
        remaining = [True, True, None, False, False]
        for n in (1, 2, 4, 5):
            estimates = []
            for sample in itertools.combinations(remaining, n):
                w = estimate_window(7, fixed, list(sample))
                self.assertLessEqual(w["lower_low"], 3)
                self.assertGreaterEqual(w["upper_low"], 3)
                estimates.append(w["estimated_low"])
            self.assertAlmostEqual(np.mean(estimates), 3.)

    def test_zero_audit_with_unobserved_remainder_blocks_overall_point(self):
        census = estimate_window(2, [True, None], [])
        self.assertEqual(census["estimated_low"], 1)
        total = summarize([census, estimate_window(5, [False], [])])
        self.assertIsNone(total["estimated_low"])
        self.assertIsNone(total["estimated_missed_low"])
        self.assertEqual((total["lower_low"], total["upper_low"], total["windows_without_audit_support"]), (1, 5, 1))
        self.assertEqual(summarize([estimate_window(10, [True, False], [True, False])])["estimated_missed_low"], 3)
