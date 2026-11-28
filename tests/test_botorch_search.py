import unittest
import numpy as np
from quality_estimation.botorch_search import next_log_expected_improvement


class BotorchSearchContractTest(unittest.TestCase):
    def test_initial_design_has_no_optional_dependency_and_skips_existing(self):
        index,info=next_log_expected_improvement([[0],[.5],[1]],[0],[.2],[0,2,1])
        self.assertEqual(index,2);self.assertEqual(info['stage'],'common_initial_design')

    def test_invalid_history_and_exhaustion_rejected_before_optional_import(self):
        for history,loss in [([0,0],[.1,.1]),([3],[.1]),([0],[float('nan')]),([0,1,2],[.1,.2,.3])]:
            with self.subTest(history=history):
                with self.assertRaises(ValueError):next_log_expected_improvement([[0],[.5],[1]],history,loss,[])
