import importlib.util
from pathlib import Path
import sys
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/scene_cycle_20261128/assess_confirmation.py'
original_path=sys.path.copy();sys.path.insert(0,str(path.parent))
try:
    spec=importlib.util.spec_from_file_location('confirmation_review_gate_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
finally:sys.path[:]=original_path


class ConfirmationReviewGateTests(unittest.TestCase):
    gate=dict(minimum_near_error_support=20,minimum_error_roads=2,
              minimum_utility_recall_ratio_vs_random=1.25,minimum_near_error_road_macro_recall_gain_vs_random=.02)
    random=dict(utility_recall=.1,near_error_road_macro=.1)

    def test_mathematical_boundary_is_not_rejected_by_float_rounding(self):
        value=dict(near_error_positives=20,near_error_positive_roads=2,utility_recall=.125,near_error_road_macro=.12)
        self.assertTrue(module.review_gate(value,self.random,self.gate)['passes'])

    def test_high_selected_value_does_not_override_insufficient_support(self):
        value=dict(near_error_positives=19,near_error_positive_roads=1,utility_recall=.9,near_error_road_macro=.9)
        result=module.review_gate(value,self.random,self.gate)
        self.assertFalse(result['passes']);self.assertIn('insufficient_near_error_support',result['failed_gates'])
        self.assertIn('insufficient_error_segments',result['failed_gates'])

    def test_undefined_recall_is_not_filled_with_zero_or_one(self):
        value=dict(near_error_positives=0,near_error_positive_roads=0,utility_recall=None,near_error_road_macro=None)
        self.assertFalse(module.review_gate(value,dict(utility_recall=None,near_error_road_macro=None),self.gate)['passes'])


if __name__=='__main__':unittest.main()
