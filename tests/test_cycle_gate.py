import copy
import unittest

from quality_estimation.cycle_gate import assess_update, check_paired, condition_deltas


class CycleGateTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(schema_version="ml-cycle-acceptance-1", minimum_new_lane_gain=0.,
                           maximum_retention_lane_loss=.005, maximum_quality_mae_increase=.01)
        self.base = dict(new_lane_iou=.6, retention_lane_iou=.8, quality_mae=.1)

    def test_new_improvement_cannot_override_retention_regression(self):
        result = assess_update(self.base, dict(new_lane_iou=.7, retention_lane_iou=.77, quality_mae=.08), self.policy)
        self.assertFalse(result['promotion_eligible'])
        self.assertEqual(result['failed_checks'], ['historical_retention'])
        self.assertFalse(result['deployment_performed'])

    def test_matched_quality_model_must_also_remain_accurate(self):
        result = assess_update(self.base, dict(new_lane_iou=.7, retention_lane_iou=.8, quality_mae=.12), self.policy)
        self.assertEqual(result['failed_checks'], ['matched_quality_accuracy'])

    def test_within_all_research_bounds_is_eligible_without_deploying(self):
        result = assess_update(self.base, dict(new_lane_iou=.7, retention_lane_iou=.798, quality_mae=.105), self.policy)
        self.assertTrue(result['promotion_eligible'])
        self.assertFalse(result['deployment_performed'])

    def test_nonfinite_or_missing_metrics_are_rejected(self):
        for bad in (dict(self.base, quality_mae=float('nan')), dict(quality_mae=.1)):
            with self.assertRaises(ValueError): assess_update(self.base, bad, self.policy)

    def test_invalid_acceptance_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_update(self.base, self.base, dict(self.policy, maximum_retention_lane_loss=-.1))

    def paired(self):
        return {'a':dict(input_sha256={'rgb':'r','semantic':'g'},
                         metrics={'lane':dict(tp=8, fp=1, fn=2, iou=8/11)})}

    def test_same_gt_can_have_different_prediction_counts(self):
        left=self.paired();right=copy.deepcopy(left)
        right['a']['metrics']['lane']=dict(tp=9,fp=1,fn=1,iou=9/11)
        check_paired(left,right)

    def test_changed_input_or_gt_support_cannot_be_a_paired_comparison(self):
        left=self.paired()
        for key in ('rgb','semantic'):
            right=copy.deepcopy(left);right['a']['input_sha256'][key]='changed'
            with self.assertRaisesRegex(ValueError,'RGB/GT'):check_paired(left,right)
        right=copy.deepcopy(left);right['a']['metrics']['lane']['fn']=3
        with self.assertRaisesRegex(ValueError,'GT support'):check_paired(left,right)

    def test_different_sample_membership_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'rows/order'):check_paired(self.paired(),{})

    def test_changed_condition_metadata_is_rejected(self):
        left=self.paired();right=copy.deepcopy(left);right['a']['condition']={'hour':20}
        with self.assertRaisesRegex(ValueError,'condition'):check_paired(left,right)

    def test_subgroup_regression_remains_visible_when_pooled_mean_improves(self):
        left=self.paired();left['a']['condition']={'hour':13}
        left['b']=copy.deepcopy(left['a']);left['b']['condition']={'hour':20}
        left['a']['metrics']['lane']['iou']=.8;left['b']['metrics']['lane']['iou']=.4
        right=copy.deepcopy(left);right['a']['metrics']['lane']['iou']=.77;right['b']['metrics']['lane']['iou']=.6
        changes=list(condition_deltas(left,right).values())
        self.assertAlmostEqual(changes[0]['delta'],-.03)
        self.assertAlmostEqual(changes[1]['delta'],.2)

    def test_undefined_subgroup_is_reported_without_inventing_zero_iou(self):
        left=self.paired();left['a']['metrics']['lane']['iou']=None
        result=next(iter(condition_deltas(left,copy.deepcopy(left)).values()))
        self.assertEqual(result['defined_rows'],0)
        self.assertIsNone(result['delta'])
