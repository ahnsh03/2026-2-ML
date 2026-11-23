import copy
import unittest

from quality_estimation.cycle_gate import assess_update, check_paired, condition_deltas, head_support_diagnostics


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


    def test_quality_mae_by_condition_uses_each_versions_defined_frames(self):
        rows = {}
        for sid, hour, iou in [("a", 13, .7), ("b", 13, .6), ("c", 23, .4), ("d", 5, None)]:
            rows[sid] = {"condition": {"hour": hour}, "metrics": {"lane": {"iou": iou}}}
        old = {"a": .03, "b": .01, "c": .10}
        new = {"a": .07, "b": .05, "c": .02}
        groups = condition_deltas(rows, rows, paired_quality_errors=(old, new))
        day = groups['{"hour": 13}']
        self.assertAlmostEqual(day["incumbent_quality_mae"], .02)
        self.assertAlmostEqual(day["candidate_quality_mae"], .06)
        self.assertAlmostEqual(day["quality_mae_delta"], .04)
        self.assertAlmostEqual(groups['{"hour": 23}']["quality_mae_delta"], -.08)
        self.assertIsNone(groups['{"hour": 5}']["candidate_quality_mae"])
        self.assertIsNone(groups['{"hour": 5}']["quality_mae_delta"])
        self.assertNotIn("quality_mae_delta", condition_deltas(rows, rows)['{"hour": 13}'])

    def test_present_frame_iou_cannot_hide_new_predictions_on_absent_ground_truth(self):
        def scene(tp, fp, fn):
            union = tp + fp + fn
            return {"metrics": {"stopline": {"tp": tp, "fp": fp, "fn": fn, "iou": tp / union if union else None}}}
        left = {"present": scene(3, 1, 1), "added": scene(0, 0, 0), "removed": scene(0, 5, 0)}
        right = {"present": scene(4, 0, 0), "added": scene(0, 2, 0), "removed": scene(0, 0, 0)}
        head = head_support_diagnostics(left, right)["stopline"]
        self.assertEqual((head["gt_present_frames"], head["gt_absent_frames"]), (1, 2))
        self.assertEqual((head["new_false_positive_frames"], head["resolved_false_positive_frames"]), (1, 1))
        self.assertAlmostEqual(head["incumbent"]["gt_present_mean_iou"], .6)
        self.assertEqual(head["candidate"]["gt_present_mean_iou"], 1.)
        self.assertEqual(head["candidate"]["gt_absent_fp_pixels"], 2)
        self.assertAlmostEqual(head["candidate"]["pixel_precision"], 4 / 6)
        self.assertEqual(head["candidate"]["pixel_recall"], 1.)

    def test_empty_class_support_has_no_pixel_recall_or_frame_mean(self):
        empty = {"a": {"metrics": {"yellow": {"tp": 0, "fp": 0, "fn": 0, "iou": None}}}}
        head = head_support_diagnostics(empty, empty)["yellow"]
        self.assertEqual(head["gt_absent_frames"], 1)
        self.assertIsNone(head["candidate"]["gt_present_mean_iou"])
        self.assertIsNone(head["candidate"]["pixel_recall"])
        self.assertIsNone(head["candidate"]["pixel_precision"])
        self.assertIsNone(head["candidate"]["pooled_pixel_iou"])
