import unittest
import numpy as np
from scipy.special import ndtr
from quality_estimation.diagnostic_targets import auxiliary_targets,quality_or_near_defect_probability,band_diagnostic_probabilities,gt_review_flags

class DiagnosticTargetsTest(unittest.TestCase):
    def test_high_recall_does_not_hide_false_positives(self):
        c=np.zeros((1,9,3),int);c[0,6]=[93,108,7];c[0,4]=[40,10,60]
        y=auxiliary_targets(c,[.4]);self.assertAlmostEqual(y[0,1],.93);self.assertLess(y[0,3],.65)
        c[0,6]=[0,0,100];self.assertEqual(auxiliary_targets(c,[0])[0,3],0.)
    def test_independent_union_matches_closed_form(self):
        m=np.array([[.7,.9,.5,.8]]);c=np.diag([.01,.01,.01,.01])[None]
        p=quality_or_near_defect_probability(m,c);self.assertAlmostEqual(p[0],.75,places=7)
        c[0,0,3]=c[0,3,0]=.008
        self.assertLess(quality_or_near_defect_probability(m,c)[0],p[0])
    def test_four_task_near_f1_detects_false_positives_hidden_by_recall(self):
        m=np.array([[.6,.93,.4,.62]]);c=np.eye(4)[None]*.00001
        p4=band_diagnostic_probabilities(m,c);p3=band_diagnostic_probabilities(m[:,:3],c[:,:3,:3])
        self.assertGreater(p4[0,3],.999);self.assertGreater(p3[0,2],.999)
        np.testing.assert_allclose(p4.sum(1),1)
    def test_post_gt_taxonomy_requires_near_precision_and_never_certifies_safety(self):
        c=np.zeros((9,3),int);c[6]=[100,0,0];c[4]=[20,0,180]
        flags=gt_review_flags(c,.4)
        self.assertTrue(flags['near_retained_far_reduced']);self.assertTrue(flags['benchmark_low_iou']);self.assertFalse(flags['operating_failure_established'])
        c[6,1]=200;flags=gt_review_flags(c,.24)
        self.assertFalse(flags['near_retained_far_reduced']);self.assertTrue(flags['near_false_positive_issue'])
    def test_hallucinated_lane_can_have_precision_support_without_gt_support(self):
        c=np.zeros((9,3),int);c[6]=[0,200,0];flags=gt_review_flags(c,0.)
        self.assertTrue(flags['near_false_positive_issue']);self.assertIsNone(flags['near_omission'])
        self.assertFalse(flags['valid_for_low_iou_benchmark']);self.assertIsNone(flags['benchmark_low_iou']);self.assertIsNone(flags['near_precision_and_recall_retained'])
        self.assertTrue(flags['valid_for_union_supported_benchmark']);self.assertTrue(flags['union_supported_low_iou'])
    def test_threshold_uses_exact_counts_rather_than_float32_roundoff(self):
        c=np.zeros((9,3),int);c[6]=[70,0,30]
        flags=gt_review_flags(c,np.float32(.7))
        self.assertFalse(flags['benchmark_low_iou']);self.assertEqual(flags['exact_whole_iou'],.7)
        with self.assertRaisesRegex(ValueError,'counts differ'):gt_review_flags(c,.6)

if __name__=='__main__':unittest.main()
