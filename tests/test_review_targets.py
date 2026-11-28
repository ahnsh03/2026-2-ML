import unittest
import numpy as np
from quality_estimation.review_targets import support_aware_targets,gaussian_utility


class ReviewTargetsTest(unittest.TestCase):
    def test_tiny_far_gt_can_have_low_iou_without_supported_review_error(self):
        counts=np.zeros((1,9,3),int);counts[0,4,2]=10
        r=support_aware_targets(counts)
        self.assertEqual(r['whole_frame_iou'][0],0)
        self.assertEqual(r['review_utility'][0],0)
        self.assertFalse(r['near']['supported'][0])

    def test_total_near_omission_is_retained_even_with_empty_prediction(self):
        counts=np.zeros((1,9,3),int);counts[0,6,2]=100
        r=support_aware_targets(counts)
        self.assertTrue(r['near']['false_negative'][0]);self.assertEqual(r['review_utility'][0],1)

    def test_unmarked_near_false_positive_is_an_error(self):
        counts=np.zeros((1,9,3),int);counts[0,7,1]=100
        r=support_aware_targets(counts)
        self.assertTrue(r['near']['false_positive'][0]);self.assertEqual(r['scene_gt_class'][0],0)
        self.assertEqual(r['review_utility'][0],1)

    def test_far_only_and_offroad_events_have_explicit_utilities(self):
        counts=np.zeros((2,9,3),int);counts[0,4,2]=100;counts[1,8,1]=100
        r=support_aware_targets(counts)
        np.testing.assert_allclose(r['review_utility'],[.25,.5])
        np.testing.assert_array_equal(r['scene_gt_class'],[1,0])

    def test_empty_union_remains_undefined_without_fabricating_safety_label(self):
        r=support_aware_targets(np.zeros((1,9,3)))
        self.assertTrue(np.isnan(r['whole_frame_iou'][0]));self.assertEqual(r['review_utility'][0],0)

    def test_posterior_utility_uses_cross_task_covariance(self):
        mean=np.array([[.2,.8,.1]]);cov=np.array([[[.04,.01,0],[.01,.09,0],[0,0,.01]]])
        u,sd=gaussian_utility(mean,cov)
        self.assertAlmostEqual(u[0],.45)
        self.assertAlmostEqual(sd[0]**2,.04+.25**2*.09+.5**2*.01+2*.25*.01)

    def test_rejects_invalid_counts_and_negative_contrast_variance(self):
        with self.assertRaises(ValueError):support_aware_targets(-np.ones((1,9,3)))
        with self.assertRaises(ValueError):support_aware_targets(np.zeros((1,10,3)))
        with self.assertRaises(ValueError):gaussian_utility(np.zeros((1,3)),-np.eye(3)[None])
