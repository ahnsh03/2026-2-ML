import unittest
import numpy as np
from quality_estimation.support_review_selection import select_supported_review,conditional_variance_audit


class SupportReviewSelectionTest(unittest.TestCase):
    def test_conditional_information_avoids_redundant_high_variance_frames(self):
        ids=np.array(['a','b','c']);cov=np.array([[1,.99,0],[.99,1,0],[0,0,.5]])
        chosen=conditional_variance_audit(ids,cov,[0],1,.01)
        np.testing.assert_array_equal(chosen,[2])

    def test_exact_budget_and_honest_random_vs_information_metadata(self):
        ids=np.array([f'i{i:02d}'for i in range(40)]);mean=np.arange(40)/40;sd=np.ones(40);groups=np.repeat([0,1],20)
        selected,score,detail=select_supported_review(ids,mean,sd,groups,20,balanced_floor=.25,
            audit_fraction=.1,information_fraction=.2,information_mode='conditional',covariance=np.eye(40),noise_variance=.05)
        self.assertEqual(len(set(selected.tolist())),20)
        self.assertEqual(detail['information_slots'],4);self.assertEqual(detail['random_audit_slots'],2)
        self.assertEqual(detail['reasons'][-2:],['random_audit']*2)
        self.assertAlmostEqual(detail['conditional_inclusion_probability'][-1],2/22)

    def test_selection_is_invariant_to_input_permutation(self):
        ids=np.array([f'i{i:02d}'for i in range(30)]);mean=np.repeat([.1,.5,1.],10);sd=np.linspace(1,0,30);group=np.tile([0,1,2],10)
        p=np.random.default_rng(1).permutation(30)
        first=select_supported_review(ids,mean,sd,group,10,audit_fraction=0.,information_fraction=.2)[0]
        other=select_supported_review(ids[p],mean[p],sd[p],group[p],10,audit_fraction=0.,information_fraction=.2)[0]
        np.testing.assert_array_equal(ids[first],ids[p][other])

    def test_missing_covariance_or_corrupt_budget_rejected(self):
        ids=np.array(['a','b','c','d']);x=np.ones(4)
        with self.assertRaises(ValueError):select_supported_review(ids,x,x,np.zeros(4),4,information_fraction=.5,information_mode='conditional')
        with self.assertRaises(ValueError):select_supported_review(ids,x,x,np.zeros(4),4,audit_fraction=.8,information_fraction=.5)
        with self.assertRaises(ValueError):conditional_variance_audit(ids,-np.eye(4),[],1,.1)
