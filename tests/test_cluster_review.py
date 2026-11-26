import unittest
import numpy as np
from quality_estimation.cluster_review import review_scores,select_review,mixture_priority

class ClusterReviewTest(unittest.TestCase):
    def test_equal_uncertainties_do_not_invent_an_exploration_order(self):
        values=mixture_priority(np.full(6,.6),np.ones(6),.25)
        np.testing.assert_allclose(values,np.full(6,.575))
    def test_quota_redistributes_exhausted_rare_cluster_and_never_duplicates(self):
        ids=np.array(['a','b','c','d','e','f']);groups=np.array([0,1,1,1,1,1]);score=np.arange(6,dtype=float)
        got=select_review(ids,score,groups,5,quota_exponent=0.)
        self.assertEqual(len(set(got)),5);self.assertIn(0,got)
        self.assertEqual(got.tolist(),[0,5,4,3,2])
    def test_audit_has_fixed_reproducible_budget_and_zero_budget_works(self):
        args=(np.array([str(i) for i in range(20)]),np.arange(20.),np.arange(20)%3,10)
        a=select_review(*args,audit_fraction=.4,seed=8);b=select_review(*args,audit_fraction=.4,seed=8)
        np.testing.assert_array_equal(a,b);self.assertEqual(len(set(a)),10)
        self.assertEqual(len(select_review(args[0],args[1],args[2],0)),0)
    def test_posterior_probability_and_no_gt_signature(self):
        score=review_scores(np.array([.5,.7,.9]),np.array([.01,.01,.01]))
        self.assertGreater(score[0],score[1]);self.assertAlmostEqual(score[1],.5);self.assertGreater(score[1],score[2])
        with self.assertRaises(ValueError):review_scores([.5],[-1.])

if __name__=='__main__':unittest.main()
