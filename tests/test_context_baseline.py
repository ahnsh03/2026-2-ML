import unittest
import numpy as np
from quality_estimation.context_baseline import fit_context_mean,predict_context_mean


class ContextBaselineTest(unittest.TestCase):
 def test_training_only_shrinkage_and_unsupported_context_fallback(self):
  y=np.r_[np.full(30,.2),np.full(30,.8),[np.nan]];groups=np.r_[np.zeros(30,int),np.ones(30,int),9];roads=np.r_[np.tile([0,1],15),np.zeros(30,int),0]
  model=fit_context_mean(y,groups,roads,prior_rows=30)
  self.assertEqual(model['defined_training_rows'],60)
  np.testing.assert_allclose(predict_context_mean(model,[0,1,42]),[.35,.5,.5])
  before=predict_context_mean(model,[0,1]);predict_context_mean(model,[42,1,0]);np.testing.assert_array_equal(before,predict_context_mean(model,[0,1]))
 def test_undefined_or_invalid_targets_are_never_filled_with_perfect_quality(self):
  with self.assertRaises(ValueError):fit_context_mean([np.nan],[0],[0])
  with self.assertRaises(ValueError):fit_context_mean([1.1],[0],[0])
  with self.assertRaises(ValueError):fit_context_mean([.5],[0,1],[0])
