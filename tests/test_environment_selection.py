import unittest
import numpy as np
from quality_estimation.environment_selection import fit_environment_reference,thresholds,residual_distribution,calibrated_priority,select_environment_review

class EnvironmentSelectionTest(unittest.TestCase):
 def test_supported_reference_can_move_both_directions_and_unsupported_falls_back(self):
  q=np.r_[np.linspace(.2,.6,60),np.linspace(.7,.95,60),[.1]*2];g=np.r_[np.zeros(60,int),np.ones(60,int),[2,2]];roads=np.tile(np.arange(3),41)[:122];xy=np.column_stack((np.arange(122)*10,np.zeros(122)))
  ref=fit_environment_reference(q,g,roads,xy,shrinkage=20)
  values=thresholds(ref,[0,1,2,99]);self.assertLess(values[0],.7);self.assertGreater(values[1],.7);self.assertEqual(values[2],.7);self.assertEqual(values[3],.7)
  self.assertEqual(thresholds(ref,[1],unsupported=[True])[0],.7)
 def test_empirical_priority_order_and_scale(self):
  r=residual_distribution([.3,.5,.7],[.5,.5,.5]);p=calibrated_priority([.3,.5,.8],r)
  self.assertTrue(np.all(np.diff(p)<=0));self.assertTrue(np.all((p>0)&(p<1)))
  np.testing.assert_allclose(calibrated_priority([.5],r,scale=[1]),calibrated_priority([.5],r))
  with self.assertRaises(ValueError):calibrated_priority([.5],r[::-1])
 def test_partial_floor_keeps_budget_and_allows_cross_group_threshold_effect(self):
  ids=np.array([f'i{i}' for i in range(12)]);groups=np.repeat([0,1],6);score=np.arange(12,dtype=float)
  first=select_environment_review(ids,score,groups,6,balanced_floor=.25,audit_fraction=0.)
  second=select_environment_review(ids,score+np.where(groups==0,20,0),groups,6,balanced_floor=.25,audit_fraction=0.)
  self.assertEqual(len(set(first)),6);self.assertGreater((groups[second]==0).sum(),(groups[first]==0).sum())
  for floor in [0.,.5,1.]:
   x=select_environment_review(ids,score,groups,12,balanced_floor=floor,audit_fraction=.1);self.assertEqual(set(x),set(range(12)))
  with self.assertRaises(ValueError):select_environment_review(np.array(['a','a']),[0.,1.],[0,1],1)
if __name__=='__main__':unittest.main()
