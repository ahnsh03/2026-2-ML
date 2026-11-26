import unittest
import numpy as np
from scipy.special import ndtr
from quality_estimation.diagnostic_targets import auxiliary_targets,quality_or_near_defect_probability,band_diagnostic_probabilities

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

if __name__=='__main__':unittest.main()
