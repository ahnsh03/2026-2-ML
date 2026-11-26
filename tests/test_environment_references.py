import unittest
import numpy as np
from quality_estimation.environment_references import fit_relative_references,reference_thresholds

class EnvironmentReferenceTest(unittest.TestCase):
    def test_supported_relative_reference_never_changes_absolute_threshold(self):
        q=np.full(60,.4);c=np.zeros(60,int);roads=np.arange(60)%3;xy=np.column_stack([np.arange(60)*6,np.zeros(60)])
        spec=fit_relative_references(q,c,roads,xy,shrinkage=60)
        self.assertAlmostEqual(reference_thresholds(spec,[0])[0],.55)
        self.assertEqual(spec['absolute_quality_threshold'],.7)
        self.assertEqual(reference_thresholds(spec,[0],ood=[True])[0],.7)
    def test_repeated_frames_at_one_position_do_not_create_support(self):
        spec=fit_relative_references(np.full(100,.2),np.zeros(100,int),np.arange(100)%2,np.zeros((100,2)))
        self.assertFalse(spec['groups']['0']['eligible']);self.assertEqual(reference_thresholds(spec,[0,999]).tolist(),[.7,.7])

if __name__=='__main__':unittest.main()
