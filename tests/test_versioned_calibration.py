import unittest
import numpy as np
from quality_estimation.versioned_calibration import fit_road_calibration,apply_road_calibration,gaussian_nll

class VersionedCalibrationTest(unittest.TestCase):
    def test_four_targets_retain_psd_and_never_shrink(self):
        rng=np.random.default_rng(42);mean=np.full((90,4),.6);truth=np.clip(mean+rng.normal(0,.12,mean.shape),0,1);cov=np.tile(np.eye(4)*.001,(90,1,1));roads=np.repeat(['a','b','c'],30)
        spec=fit_road_calibration(mean,cov,truth,roads,correlation_blend=.5)
        self.assertTrue(np.all(np.array(spec['scale'])>=1))
        m,c=apply_road_calibration(mean,cov,spec);self.assertGreater(np.linalg.eigvalsh(c).min(),0);self.assertTrue(np.isfinite(gaussian_nll(m,c,truth)).all())
    def test_partial_undefined_targets_and_negative_covariance_rejected(self):
        m=np.full((60,3),.8);y=m.copy();y[:20,2]=np.nan;c=np.tile(np.eye(3)*.01,(60,1,1));roads=np.repeat(['a','b','c'],20)
        spec=fit_road_calibration(m,c,y,roads);self.assertEqual(spec['counts'],[60,60,40])
        c[:,0,0]=-1
        with self.assertRaises(ValueError):fit_road_calibration(m,c,y,roads)

if __name__=='__main__':unittest.main()
