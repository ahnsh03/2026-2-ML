import unittest
import numpy as np
from scipy.special import ndtr
from quality_estimation.drive_calibration import (
    spatial_probabilities, spatial_labels, fit_probability_map,
    apply_probability_map, fit_monotone_binary, apply_monotone_binary,
    fit_moments, apply_moments,
)
from quality_estimation.drive_review import calibrated_review, review_priority, assign_environment

class DriveCalibrationTest(unittest.TestCase):
    def test_joint_classes_match_independent_gaussian_and_partition(self):
        m=np.array([[.6,.85,.45],[.9,.6,.8]])
        c=np.tile(np.diag([.01,.04,.09]),(2,1,1))
        p=spatial_probabilities(m,c)
        n=ndtr((.8-m[:,1])/.2);f=ndtr((.5-m[:,2])/.3)
        expected=np.column_stack([(1-n)*(1-f),n*(1-f),(1-n)*f,n*f])
        np.testing.assert_allclose(p,expected,atol=1e-10)
        np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_array_equal(spatial_labels([[.7,.8,.5],[.7,.7,.6],[.7,.9,.4],[.7,.7,.4],[.7,np.nan,.4]]),[0,1,2,3,-1])

    def test_probability_map_is_portable_normalized_and_improves_overconfidence(self):
        p=np.tile([.97,.01,.01,.01],(200,1));y=np.tile([0,0,1,2,3],40)
        roads=np.repeat(['a','b','c','d'],50)
        fit=fit_probability_map(p,y,roads,kind='vector')
        q=apply_probability_map(p,fit)
        self.assertTrue(np.isfinite(q).all());self.assertGreaterEqual(q.min(),0)
        np.testing.assert_allclose(q.sum(1),1)
        self.assertLess(-np.log(q[np.arange(200),y]).mean(),-np.log(p[np.arange(200),y]).mean())

    def test_monotone_binary_preserves_order_and_handles_missing_class(self):
        p=np.linspace(.01,.99,100);y=(p>.7).astype(int);roads=np.tile(['a','b'],50)
        fit=fit_monotone_binary(p,y,roads);q=apply_monotone_binary(p,fit)
        self.assertEqual(fit['kind'],'monotone_platt')
        self.assertTrue((np.diff(q)>=0).all())
        fallback=fit_monotone_binary(p,np.zeros(100,int),roads)
        np.testing.assert_array_equal(apply_monotone_binary(p,fallback),p)

    def test_absent_spatial_class_uses_temperature_instead_of_class_bias(self):
        p=np.tile([.4,.3,.2,.1],(80,1));labels=np.tile([0,1],40);roads=np.repeat(['a','b'],40)
        fit=fit_probability_map(p,labels,roads,kind='vector')
        self.assertEqual(fit['requested_kind'],'vector')
        self.assertEqual(fit['kind'],'temperature')
        np.testing.assert_allclose(apply_probability_map(p,fit).sum(1),1)

    def test_environment_fallback_and_correlation_are_positive_definite(self):
        rng=np.random.default_rng(7);m=np.full((120,3),.5);c=np.tile(np.eye(3)*.01,(120,1,1))
        y=m+rng.normal(0,.08,(120,3));env=np.r_[np.zeros(100,int),np.ones(20,int)]
        roads=np.r_[np.tile(['a','b'],50),np.repeat('c',20)]
        fit=fit_moments(m,c,y,env,roads,conditional=True,correlation_weight=.5)
        self.assertIn('0',fit['groups']);self.assertNotIn('1',fit['groups'])
        cm,cc,fallback=apply_moments(m,c,env,fit)
        self.assertFalse(fallback[:100].any());self.assertTrue(fallback[100:].all())
        self.assertTrue((np.linalg.eigvalsh(cc)>0).all());self.assertTrue(np.isfinite(cm).all())

    def test_ood_environment_cannot_get_conditional_fit(self):
        rng=np.random.default_rng(17);m=np.full((80,3),.5)
        c=np.tile(np.eye(3)*.01,(80,1,1));y=m+rng.normal(0,.1,m.shape)
        roads=np.tile(['a','b'],40);env=np.full(80,-1)
        fit=fit_moments(m,c,y,env,roads,conditional=True)
        self.assertEqual(fit['groups'],{})
        self.assertTrue(apply_moments(m,c,env,fit)[2].all())

    def test_gt_free_review_preserves_global_fallback_and_joint_partition(self):
        m=np.array([[.6,.85,.45],[.9,.6,.8]]);c=np.tile(np.eye(3)*.01,(2,1,1))
        moment=dict(global_fit=dict(bias=[0.]*3,scale=[1.]*3),groups={},correlation_weight=0.)
        cal=dict(event_definition=dict(global_iou=.7,near_recall=.8,far_recall=.5),moments=moment,
                 joint_map=dict(temperature=1.,bias=[0.]*4),global_map=dict(kind='identity'),
                 environment_thresholds={'0':dict(threshold=.55)})
        result=calibrated_review(m,c,np.array([0,-1]),cal,[.5,1.])
        np.testing.assert_allclose(result['spatial'].sum(1),1)
        self.assertEqual(result['environment_threshold'][1],.7)
        self.assertEqual(result['environment_relative_probability'][1],result['global_probability'][1])
        self.assertLess(result['environment_relative_probability'][0],result['global_probability'][0])
        np.testing.assert_array_equal(review_priority(result['global_probability'],result['spatial'],result['environment_relative_probability'],None),result['global_probability'])

    def test_environment_uses_rgb_only_and_unknown_distance_falls_back(self):
        profile=dict(ordered_features=['a','b'],selected=[0,1],mean=[0.,0.],scale=[1.,1.],centers=[[0.,0.],[2.,2.]],policy_map=[0,1],maximum_reference_distance=1.)
        fine,distance,coarse=assign_environment([[0.,0.],[20.,20.]],['a','b'],profile)
        np.testing.assert_array_equal(coarse,[0,-1])
        with self.assertRaises(ValueError):assign_environment([[0.,0.]],['b','a'],profile)

if __name__=='__main__':unittest.main()
