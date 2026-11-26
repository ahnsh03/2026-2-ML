import unittest
import numpy as np
from quality_estimation.region_environment import RegionEnvironment,export_environment
from quality_estimation.spatial_features import SpatialProfile

class RegionEnvironmentTest(unittest.TestCase):
    def test_previously_constant_descriptor_change_is_an_audit_flag(self):
        x=np.array([[.1,.2,0.],[.2,.3,0.],[.8,.7,0.],[.9,.8,0.]])
        archive=dict(selected=np.array([0,1]),mean=np.zeros(2),scale=np.ones(2),centers=np.array([[.15,.25],[.85,.75]]))
        spec=export_environment(archive,['hood.intensity','upper_top_left.intensity','hood.glare_fraction'],x)
        result=RegionEnvironment(spec).assign(np.array([[.15,.25,.02]]))
        self.assertTrue(result['environment_constant_feature_changed'][0]);self.assertTrue(result['environment_out_of_reference'][0])
    def spec(self):
        x=np.array([[.1,.2],[.2,.3],[.8,.7],[.9,.8]])
        archive=dict(selected=np.array([0,1]),mean=np.zeros(2),scale=np.ones(2),centers=np.array([[.15,.25],[.85,.75]]))
        return export_environment(archive,['hood.intensity','upper_top_left.intensity'],x),x
    def test_centroids_assign_without_faiss_and_detect_reference_distance(self):
        spec,x=self.spec();model=RegionEnvironment(spec)
        np.testing.assert_array_equal(model.assign(x)['environment_cluster'],[0,0,1,1])
        self.assertTrue(model.assign(np.array([[10.,10.]]))['environment_out_of_reference'][0])
    def test_perception_or_gt_columns_and_geometry_mismatch_rejected(self):
        spec,_=self.spec();spec['ordered_features'][0]='near_left.lane_fraction'
        with self.assertRaises(ValueError):RegionEnvironment(spec)
        spec,_=self.spec();spec['regions_sha256']='0'*64
        with self.assertRaises(ValueError):RegionEnvironment(spec)

if __name__=='__main__':unittest.main()
