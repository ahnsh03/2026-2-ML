import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common.bridge import sha256_file
from quality_estimation.perception_identity import baseline_identity
from quality_estimation.spatial_identity import bind_model_perception
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_environment import export_environment
from quality_estimation.diagnostic_targets import FOUR_TARGET_NAMES
from quality_estimation.versioned_review import VersionedReview

class VersionedReviewTest(unittest.TestCase):
    def bundle(self,root):
        profile=SpatialProfile.load();identity=baseline_identity();rng=np.random.default_rng(44);x=rng.uniform(0,1,(16,2));y=np.column_stack([.5+.3*x[:,0],.9-.4*x[:,1],.3+.4*x[:,0]])
        model=SpatialQualityGP().fit(x,y);model.feature_names_=np.array(['hood.intensity','near_left.lane_fraction']);model.target_names_=np.array(FOUR_TARGET_NAMES[:3]);model.profile_version_=np.array(profile.version);model.regions_sha256_=np.array(profile.regions_sha256);bind_model_perception(model,identity,'ema_state_dict');model.save(root/'quality_gp.npz')
        digest=sha256_file(root/'quality_gp.npz')
        def write(name,obj):(root/name).write_text(json.dumps(obj))
        archive=dict(selected=np.array([0,1]),mean=np.zeros(2),scale=np.ones(2),centers=np.array([[.2,.3],[.8,.7]]))
        write('environment.json',export_environment(archive,['hood.intensity','upper_top_left.intensity'],x,profile))
        write('calibration.json',dict(bias=[0.,0.,0.],scale=[1.,1.,1.],correlation_blend=0.,model_sha256=digest,perception_identity=identity))
        write('policy.json',dict(score_family='probability',uncertainty_weight=.25,quota_exponent=0.,audit_fraction=.1,absolute_quality_threshold=.7,
              model_sha256=digest,perception_identity=identity,environment_sha256=sha256_file(root/'environment.json'),calibration_sha256=sha256_file(root/'calibration.json')))
        write('bundle_manifest.json',dict(version='roi9-versioned-review-bundle-1',profile=profile.version,regions_sha256=profile.regions_sha256,
              perception_identity=identity,perception_weights='ema_state_dict',checkpoint_sha256=identity['checkpoint_sha256'],ordered_features=model.feature_names_.tolist(),target_names=model.target_names_.tolist(),
              artifact_sha256={name:sha256_file(root/name) for name in ['quality_gp.npz','environment.json','calibration.json','policy.json']}))
        return x
    def test_replay_batch_size_does_not_change_archive_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=self.bundle(root);review=VersionedReview(root,perception_identity=baseline_identity());full=review.predict(x,x)
            pieces=[review.predict(x[i:i+4],x[i:i+4]) for i in range(0,len(x),4)];joined={k:np.concatenate([p[k] for p in pieces]) for k in full}
            for key in full:np.testing.assert_allclose(full[key],joined[key],atol=1e-10)
            ids=np.array([str(i) for i in range(len(x))]);a,pa=review.select(ids,full,5);b,pb=review.select(ids,joined,5)
            np.testing.assert_array_equal(a,b);np.testing.assert_allclose(pa,pb,atol=1e-10)
    def test_wrong_perception_identity_or_tampered_environment_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.bundle(root);wrong=dict(baseline_identity(),kind='adaptation',checkpoint_sha256='1'*64,adaptation_run_sha256='2'*64)
            with self.assertRaisesRegex(ValueError,'version pair'):VersionedReview(root,perception_identity=wrong)
            (root/'environment.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):VersionedReview(root)
    def test_common_policy_can_ignore_environment_quotas_without_changing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=self.bundle(root);review=VersionedReview(root);pred=review.predict(x,x)
            review.policy.update(environment_quotas_enabled=False,audit_fraction=0.,uncertainty_weight=0.)
            ids=np.array([f'{i:03d}' for i in range(len(x))]);a,_=review.select(ids,pred,5)
            pred['environment_cluster']=np.arange(len(x));b,_=review.select(ids,pred,5)
            np.testing.assert_array_equal(a,b)
            expected=np.lexsort((ids,-pred['low_quality_probability']))[:5]
            np.testing.assert_array_equal(a,expected)
    def test_near_far_contrast_uses_posterior_cross_covariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=self.bundle(root);pred=VersionedReview(root).predict(x,x);cov=pred['risk_covariance']
            np.testing.assert_allclose(pred['near_minus_far_recall_sd']**2,cov[:,1,1]+cov[:,2,2]-2*cov[:,1,2])

if __name__=='__main__':unittest.main()
