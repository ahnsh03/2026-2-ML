import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from common.bridge import sha256_file
from quality_estimation.perception_identity import baseline_identity
from quality_estimation.portable_forest import PortableForest
from quality_estimation.scene_environment import SceneEnvironment,PHOTO_COLUMNS
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_identity import bind_model_perception
from quality_estimation.review_targets import ERROR_TARGET_NAMES
from quality_estimation.support_aware_review import SupportAwareReview


class SupportAwareReviewTests(unittest.TestCase):
    def fixture(self,root,error_kind='GP'):
        rng=np.random.default_rng(7);x=rng.uniform(0,1,(36,2));hist=rng.uniform(0,1,(36,4));raw=rng.uniform(0,1,(36,8))
        env=root/'environment';env.mkdir();profile=SpatialProfile.load();cols=['hood.intensity','near_left.lane_fraction']
        hcols=['rgb_hist_'+str(i)for i in range(4)];identity=dict(baseline_identity(),kind='adaptation',checkpoint_sha256='a'*64,adaptation_run_sha256='b'*64)
        context=dict(profile=profile.version,regions_sha256=profile.regions_sha256)
        for name,count in [('weather',6),('lighting',3),('presence',2)]:
            model=RandomForestClassifier(n_estimators=3,max_depth=3,random_state=4).fit(hist,np.arange(len(x))%count)
            PortableForest.from_sklearn(model,hcols,context).save(env/(name+'.npz'))
        np.savez(env/'transforms.npz',hist_mean=np.zeros(4),hist_scale=np.ones(4),pca_mean=np.zeros(4),
                 pca_components=np.eye(4),pca_variance=np.ones(4),photo_mean=np.zeros(4),photo_scale=np.ones(4),illumination_edges=[.2,.6])
        np.savez(env/'centers.npz',centers=np.zeros((2,11)),active=np.arange(11))
        def write(path,value):path.write_text(json.dumps(value))
        write(env/'environment_manifest.json',dict(version='rgb-scene-environment-1',representation='prob11',ordered_features=hcols+PHOTO_COLUMNS,
            classifier_ordered_features=hcols,**context,cluster_distance_limits={'0':10.,'1':10.},minimum_weather_confidence=0.,minimum_presence_confidence=0.,
            artifact_sha256={p.name:sha256_file(p)for p in env.iterdir()}))
        def gp(path,y,names):
            model=SpatialQualityGP().fit(x,y);model.feature_names_=np.array(cols);model.target_names_=np.array(names)
            model.profile_version_=np.array(profile.version);model.regions_sha256_=np.array(profile.regions_sha256)
            bind_model_perception(model,identity,'model_state_dict');model.save(path)
        gp(root/'quality_model.npz',.3+.6*x[:,0],['whole_frame_lane_iou'])
        y=np.c_[x[:,0]>.5,x[:,1]>.5,x[:,0]<.2].astype(float)
        if error_kind=='GP':gp(root/'error_model.npz',y,list(ERROR_TARGET_NAMES))
        else:
            for j,target in enumerate(ERROR_TARGET_NAMES):
                model=RandomForestRegressor(n_estimators=3,random_state=4).fit(x,y[:,j])
                PortableForest.from_sklearn(model,cols,dict(context,checkpoint_sha256=identity['checkpoint_sha256'],perception_identity=identity,target=target)).save(root/f'error_model_{j}.npz')
        hashes={p.name:sha256_file(p)for p in root.glob('*.npz')}
        write(root/'calibration.json',dict(perception_identity=identity,model_sha256=hashes,signed_quality_residuals=[-.2,0.,.2],interval_residual_bounds=[-.2,.2]))
        write(root/'policy.json',dict(perception_identity=identity,model_sha256=hashes,mode='supported_utility',grouping='global',beta=0.,balanced_floor=0.,
            information_fraction=.1 if error_kind=='GP'else 0.,information_mode='conditional',audit_fraction=.1,utility_weights=[1.,.25,.5],adopted=False))
        write(root/'bundle_manifest.json',dict(version='support-aware-review-bundle-1',perception_identity=identity,perception_weights='model_state_dict',
            checkpoint_sha256=identity['checkpoint_sha256'],**context,quality_kind='GP',error_kind=error_kind,quality_columns=cols,error_columns=cols,
            ordered_features=cols,artifact_sha256={str(p.relative_to(root)):sha256_file(p)for p in root.rglob('*')if p.is_file()}))
        return x,np.c_[hist,raw],identity

    def test_mixed_models_and_gp_are_batch_invariant_without_gt(self):
        for kind in ['GP','RF']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as temporary:
                root=Path(temporary);x,e,identity=self.fixture(root,kind);review=SupportAwareReview(root,perception_identity=identity)
                full=review.predict(x,e);parts=[review.predict(x[i:i+6],e[i:i+6])for i in range(0,len(x),6)]
                for name in full:np.testing.assert_allclose(full[name],np.concatenate([p[name]for p in parts]),atol=1e-9)
                self.assertFalse(full['review_score_is_probability'].any());self.assertEqual(full['error_task_scores'].shape,(36,3))
                ids=np.array([str(i)for i in range(len(x))]);chosen,_,details=review.select_with_features(ids,full,20,x)
                self.assertEqual(len(set(chosen)),20);self.assertEqual(details['random_audit_slots'],2)
                self.assertEqual(details['information_slots'],2 if kind=='GP'else 0)

    def test_changed_model_identity_or_environment_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);_,_,identity=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'pair differs'):SupportAwareReview(root,perception_identity=dict(identity,checkpoint_sha256='c'*64))
            with (root/'environment/centers.npz').open('ab')as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):SupportAwareReview(root)

    def test_conditional_audit_cannot_silently_drop_features(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);x,e,_=self.fixture(root);review=SupportAwareReview(root)
            with self.assertRaisesRegex(ValueError,'requires feature'):review.select(np.array([str(i)for i in range(len(x))]),review.predict(x,e),20)

    def test_environment_rejects_invalid_feature_order_and_nonfinite_rgb(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);_,e,_=self.fixture(root);env=SceneEnvironment(root/'environment')
            with self.assertRaises(ValueError):env.predict(np.full(e.shape,np.nan))
            path=root/'environment/environment_manifest.json';spec=json.loads(path.read_text());spec['ordered_features']=spec['ordered_features'][::-1];path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'feature order'):SceneEnvironment(root/'environment')


if __name__=='__main__':unittest.main()
