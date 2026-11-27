import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from common.bridge import sha256_file
from quality_estimation.perception_identity import baseline_identity
from quality_estimation.spatial_identity import bind_model_perception
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest
from quality_estimation.environment_review import EnvironmentReview

class EnvironmentReviewTest(unittest.TestCase):
 def bundle(self,root,kind):
  profile=SpatialProfile.load();identity=baseline_identity();x=np.random.default_rng(42).uniform(0,1,(24,2));cols=['hood.intensity','near_left.lane_fraction'];envcols=['hood.intensity','upper_top_left.intensity'];y=.3+.6*x[:,0];context=dict(profile=profile.version,regions_sha256=profile.regions_sha256)
  if kind=='GP':
   m=SpatialQualityGP().fit(x,y);m.feature_names_=np.array(cols);m.target_names_=np.array(['whole_frame_lane_iou']);m.profile_version_=np.array(profile.version);m.regions_sha256_=np.array(profile.regions_sha256);bind_model_perception(m,identity,'ema_state_dict');m.save(root/'quality_model.npz')
  else:
   m=RandomForestRegressor(n_estimators=5,random_state=42).fit(x,y);PortableForest.from_sklearn(m,cols,dict(context,checkpoint_sha256=identity['checkpoint_sha256'],perception_identity=identity,perception_weights='ema_state_dict')).save(root/'quality_model.npz')
  for name,nclasses in [('weather',6),('lighting',3)]:
   m=RandomForestClassifier(n_estimators=5,random_state=42).fit(x,np.arange(24)%nclasses);PortableForest.from_sklearn(m,envcols,context).save(root/(name+'.npz'))
  np.savez(root/'centers.npz',centers=np.zeros((2,9)),active=np.arange(9))
  def write(name,d):(root/name).write_text(json.dumps(d))
  digest=sha256_file(root/'quality_model.npz');write('calibration.json',dict(perception_identity=identity,model_sha256=digest,signed_residuals=[-.2,-.1,0.,.1,.2]));write('policy.json',dict(perception_identity=identity,model_sha256=digest,absolute_quality_threshold=.7,grouping='global',reference={'absolute_threshold':.7,'groups':{}},cluster_distance_limits={'0':5.,'1':5.},minimum_tag_confidence=0.,balanced_floor=0.,audit_fraction=.1,uncertainty_weight=0.))
  write('bundle_manifest.json',dict(version='environment-quality-review-bundle-1',quality_model_kind=kind,perception_identity=identity,perception_weights='ema_state_dict',checkpoint_sha256=identity['checkpoint_sha256'],profile=profile.version,regions_sha256=profile.regions_sha256,ordered_features=cols,environment_ordered_features=envcols,artifact_sha256={p.name:sha256_file(p) for p in root.iterdir()}))
  return x
 def test_both_model_kinds_are_batch_invariant_and_never_require_gt(self):
  for kind in ['GP','RF']:
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);x=self.bundle(root,kind);review=EnvironmentReview(root);full=review.predict(x,x);parts=[review.predict(x[i:i+4],x[i:i+4]) for i in range(0,len(x),4)];joined={k:np.concatenate([p[k] for p in parts]) for k in full}
    for key in full:np.testing.assert_allclose(full[key],joined[key],atol=1e-10)
    ids=np.array([str(i) for i in range(len(x))]);a,pa=review.select(ids,full,7);b,pb=review.select(ids,joined,7);np.testing.assert_array_equal(a,b);np.testing.assert_allclose(pa,pb)
 def test_unsupported_rgb_groups_use_common_threshold(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);x=self.bundle(root,'GP');review=EnvironmentReview(root);review.policy['grouping']='rgb_lighting'
   review.policy['reference']={'absolute_threshold':.7,'groups':{str(i):{'threshold':.5+i*.05} for i in range(3)}}
   review.policy['cluster_distance_limits']={'0':0.,'1':0.};pred=review.predict(x,x)
   self.assertTrue(pred['environment_ood'].all());np.testing.assert_array_equal(pred['review_threshold'],np.full(len(x),.7))
 def test_exploration_matches_frozen_search_on_complete_buffer(self):
  from quality_estimation.environment_selection import calibrated_priority
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);x=self.bundle(root,'GP');review=EnvironmentReview(root);review.policy['uncertainty_weight']=.25;pred=review.predict(x,x);ids=np.array([str(i) for i in range(len(x))]);_,priority=review.select(ids,pred,7)
   expected=calibrated_priority(pred['estimated_iou'],review.residuals,threshold=pred['review_threshold'],scale=pred['quality_predictive_scale'],uncertainty_weight=.25)
   np.testing.assert_allclose(priority,expected,atol=1e-12)
 def test_coarse_rgb_groups_match_predicted_tags_without_metadata(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);x=self.bundle(root,'GP');review=EnvironmentReview(root)
   for grouping,field in [('rgb_weather','predicted_weather'),('rgb_lighting','predicted_lighting')]:
    review.policy['grouping']=grouping;pred=review.predict(x,x);np.testing.assert_array_equal(pred['review_group'],pred[field])
 def test_identity_and_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.bundle(root,'GP');wrong=dict(baseline_identity(),checkpoint_sha256='0'*64)
   with self.assertRaises(ValueError):EnvironmentReview(root,perception_identity=wrong)
   (root/'policy.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'artifact changed'):EnvironmentReview(root)
if __name__=='__main__':unittest.main()
