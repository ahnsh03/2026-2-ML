"""Version-bound Q regression and supervised-RGB/e-means environment review.

Simulator metadata and GT are not runtime inputs. RGB tags are predictions of
simulator categories; they do not establish real-weather or operating normality.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from scipy.spatial.distance import cdist
from common.bridge import sha256_file
from .spatial_features import SpatialProfile
from .spatial_gp import SpatialQualityGP
from .spatial_identity import validate_model_perception
from .perception_identity import validate_identity
from .portable_forest import PortableForest
from .environment_selection import calibrated_priority,thresholds,select_environment_review
from .cluster_review import mixture_priority

class EnvironmentReview:
 def __init__(self,bundle,*,perception_identity=None):
  root=Path(bundle).resolve();self.root=root;self.spec=json.loads((root/'bundle_manifest.json').read_text());self.model_filename='quality_model.npz'
  if self.spec['version'] not in ['environment-quality-review-bundle-1','environment-quality-review-bundle-2']:raise ValueError('unknown environment review bundle')
  identity=validate_identity(self.spec['perception_identity'])
  if perception_identity is not None and identity!=validate_identity(perception_identity):raise ValueError('perception/quality version pair differs')
  required={self.model_filename,'weather.npz','lighting.npz','centers.npz','calibration.json','policy.json'}
  if not required.issubset(self.spec['artifact_sha256']):raise ValueError('incomplete environment review bundle')
  for name,digest in self.spec['artifact_sha256'].items():
   path=(root/name).resolve();path.relative_to(root)
   if sha256_file(path)!=digest:raise ValueError('environment review artifact changed: '+name)
  self.profile=SpatialProfile.load()
  if self.spec['profile']!=self.profile.version or self.spec['regions_sha256']!=self.profile.regions_sha256:raise ValueError('review camera profile differs')
  self.columns=self.spec['ordered_features'];self.target_names=['whole_frame_lane_iou'];self.kind=self.spec['quality_model_kind']
  if self.kind=='GP':
   self.model=SpatialQualityGP.load(root/self.model_filename);validate_model_perception(self.model,self.spec)
   if self.model.feature_names_.tolist()!=self.columns or self.model.target_names_.tolist()!=self.target_names:raise ValueError('GP feature/target order differs')
   if str(self.model.checkpoint_sha256_)!=identity['checkpoint_sha256'] or str(self.model.profile_version_)!=self.profile.version or str(self.model.regions_sha256_)!=self.profile.regions_sha256:raise ValueError('embedded GP geometry/perception differs')
  elif self.kind=='RF':
   self.model=PortableForest.load(root/self.model_filename);self.model.validate_context(self.columns,self.profile.version,self.profile.regions_sha256,identity['checkpoint_sha256'])
   if self.model.metadata['context']['perception_identity']!=identity:raise ValueError('RF perception identity differs')
  else:raise ValueError('unsupported quality model')
  self.weather=PortableForest.load(root/'weather.npz');self.lighting=PortableForest.load(root/'lighting.npz');ecols=self.spec['environment_ordered_features'];self.environment=SimpleNamespace(columns=ecols)
  for model in [self.weather,self.lighting]:
   model.validate_context(ecols,self.profile.version,self.profile.regions_sha256)
   if model.kind!='classification':raise ValueError('environment classifier required')
  if self.weather.classes_.tolist()!=list(range(6)) or self.lighting.classes_.tolist()!=list(range(3)):raise ValueError('environment class order differs')
  with np.load(root/'centers.npz',allow_pickle=False) as z:self.centers=z['centers'];self.active=z['active']
  if self.centers.ndim!=2 or self.centers.shape[1]!=len(self.active) or not np.isfinite(self.centers).all() or self.active.dtype.kind not in 'iu' or not len(self.active) or len(set(self.active.tolist()))!=len(self.active) or ((self.active<0)|(self.active>=9)).any():raise ValueError('invalid environment centers')
  self.calibration=json.loads((root/'calibration.json').read_text());self.policy=json.loads((root/'policy.json').read_text())
  for component in [self.calibration,self.policy]:
   if component['perception_identity']!=identity or component['model_sha256']!=sha256_file(root/self.model_filename):raise ValueError('calibration/policy model differs')
  if self.policy['absolute_quality_threshold']!=.7:raise ValueError('absolute quality benchmark differs')
  self.residuals=np.asarray(self.calibration['signed_residuals'],float)
  if self.residuals.ndim!=1 or not len(self.residuals) or not np.isfinite(self.residuals).all() or (np.diff(self.residuals)<0).any():raise ValueError('invalid calibration residual distribution')
  if self.spec['version']=='environment-quality-review-bundle-2' and ('interval_residual_bounds' not in self.calibration or self.calibration.get('interval_method') not in ['pooled','road_envelope']):raise ValueError('explicit interval calibration required for v2')
  self.interval_bounds=np.asarray(self.calibration.get('interval_residual_bounds',np.quantile(self.residuals,[.05,.95])),float)
  if self.interval_bounds.shape!=(2,) or not np.isfinite(self.interval_bounds).all() or self.interval_bounds[0]>self.interval_bounds[1]:raise ValueError('invalid interval residual bounds')
  if self.policy['grouping'] not in ['global','rgb_cluster','rgb_tag','rgb_weather','rgb_lighting']:raise ValueError('unknown runtime environment grouping')
  if any(not 0<=self.policy[k]<=1 for k in ['balanced_floor','audit_fraction','uncertainty_weight']):raise ValueError('invalid review mixture')

 def predict(self,quality_vectors,environment_vectors):
  if len(quality_vectors)!=len(environment_vectors):raise ValueError('input row counts differ')
  if self.kind=='GP':
   raw,cov=self.model.predict(quality_vectors);mean=np.clip(raw[:,0],0,1);sd=np.sqrt(np.maximum(cov[:,0,0],1e-6));noise=float(self.model.noise_variance)*float(self.model.y_scale_[0])**2;epistemic=np.sqrt(np.maximum(cov[:,0,0]-noise,0))
  else:mean=np.clip(self.model.predict(quality_vectors),0,1);sd=np.ones(len(mean));epistemic=np.zeros(len(mean))
  wp=self.weather.predict_proba(environment_vectors);lp=self.lighting.predict_proba(environment_vectors);embedding=np.sqrt(np.maximum(np.c_[wp,lp],0))[:,self.active];distance=cdist(embedding,self.centers,'sqeuclidean');groups=distance.argmin(1);nearest=np.sqrt(distance.min(1));w=wp.argmax(1);l=lp.argmax(1)
  limits=np.array([self.policy['cluster_distance_limits'].get(str(int(g)),0.) for g in groups]);ood=(nearest>limits)|(wp.max(1)<self.policy.get('minimum_tag_confidence',.6))|(lp.max(1)<self.policy.get('minimum_tag_confidence',.6))
  tags=w*3+l;review_groups=groups if self.policy['grouping']=='rgb_cluster' else tags if self.policy['grouping']=='rgb_tag' else w if self.policy['grouping']=='rgb_weather' else l if self.policy['grouping']=='rgb_lighting' else np.zeros(len(mean),int)
  tau=thresholds(self.policy['reference'],review_groups,unsupported=ood) if self.policy['grouping']!='global' else np.full(len(mean),.7)
  low=calibrated_priority(mean,self.residuals,scale=sd);priority=calibrated_priority(mean,self.residuals,threshold=tau,scale=sd)
  low_res,high_res=self.interval_bounds;lo=np.clip(mean+low_res*sd,0,1);hi=np.clip(mean+high_res*sd,0,1)
  return dict(estimated_iou=mean,quality_predictive_scale=sd,quality_epistemic_sd=epistemic,gp_uncertainty_available=np.full(len(mean),self.kind=='GP',bool),interval90_lower=lo,interval90_upper=hi,low_quality_probability=low,review_probability=priority,environment_cluster=groups,environment_out_of_distribution=ood,review_group=review_groups,environment_ood=ood,environment_distance=nearest,predicted_weather=w,predicted_lighting=l,weather_confidence=wp.max(1),lighting_confidence=lp.max(1),review_threshold=tau)

 def select(self,ids,predictions,budget,*,seed=271127):
  score=mixture_priority(predictions['review_probability'],predictions['quality_predictive_scale'],self.policy['uncertainty_weight'])
  chosen=select_environment_review(np.asarray(ids),score,predictions['review_group'],int(budget),balanced_floor=self.policy['balanced_floor'],audit_fraction=self.policy['audit_fraction'],seed=seed)
  return chosen,score
