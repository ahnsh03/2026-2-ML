"""Version-bound GP/environment review on vectors, with no GT or scene-metadata input."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
from common.bridge import sha256_file
from .spatial_features import SpatialProfile
from .spatial_gp import SpatialQualityGP
from .spatial_identity import validate_model_perception
from .perception_identity import validate_identity
from .region_environment import RegionEnvironment
from .versioned_calibration import apply_road_calibration
from .diagnostic_targets import (FOUR_TARGET_NAMES,band_diagnostic_probabilities,near_defect_probability,
                                quality_or_near_defect_probability)
from .expected_deficit import bounded_expected_deficit
from .cluster_review import select_review,mixture_priority


class VersionedReview:
    def __init__(self,bundle,*,perception_identity=None):
        root=Path(bundle).resolve();self.root=root;self.spec=json.loads((root/'bundle_manifest.json').read_text())
        if self.spec['version']!='roi9-versioned-review-bundle-1':raise ValueError('unknown versioned review bundle')
        identity=validate_identity(self.spec['perception_identity'])
        if not {'quality_gp.npz','environment.json','calibration.json','policy.json'}.issubset(self.spec['artifact_sha256']):raise ValueError('incomplete review bundle inventory')
        if perception_identity is not None and validate_identity(perception_identity)!=identity:raise ValueError('perception/quality version pair differs')
        for name,digest in self.spec['artifact_sha256'].items():
            path=(root/name).resolve();path.relative_to(root)
            if sha256_file(path)!=digest:raise ValueError('versioned review artifact changed: '+name)
        self.profile=SpatialProfile.load()
        if self.spec['profile']!=self.profile.version or self.spec['regions_sha256']!=self.profile.regions_sha256:raise ValueError('review camera profile differs')
        self.model=SpatialQualityGP.load(root/'quality_gp.npz');validate_model_perception(self.model,self.spec)
        self.columns=self.model.feature_names_.tolist();self.target_names=self.model.target_names_.tolist()
        if self.target_names!=list(FOUR_TARGET_NAMES[:len(self.target_names)]) or len(self.target_names) not in (3,4):raise ValueError('unknown diagnostic target semantics')
        if str(self.model.checkpoint_sha256_)!=identity['checkpoint_sha256']:raise ValueError('embedded GP checkpoint differs')
        if self.columns!=self.spec['ordered_features'] or self.target_names!=self.spec['target_names']:raise ValueError('review feature/target order differs')
        if str(self.model.profile_version_)!=self.profile.version or str(self.model.regions_sha256_)!=self.profile.regions_sha256:raise ValueError('embedded GP geometry differs')
        self.environment=RegionEnvironment.load(root/'environment.json',self.profile)
        self.calibration=json.loads((root/'calibration.json').read_text());self.policy=json.loads((root/'policy.json').read_text())
        for component in [self.calibration,self.policy]:
            if component['model_sha256']!=sha256_file(root/'quality_gp.npz') or component['perception_identity']!=identity:raise ValueError('review calibration/policy model differs')
        if self.policy['environment_sha256']!=sha256_file(root/'environment.json') or self.policy['calibration_sha256']!=sha256_file(root/'calibration.json'):raise ValueError('review policy environment/calibration differs')
        if any(not 0<=self.policy[k]<=1 for k in ['uncertainty_weight','quota_exponent','audit_fraction']):raise ValueError('invalid review mixture parameters')
        if self.policy.get('audit_rounding','floor') not in ('floor','ceil'):raise ValueError('invalid review audit rounding')
        if self.policy['absolute_quality_threshold']!=.7:raise ValueError('research benchmark quality threshold differs')
        if self.policy['score_family']=='union_near' and len(self.target_names)!=4:raise ValueError('near-F1 union requires four-task GP')

    def predict(self,quality_vectors,environment_vectors):
        if len(quality_vectors)!=len(environment_vectors):raise ValueError('quality/environment input rows differ')
        raw,cov=self.model.predict(quality_vectors)
        if self.policy.get('calibration_enabled',True):mean,cal_cov=apply_road_calibration(raw,cov,self.calibration)
        else:mean,cal_cov=raw,cov
        sd=np.sqrt(np.maximum(cal_cov[:,0,0],1e-12));low=ndtr((.7-mean[:,0])/sd)
        noise=float(self.model.noise_variance)*float(self.model.y_scale_[0])**2
        epistemic=np.sqrt(np.maximum(cov[:,0,0]-noise,0))
        output=dict(raw_mean=raw,raw_covariance=cov,risk_mean=mean,risk_covariance=cal_cov,estimated_iou=np.clip(raw[:,0],0,1),
                    low_quality_probability=low,quality_predictive_sd=sd,quality_epistemic_sd=epistemic,
                    expected_quality_deficit=bounded_expected_deficit(mean[:,0],cal_cov[:,0,0],.7),
                    band_diagnostic_probabilities=band_diagnostic_probabilities(mean,cal_cov))
        output['near_minus_far_recall_mean']=mean[:,1]-mean[:,2]
        output['near_minus_far_recall_sd']=np.sqrt(np.maximum(cal_cov[:,1,1]+cal_cov[:,2,2]-2*cal_cov[:,1,2],0))
        if len(self.target_names)==4:
            output['near_f1_defect_probability']=near_defect_probability(mean,cal_cov)
            output['quality_or_near_f1_defect_probability']=quality_or_near_defect_probability(mean,cal_cov)
        output.update(self.environment.assign(environment_vectors));return output

    def select(self,ids,predictions,budget,*,seed=261126):
        """Rank uncertainty over the complete review buffer, never each inference batch."""
        family=self.policy['score_family']
        if family=='probability':score=predictions['low_quality_probability']
        elif family=='expected_deficit':score=predictions['expected_quality_deficit']/.7
        elif family=='mean':score=1-predictions['estimated_iou']
        elif family=='union_near':score=predictions['quality_or_near_f1_defect_probability']
        else:raise ValueError('unknown review priority family')
        priority=mixture_priority(score,predictions['quality_epistemic_sd'],self.policy['uncertainty_weight'])
        groups=predictions['environment_cluster'] if self.policy.get('environment_quotas_enabled',True) else np.zeros(len(ids),int)
        indices=select_review(np.asarray(ids),priority,groups,int(budget),quota_exponent=self.policy['quota_exponent'],audit_fraction=self.policy['audit_fraction'],seed=seed,audit_rounding=self.policy.get('audit_rounding','floor'))
        return indices,priority
