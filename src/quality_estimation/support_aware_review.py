"""Version-bound whole-frame quality and support-aware error-value review.

Error scores are Gaussian/RF regressions of offline event indicators, not
calibrated Bernoulli probabilities or operational safety/normality decisions.
"""
from pathlib import Path
import json
import numpy as np
from common.bridge import sha256_file
from .spatial_features import SpatialProfile
from .spatial_gp import SpatialQualityGP
from .spatial_identity import validate_model_perception
from .perception_identity import validate_identity
from .portable_forest import PortableForest
from .scene_environment import SceneEnvironment
from .review_targets import ERROR_TARGET_NAMES,gaussian_utility
from .environment_selection import calibrated_priority,thresholds
from .support_review_selection import select_supported_review


class SupportAwareReview:
    model_filename='quality_model.npz'
    near_metric_name='supported_error_event_score_not_iou_or_recall'

    def __init__(self,bundle,*,perception_identity=None):
        self.root=Path(bundle).resolve();self.spec=json.loads((self.root/'bundle_manifest.json').read_text())
        if self.spec['version']!='support-aware-review-bundle-1':raise ValueError('unknown support-aware bundle')
        self.identity=validate_identity(self.spec['perception_identity'])
        if perception_identity is not None and validate_identity(perception_identity)!=self.identity:raise ValueError('perception/review pair differs')
        self.verify_unchanged();self.profile=SpatialProfile.load()
        if self.spec['profile']!=self.profile.version or self.spec['regions_sha256']!=self.profile.regions_sha256:
            raise ValueError('support-aware camera geometry differs')
        self.quality_columns=self.spec['quality_columns'];self.error_columns=self.spec['error_columns']
        self.columns=list(dict.fromkeys(self.quality_columns+self.error_columns));self.target_names=['whole_frame_lane_iou']
        if self.columns!=self.spec['ordered_features']:raise ValueError('support-aware feature union order differs')
        self.kind=self.spec['quality_kind'];self.error_kind=self.spec['error_kind']
        if self.kind not in ['GP','RF']or self.error_kind not in ['GP','RF']:raise ValueError('unknown support-aware model family')
        self.model=self.load_model(self.root/self.model_filename,self.kind,self.quality_columns,self.target_names)
        self.error_files=['error_model.npz']if self.error_kind=='GP'else['error_model_'+str(i)+'.npz'for i in range(3)]
        self.errors=self.load_model(self.root/'error_model.npz','GP',self.error_columns,list(ERROR_TARGET_NAMES))if self.error_kind=='GP'else[
            self.load_model(self.root/name,'RF',self.error_columns,[target])for name,target in zip(self.error_files,ERROR_TARGET_NAMES)]
        self.environment=SceneEnvironment(self.root/'environment')
        self.calibration=json.loads((self.root/'calibration.json').read_text());self.policy=json.loads((self.root/'policy.json').read_text())
        hashes={name:sha256_file(self.root/name)for name in [self.model_filename]+self.error_files}
        for document in [self.calibration,self.policy]:
            if document['perception_identity']!=self.identity or document['model_sha256']!=hashes:
                raise ValueError('support-aware calibration/policy identity differs')
        self.residuals=np.asarray(self.calibration['signed_quality_residuals'],float);self.interval=np.asarray(self.calibration['interval_residual_bounds'],float)
        if (self.residuals.ndim!=1 or not len(self.residuals)or not np.isfinite(self.residuals).all()
                or(np.diff(self.residuals)<0).any()or self.interval.shape!=(2,)or not np.isfinite(self.interval).all()
                or self.interval[0]>self.interval[1]):raise ValueError('invalid support-aware residual calibration')
        self.weights=np.asarray(self.policy['utility_weights'],float)
        if self.weights.shape!=(3,)or not np.isfinite(self.weights).all()or(self.weights<0).any()or not self.weights.any():
            raise ValueError('invalid error utility weights')
        if self.policy['mode']not in ['supported_utility','quality_probability']or self.policy['grouping']not in ['global','cluster','weather','lighting','appearance']:
            raise ValueError('unknown support-aware policy')
        if self.policy['mode']=='quality_probability'and(self.policy['beta']!=0 or self.policy['information_fraction']!=0):
            raise ValueError('quality probability control cannot mix incompatible score/SD units')
        if self.error_kind!='GP'and(self.policy['beta']!=0 or self.policy['information_fraction']!=0):
            raise ValueError('RF error model has no GP latent uncertainty')
        lookup={n:i for i,n in enumerate(self.columns)}
        self.quality_take=[lookup[n]for n in self.quality_columns];self.error_take=[lookup[n]for n in self.error_columns]

    def verify_unchanged(self):
        required={'quality_model.npz','calibration.json','policy.json','environment/environment_manifest.json'}
        required.update('environment/'+name for name in ['weather.npz','lighting.npz','presence.npz','centers.npz','transforms.npz'])
        required.update(['error_model.npz']if self.spec['error_kind']=='GP'else['error_model_'+str(i)+'.npz'for i in range(3)])
        if not required.issubset(self.spec['artifact_sha256']):raise ValueError('incomplete support-aware inventory')
        for name,digest in self.spec['artifact_sha256'].items():
            path=(self.root/name).resolve();path.relative_to(self.root)
            if sha256_file(path)!=digest:raise ValueError('support-aware artifact changed: '+name)

    def load_model(self,path,kind,columns,names):
        if kind=='GP':
            model=SpatialQualityGP.load(path);validate_model_perception(model,self.spec)
            if (model.feature_names_.tolist()!=columns or model.target_names_.tolist()!=names
                    or str(model.profile_version_)!=self.profile.version or str(model.regions_sha256_)!=self.profile.regions_sha256):
                raise ValueError('support-aware GP input/target/geometry differs')
        elif kind=='RF':
            model=PortableForest.load(path);model.validate_context(columns,self.profile.version,self.profile.regions_sha256,self.identity['checkpoint_sha256'])
            if model.metadata['context']['perception_identity']!=self.identity or model.metadata['context']['target']!=names[0]:
                raise ValueError('support-aware RF target/identity differs')
        else:raise ValueError('unsupported model family')
        return model

    def predict(self,vectors,environment_vectors):
        x=np.asarray(vectors,float)
        if x.ndim!=2 or x.shape[1]!=len(self.columns)or not np.isfinite(x).all()or len(x)!=len(environment_vectors):
            raise ValueError('support-aware input vectors differ')
        qx=x[:,self.quality_take];ex=x[:,self.error_take]
        if self.kind=='GP':
            raw,cov=self.model.predict(qx);mean=np.clip(raw[:,0],0,1);scale=np.sqrt(np.maximum(cov[:,0,0],1e-6))
        else:mean=np.clip(self.model.predict(qx),0,1);scale=np.ones(len(x))
        if self.error_kind=='GP':
            error,covariance=self.errors.predict(ex,observation=False)
            utility,usd=gaussian_utility(error,covariance,self.weights)
            contrast,contrast_sd=gaussian_utility(error,covariance,np.array([1.,-1.,0.]))
        else:
            error=np.column_stack([m.predict(ex)for m in self.errors]);utility=error@self.weights;usd=np.zeros(len(x))
            contrast=error[:,0]-error[:,1];contrast_sd=np.zeros(len(x))
        env=self.environment.predict(environment_vectors)
        grouping=self.policy['grouping'];groups=np.zeros(len(x),int)if grouping=='global'else env[{'cluster':'environment_cluster','weather':'predicted_weather','lighting':'predicted_lighting','appearance':'appearance_bin'}[grouping]].copy()
        if grouping!='global':groups[env['environment_ood']]=-1
        reference=self.policy.get('reference',{'absolute_threshold':.7,'groups':{}})
        threshold=thresholds(reference,groups,unsupported=env['environment_ood'])
        low=calibrated_priority(mean,self.residuals,scale=scale)
        qscore=calibrated_priority(mean,self.residuals,scale=scale,threshold=threshold)
        score=qscore if self.policy['mode']=='quality_probability'else utility
        lower,upper=self.interval
        return dict(estimated_iou=mean,quality_predictive_scale=scale,interval90_lower=np.clip(mean+lower*scale,0,1),
                    interval90_upper=np.clip(mean+upper*scale,0,1),low_quality_probability=low,review_score=score,
                    review_score_is_probability=np.full(len(x),self.policy['mode']=='quality_probability',bool),
                    quality_gp_uncertainty_available=np.full(len(x),self.kind=='GP',bool),
                    review_gp_uncertainty_available=np.full(len(x),self.error_kind=='GP',bool),
                    error_task_scores=np.clip(error,0,1),utility_raw_mean=utility,utility_latent_sd=usd,
                    near_minus_far_error_score=contrast,near_minus_far_error_latent_sd=contrast_sd,
                    review_group=groups,review_threshold=threshold,
                    review_threshold_used=np.full(len(x),self.policy['mode']=='quality_probability',bool),**env)

    def select_with_features(self,ids,predictions,budget,vectors,*,seed=281128):
        covariance=None;noise=None
        if self.policy['information_fraction']>0 and self.policy['information_mode']=='conditional':
            if self.error_kind!='GP':raise ValueError('conditional variance audit requires GP')
            covariance=self.errors.posterior_covariance(np.asarray(vectors)[:,self.error_take],contrast=self.weights)
            noise=max(float(self.errors.noise_variance*np.sum((self.weights*self.errors.y_scale_)**2)),1e-8)
        selected,score,details=select_supported_review(ids,predictions['review_score'],predictions['utility_latent_sd'],
            predictions['review_group'],int(budget),beta=self.policy['beta'],balanced_floor=self.policy['balanced_floor'],
            audit_fraction=self.policy['audit_fraction'],information_fraction=self.policy['information_fraction'],
            information_mode=self.policy['information_mode'],covariance=covariance,noise_variance=noise,seed=seed)
        return selected,score,details

    def select(self,ids,predictions,budget,*,seed=281128):
        if self.policy['information_fraction']>0 and self.policy['information_mode']=='conditional':
            raise ValueError('conditional audit requires feature vectors')
        selected,score,_=self.select_with_features(ids,predictions,budget,np.empty((len(ids),0)),seed=seed)
        return selected,score
