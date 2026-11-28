"""Version-bound RF utility and GP signed contrast ranks for review selection.

The fixed Cal reference contains no online GT input. A z-score uses the full
GP near-minus-far latent covariance. Neither z nor the blended score is a
calibrated probability. Unsupported validation must keep this policy optional.
"""
from pathlib import Path
import json
import numpy as np
from common.bridge import sha256_file
from .perception_identity import validate_identity


def standardized_contrast(mean,sd,*,floor=1e-4,cap=10.):
    mean=np.asarray(mean,float);sd=np.asarray(sd,float)
    if mean.shape!=sd.shape or mean.ndim!=1 or not np.isfinite(mean).all()or not np.isfinite(sd).all()or(sd<0).any():raise ValueError('finite aligned GP contrast/SD required')
    if floor<=0 or cap<=0 or not np.isfinite([floor,cap]).all():raise ValueError('invalid fixed contrast safeguard')
    return np.clip(mean/np.maximum(sd,floor),-cap,cap)


class ContrastRankPolicy:
    def __init__(self,path,rf_bundle,gp_bundle,identity):
        self.path=Path(path);self.spec=json.loads(self.path.read_text());d=self.spec
        if d['schema']!='gp-contrast-rank-policy-1'or validate_identity(d['perception_identity'])!=validate_identity(identity):raise ValueError('contrast policy identity differs')
        for key,bundle in [('rf',rf_bundle),('gp',gp_bundle)]:
            if sha256_file(Path(bundle)/'bundle_manifest.json')!=d['bundle_manifest_sha256'][key]:raise ValueError('contrast policy/model differs')
        if not 0<=d['gp_rank_weight']<=1:raise ValueError('invalid contrast blend weight')
        self.rf=np.asarray(d['reference_rf_utility'],float);self.z=np.asarray(d['reference_gp_z'],float)
        for values in [self.rf,self.z]:
            if values.ndim!=1 or not len(values)or not np.isfinite(values).all()or(np.diff(values)<0).any():raise ValueError('invalid frozen Cal rank reference')
    def score(self,rf,gp):
        z=standardized_contrast(gp['near_minus_far_error_score'],gp['near_minus_far_error_latent_sd'],floor=self.spec['sd_floor'],cap=self.spec['z_cap'])
        utility=np.asarray(rf['utility_raw_mean'],float)
        if utility.shape!=z.shape or not np.isfinite(utility).all():raise ValueError('RF utility and GP contrast differ')
        rank=lambda ref,x:np.searchsorted(ref,x,side='right')/len(ref)
        alpha=self.spec['gp_rank_weight']
        return (1-alpha)*rank(self.rf,utility)+alpha*rank(self.z,z)
