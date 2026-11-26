"""Portable frozen nine-region RGB centroids; Faiss is not needed for assignment."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
from .region_candidates import candidate_schema
from .spatial_features import SpatialProfile

class RegionEnvironment:
    def __init__(self,spec,profile=None):
        self.profile=profile if profile is not None else SpatialProfile.load()
        if spec.get('version')!='roi9-environment-centers-1' or spec.get('profile')!=self.profile.version or spec.get('regions_sha256')!=self.profile.regions_sha256:
            raise ValueError('environment geometry/version differs')
        self.spec=spec;self.columns=list(spec['ordered_features']);allowed=set(candidate_schema(self.profile)['rgb'])
        if not self.columns or len(set(self.columns))!=len(self.columns) or any(n not in allowed for n in self.columns):raise ValueError('environment input must be unique RGB-only columns')
        self.selected=np.asarray(spec['selected'],int);self.mean=np.asarray(spec['mean'],float);self.scale=np.asarray(spec['scale'],float);self.centers=np.asarray(spec['centers'],float);self.radius=np.asarray(spec['reference_radius99'],float)
        n=len(self.selected)
        if self.selected.ndim!=1 or len(set(self.selected))!=n or n==0 or (self.selected<0).any() or (self.selected>=len(self.columns)).any():raise ValueError('invalid environment selection')
        if self.mean.shape!=(n,) or self.scale.shape!=(n,) or self.centers.ndim!=2 or self.centers.shape[1]!=n or self.centers.shape[0]<2 or self.radius.shape!=(len(self.centers),):raise ValueError('environment parameter dimensions differ')
        if not all(np.isfinite(v).all() for v in [self.mean,self.scale,self.centers,self.radius]) or (self.scale<=0).any() or (self.radius<0).any():raise ValueError('invalid environment normalization')
        self.constant_indices=np.asarray(spec.get('constant_feature_indices',[]),int);self.constant_values=np.asarray(spec.get('constant_feature_values',[]),float)
        if self.constant_values.shape!=self.constant_indices.shape or (self.constant_indices<0).any() or (self.constant_indices>=len(self.columns)).any() or set(self.constant_indices)&set(self.selected):raise ValueError('invalid constant-feature reference')
    @classmethod
    def load(cls,path,profile=None):return cls(json.loads(Path(path).read_text()),profile)
    def assign(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(self.columns) or not np.isfinite(x).all():raise ValueError('environment RGB feature order/values differ')
        distances=np.sqrt(cdist((x[:,self.selected]-self.mean)/self.scale,self.centers,'sqeuclidean'))
        label=distances.argmin(1);distance=distances[np.arange(len(x)),label];nearest=np.sort(distances,axis=1)[:,:2]
        changed=np.any(abs(x[:,self.constant_indices]-self.constant_values)>self.spec.get('constant_feature_tolerance',.001),axis=1) if len(self.constant_indices) else np.zeros(len(x),bool)
        return dict(environment_cluster=label,environment_distance=distance,environment_margin=nearest[:,1]-nearest[:,0],
                    environment_constant_feature_changed=changed,environment_out_of_reference=(distance>self.radius[label])|changed)

def export_environment(archive,columns,reference_values,profile=None,*,source_sha256=None):
    profile=profile if profile is not None else SpatialProfile.load();x=np.asarray(reference_values,float)
    selected=np.asarray(archive['selected'],int);mean=np.asarray(archive['mean']);scale=np.asarray(archive['scale']);centers=np.asarray(archive['centers'])
    distances=np.sqrt(cdist((x[:,selected]-mean)/scale,centers,'sqeuclidean'));labels=distances.argmin(1)
    radius=[float(np.quantile(distances[labels==k,k],.99,method='higher')) if (labels==k).any() else 0. for k in range(len(centers))]
    inactive=np.setdiff1d(np.arange(len(columns)),selected)
    spec=dict(version='roi9-environment-centers-1',profile=profile.version,regions_sha256=profile.regions_sha256,ordered_features=list(columns),
              selected=selected.tolist(),mean=mean.tolist(),scale=scale.tolist(),centers=centers.tolist(),reference_radius99=radius,source_sha256=source_sha256,
              constant_feature_indices=inactive.tolist(),constant_feature_values=x[:,inactive].mean(0).tolist(),constant_feature_tolerance=.001,
              interpretation='RGB appearance groups; centroid distance is not a weather probability. Empirical 99% reference radius is an audit flag, not an OOD guarantee.')
    RegionEnvironment(spec,profile);return spec
