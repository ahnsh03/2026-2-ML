"""Portable RGB environment and expected near-marking context, without runtime GT."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
from common.bridge import sha256_file
from .portable_forest import PortableForest
from .spatial_features import SpatialProfile

PHOTO_COLUMNS=['hood.intensity','upper_top_left.intensity','upper_top_right.intensity',
               'upper_top_left.intensity_std','upper_top_right.intensity_std','hood.glare_fraction',
               'hood.dark_fraction','global.dark_fraction']


def photo_summary(raw):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=8 or not np.isfinite(raw).all():raise ValueError('eight finite RGB photo summaries required')
    return np.column_stack([.5*raw[:,0]+.25*(raw[:,1]+raw[:,2]),
                            .5*(raw[:,3]+raw[:,4]),raw[:,5],.5*(raw[:,6]+raw[:,7])])


class SceneEnvironment:
    def __init__(self,directory):
        self.root=Path(directory).resolve();self.spec=json.loads((self.root/'environment_manifest.json').read_text())
        if self.spec['version']!='rgb-scene-environment-1':raise ValueError('unknown RGB scene environment format')
        self.verify_unchanged()
        self.columns=self.spec['ordered_features'];self.hist_columns=self.spec['classifier_ordered_features']
        if self.columns!=self.hist_columns+PHOTO_COLUMNS or len(set(self.columns))!=len(self.columns):
            raise ValueError('RGB environment feature order differs')
        profile=SpatialProfile.load()
        if self.spec['profile']!=profile.version or self.spec['regions_sha256']!=profile.regions_sha256:
            raise ValueError('RGB environment camera differs')
        self.models={name:PortableForest.load(self.root/(name+'.npz'))for name in ['weather','lighting','presence']}
        for name,model in self.models.items():
            model.validate_context(self.hist_columns,profile.version,profile.regions_sha256)
            if model.kind!='classification' or model.classes_.tolist()!=list(range({'weather':6,'lighting':3,'presence':2}[name])):
                raise ValueError('RGB environment classifier target order differs')
        with np.load(self.root/'centers.npz',allow_pickle=False)as z:self.centers=z['centers'];self.active=z['active']
        with np.load(self.root/'transforms.npz',allow_pickle=False)as z:self.transforms={k:z[k]for k in z.files}
        dimensions={'rgb_hist_pca16':len(self.transforms['pca_variance']),'prob11':11,'photo12':12}
        self.representation=self.spec['representation']
        if self.representation not in dimensions:raise ValueError('unknown RGB environment representation')
        if (self.centers.ndim!=2 or self.centers.shape[1]!=len(self.active) or self.active.dtype.kind not in 'iu'
                or not np.isfinite(self.centers).all() or not len(self.active) or len(set(self.active.tolist()))!=len(self.active)
                or (self.active<0).any() or (self.active>=dimensions[self.representation]).any()):
            raise ValueError('invalid RGB environment centers')
        if any(not np.isfinite(v).all()for v in self.transforms.values()):raise ValueError('nonfinite RGB environment transform')
        if (self.transforms['hist_scale']<=0).any()or(self.transforms['photo_scale']<=0).any()or(self.transforms['pca_variance']<=0).any():
            raise ValueError('invalid RGB environment normalization')

    def verify_unchanged(self):
        required={'weather.npz','lighting.npz','presence.npz','centers.npz','transforms.npz'}
        if not required.issubset(self.spec['artifact_sha256']):raise ValueError('incomplete RGB environment inventory')
        for name,digest in self.spec['artifact_sha256'].items():
            path=(self.root/name).resolve();path.relative_to(self.root)
            if sha256_file(path)!=digest:raise ValueError('RGB environment artifact changed: '+name)

    def predict(self,vectors):
        x=np.asarray(vectors,float)
        if x.ndim!=2 or x.shape[1]!=len(self.columns)or not np.isfinite(x).all():raise ValueError('RGB environment inputs differ')
        hist=x[:,:len(self.hist_columns)];raw=x[:,len(self.hist_columns):];photo=photo_summary(raw)
        prob={name:model.predict_proba(hist)for name,model in self.models.items()};t=self.transforms
        if self.representation=='rgb_hist_pca16':
            embedding=(((hist-t['hist_mean'])/t['hist_scale']-t['pca_mean'])@t['pca_components'].T)/np.sqrt(t['pca_variance'])
        elif self.representation=='prob11':embedding=np.sqrt(np.maximum(np.c_[prob['weather'],prob['lighting'],prob['presence']],0))
        else:embedding=np.c_[np.sqrt(np.maximum(np.c_[prob['weather'],prob['presence']],0)),np.clip((photo-t['photo_mean'])/t['photo_scale'],-3,3)*.25]
        distance=cdist(embedding[:,self.active],self.centers,'sqeuclidean');groups=distance.argmin(1);nearest=np.sqrt(distance.min(1))
        limits=np.array([self.spec['cluster_distance_limits'].get(str(int(g)),0.)for g in groups])
        ood=(nearest>limits)|(prob['weather'].max(1)<self.spec.get('minimum_weather_confidence',.5))|(prob['presence'].max(1)<self.spec.get('minimum_presence_confidence',.6))
        return dict(environment_cluster=groups,environment_distance=nearest,environment_ood=ood,
                    predicted_weather=prob['weather'].argmax(1),weather_confidence=prob['weather'].max(1),
                    predicted_lighting=prob['lighting'].argmax(1),lighting_confidence=prob['lighting'].max(1),
                    expected_near_marking_score=prob['presence'][:,1],
                    observed_illumination=photo[:,0],appearance_bin=np.digitize(photo[:,0],t['illumination_edges']))
