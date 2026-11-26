"""Offline environment-relative review references, separate from absolute failure flags."""
import numpy as np

def weighted_quantile(values,weights,quantile):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];w=np.asarray(weights,float)[order]
    return float(v[min(int(np.searchsorted(np.cumsum(w),quantile*w.sum(),side='left')),len(v)-1)])

def fit_relative_references(quality,clusters,roads,xy,*,quantile=.25,shrinkage=100.,absolute_threshold=.7,min_rows=30,min_roads=2):
    q=np.asarray(quality,float);c=np.asarray(clusters);roads=np.asarray(roads);xy=np.asarray(xy,float)
    if q.ndim!=1 or c.shape!=q.shape or roads.shape!=q.shape or xy.shape!=(len(q),2) or not np.isfinite(xy).all():raise ValueError('reference dimensions differ')
    if not 0<quantile<1 or shrinkage<=0 or not 0<absolute_threshold<1:raise ValueError('invalid reference parameters')
    groups={}
    for key in np.unique(c):
        mask=(c==key)&np.isfinite(q);n=int(mask.sum());g=roads[mask];unique,inv,count=np.unique(g,return_inverse=True,return_counts=True)
        effective=len(np.unique(np.floor(xy[mask]/5).astype(np.int64),axis=0))
        valid=key>=0 and n>=min_rows and len(unique)>=min_roads and effective>=10
        value=weighted_quantile(q[mask],1/count[inv],quantile) if valid else None
        weight=effective/(effective+shrinkage) if valid else 0.
        threshold=float(np.clip((1-weight)*absolute_threshold+weight*value,.3,absolute_threshold)) if valid else absolute_threshold
        groups[str(int(key))]=dict(threshold=threshold,reference_quantile=value,rows=n,roads=len(unique),unique_5m_cells=effective,shrinkage_weight=weight,eligible=valid)
    return dict(version='relative-review-reference-1',absolute_quality_threshold=absolute_threshold,groups=groups,quantile=quantile,shrinkage=shrinkage,
                scope='Relative priority within observed environment distribution. This never converts low absolute IoU to a quality pass or operational normality label.')

def reference_thresholds(spec,clusters,*,ood=None):
    clusters=np.asarray(clusters);values=np.array([spec['groups'].get(str(int(c)),{}).get('threshold',spec['absolute_quality_threshold']) for c in clusters])
    if ood is not None:values[np.asarray(ood,bool)]=spec['absolute_quality_threshold']
    return values
