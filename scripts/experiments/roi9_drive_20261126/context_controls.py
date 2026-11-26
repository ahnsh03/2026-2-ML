"""Evaluate previously frozen context-only controls, without additional fitting.

RGB54 uses the primary GP kernel settings, not an independently tuned best RGB
model. Cluster/metadata predictions are old training-IoU medians, not weather
classification accuracy. Weather/time metadata is an offline oracle only.
"""
from pathlib import Path
import argparse
import numpy as np
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from feature_study import read,save,sha,load,subset
from calibrate import assign_environment,group_mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','extension','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();e=a.extension;root=a.study/'context_control_frozen'
    record=read(root/'frozen.json')
    assert sha(root/'rgb54_GP.npz')==record['model_sha256']
    assert record['created_utc']<read(e/'plans/frozen.json')['created_utc']
    index=read(e/'registry/index_full.json');thin=read(e/'registry/index_thinned.json')
    data,schema=load(e/'packs',list(index))
    ids={r['sample_id'] for v in thin.values() for r in v['rows']}
    data=subset(data,np.isin(data['ids'],list(ids)))
    model=SpatialQualityGP.load(root/'rgb54_GP.npz')
    mean,_=model.predict(feature_matrix(data,schema,record['ordered_features']))
    fine,_,_=assign_environment(data,schema,read(root/'environment_profile.json'))
    cluster=np.array([record['cluster_medians'].get(str(int(k)),{'median':record['fallback_median']})['median'] for k in fine])
    metadata=np.array([record['metadata_medians'].get(f'{w}:{h}',{'median':record['fallback_median']})['median'] for w,h in zip(data['weather'],data['hour'])])
    with np.load(e/'evaluation/predictions.npz',allow_pickle=False) as z:
        lookup={sid:i for i,sid in enumerate(z['ids'])}
        ix=[lookup[sid] for sid in data['ids']]
        primary=z['raw_mean'][ix,0];rf=z['rf_mean'][ix,0]
    candidates=dict(primary_GP28=primary,matched_RF28=rf,RGB54_fixed_kernel_GP=mean[:,0],e_means_old_IoU_median=cluster,weather_time_old_IoU_median_ORACLE=metadata)
    target=data['Y'][:,0];results={}
    for name,q in candidates.items():
        error=abs(np.clip(q,0,1)-target)
        results[name]=dict(road_macro_mae=group_mean(error,data['site']),byroad={str(g):float(error[data['site']==g].mean()) for g in np.unique(data['site'])})
    conditions=[]
    for w,h in sorted(set(zip(data['weather'],data['hour']))):
        ix=(data['weather']==w)&(data['hour']==h)
        conditions.append(dict(weather=int(w),hour=int(h),rows=int(ix.sum()),GP_mae=float(abs(np.clip(primary[ix],0,1)-target[ix]).mean()),RF_mae=float(abs(np.clip(rf[ix],0,1)-target[ix]).mean())))
    save(a.output,dict(context_frozen_sha256=sha(root/'frozen.json'),evaluation_sha256=sha(e/'evaluation/results.json'),rows=len(target),roads=len(set(data['site'])),results=results,conditions=conditions,scope=__doc__))


if __name__=='__main__':main()
