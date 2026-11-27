"""Fixed quality-fit controls: RGB environment identity without perception features."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from quality_estimation.context_baseline import fit_context_mean,predict_context_mean
from study_data import read,labels,lighting,macro


def context_controls(study,version,review,data,prediction,review_table):
    S=Path(study);spec=read(S/'protocol/context_controls_frozen.json');index=read(S/'protocol/quality_index_frozen.json');fit_ids=[r['sample_id'] for r in index['quality_fit']['rows']]
    if version=='f1':
        with np.load(S/'g1_packs/quality_fit.npz') as z:fit={k:z[k] for k in z.files}
    else:
        with np.load(S/'f0_quality/base28_GP_predictions.npz') as z:lookup={s:i for i,s in enumerate(z['ids'])};fit_q=z['Y'][[lookup[s] for s in fit_ids]]
        fit={key:np.array([r[key] for r in index['quality_fit']['rows']]) for key in ['site','weather','hour']};fit['ids']=np.array(fit_ids)
    fit_q=labels(fit)[0] if version=='f1' else fit_q
    if fit['ids'].tolist()!=fit_ids:raise ValueError('context-control quality-fit ordering differs')
    archives=[np.load(S/name) for name in ['environment_features.npz','environment_promoted_features.npz']];all_ids=np.concatenate([z['ids'] for z in archives]);all_x=np.concatenate([z['X'] for z in archives]);lookup={s:i for i,s in enumerate(all_ids)};columns=archives[0]['columns'].tolist();hx=all_x[[lookup[s] for s in fit_ids]][:,[columns.index(n) for n in review.environment.columns]]
    wp=review.weather.predict_proba(hx);lp=review.lighting.predict_proba(hx);embedding=np.sqrt(np.maximum(np.c_[wp,lp],0))[:,review.active];clusters=cdist(embedding,review.centers,'sqeuclidean').argmin(1);joint=fit['weather'].astype(int)*3+lighting(fit['hour']);rgb_joint=wp.argmax(1)*3+lp.argmax(1)
    definitions={'rgb_lighting_mean':(lp.argmax(1),prediction['predicted_lighting'],False),'rgb_tag_mean':(rgb_joint,prediction['predicted_weather']*3+prediction['predicted_lighting'],False),'rgb_cluster_mean':(clusters,prediction['environment_cluster'],False),'sim_tag_mean_oracle':(joint,data['weather'].astype(int)*3+lighting(data['hour']),True)}
    q,harm,normal=labels(data);valid=np.isfinite(q);out={}
    for name,(train_groups,test_groups,oracle) in definitions.items():
        model=fit_context_mean(fit_q,train_groups,fit['site'],prior_rows=spec['prior_rows'],min_rows=spec['min_rows'],min_roads=spec['min_roads']);mean=predict_context_mean(model,test_groups);out[name]=dict(uses_simulator_metadata_at_inference=oracle,macro_road_mae=macro(abs(mean[valid]-q[valid]),data['site'][valid]),model=model,review=review_table(data['ids'],1-mean,q,harm,normal,data['site']),scope='Fit on the same defined quality-fit targets as GP/RF, with fixed shrinkage; no test/Cal targets in estimation. RGB groups computed by the frozen RGB classifiers/centers. Simulator-group control is offline only. Not used for acceptance or post-test model choice.')
    return out
