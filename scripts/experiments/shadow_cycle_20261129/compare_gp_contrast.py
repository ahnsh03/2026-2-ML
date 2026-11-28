"""Prospective development-only GP contrast selection/stacking comparison.

The parent GP never fits the stacking population. Confirmation images/labels are
not inputs. Model/policy choice remains provisional until a new frozen test.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from common.bridge import sha256_file
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.review_targets import support_aware_targets,gaussian_utility
from quality_estimation.support_review_selection import select_supported_review
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest


def read(p):return json.loads(Path(p).read_text())
def pack(p):
    receipt=read(p.with_suffix('.json'))
    if receipt['sha256']!=sha256_file(p):raise ValueError('pack changed')
    with np.load(p)as z:return {k:z[k]for k in z.files}
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def prediction(bundle,data,schema):
    x=feature_matrix(data,schema,bundle.error_columns)
    if bundle.error_kind=='GP':
        mean,cov=bundle.errors.predict(x,observation=False)
        contrast,sd=gaussian_utility(mean,cov,np.array([1.,-1.,0.]))
    else:mean=np.column_stack([m.predict(x)for m in bundle.errors]);contrast=mean[:,0]-mean[:,1];sd=np.zeros(len(x))
    return mean,contrast,sd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['fit-pack','cal-pack','dev-pack','schema','rf-bundle','gp-bundle','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fit,cal,dev=[pack(v)for v in [a.fit_pack,a.cal_pack,a.dev_pack]];schema=read(a.schema)
    rf=SupportAwareReview(a.rf_bundle);gp=SupportAwareReview(a.gp_bundle)
    if rf.identity!=gp.identity:raise ValueError('different perception versions')
    for path in [a.fit_pack,a.cal_pack,a.dev_pack]:
        if read(path.with_suffix('.json'))['perception_identity']!=rf.identity:raise ValueError('different data/model version')
    for x,y in [(fit,cal),(fit,dev),(cal,dev)]:
        if set(x['ids'])&set(y['ids'])or set(x['pixel_sha256'])&set(y['pixel_sha256']):raise ValueError('stacking pixel/ID overlap')
    ctx=dict(version='gp-contrast-development-protocol-1',perception_identity=rf.identity,
             inputs={k:sha256_file(getattr(a,k))for k in ['fit_pack','cal_pack','dev_pack','schema']},
             bundle_sha256={k:sha256_file(getattr(a,k)/'bundle_manifest.json')for k in ['rf_bundle','gp_bundle']},
             source_sha256=sha256_file(Path(__file__)),seed=291129,budget_fraction=.1,audit_fraction=.1,
             prospective_confirmation_required=True,calibration_role='Independent from parent fitting; now meta-training, cannot evaluate meta-model on it.',
             variants=['frozen_RF','frozen_GP','GP_z','GP_P_contrast_positive','RF_rank_plus_z_0.25','RF_rank_plus_z_0.5',
                       'RF_meta_raw','RF_meta_mean','RF_meta_z','GP_meta_raw','GP_meta_mean','GP_meta_z'])
    dump(a.output/'protocol.json',ctx);target={role:support_aware_targets(d['counts'])for role,d in [('cal',cal),('dev',dev)]}
    predictions={};features={};scores={};weights=np.array([1.,.25,.5]);base=rf.error_columns
    for role,data in [('cal',cal),('dev',dev)]:
        rm,_,_=prediction(rf,data,schema);gm,c,sd=prediction(gp,data,schema)
        # Fixed numerical safeguard, never chosen using test labels.
        z=np.clip(c/np.maximum(sd,1e-4),-10,10)
        predictions[role]=dict(rf_mean=rm,gp_mean=gm,contrast=c,contrast_sd=sd,z=z)
        raw=feature_matrix(data,schema,base)
        features[role]=dict(raw=raw,mean=np.c_[raw,gm],z=np.c_[raw,gm,z,sd])
        scores[role]=dict(frozen_RF=rm@weights,frozen_GP=gm@weights,GP_z=z,GP_P_contrast_positive=ndtr(z))
    # Empirical Cal ranks bring heterogeneous units onto one fixed reference scale.
    for alpha in [.25,.5]:
        for role in ['cal','dev']:
            erank=lambda name:np.searchsorted(np.sort(scores['cal'][name]),scores[role][name],side='right')/len(cal['ids'])
            scores[role]['RF_rank_plus_z_'+str(alpha)]=(1-alpha)*erank('frozen_RF')+alpha*erank('GP_z')
    y=target['cal']['error_targets'];began=time.monotonic();models=[]
    for view in ['raw','mean','z']:
        for kind in ['RF','GP']:
            x=features['cal'][view];dx=features['dev'][view];tag=kind+'_meta_'+view
            columns=base+([]if view=='raw'else['GP.near_mean','GP.far_mean','GP.offroad_mean'])+(['GP.near_minus_far_z','GP.near_minus_far_latent_sd']if view=='z'else[])
            if kind=='RF':
                ms=[RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=4,max_features=.7,random_state=291129,n_jobs=2).fit(x,y[:,j])for j in range(3)]
                pred=np.column_stack([m.predict(dx)for m in ms]);paths=[]
                for j,m in enumerate(ms):
                    path=a.output/(tag+'_'+str(j)+'.npz');PortableForest.from_sklearn(m,columns,dict(perception_identity=rf.identity,stacking_protocol_sha256=sha256_file(a.output/'protocol.json'))).save(path);paths.append(path)
            else:
                m=SpatialQualityGP(length_multiplier=2.8,noise_variance=.1,task_correlation=.5,kernel_mode='isotropic').fit(x,y)
                pred,_=m.predict(dx,observation=False);path=a.output/(tag+'.npz');m.feature_names_=np.array(columns);m.target_names_=np.array(['near_error','far_error','offroad_error']);m.save(path);paths=[path]
            scores['dev'][tag]=pred@weights;models.append(dict(tag=tag,columns=columns,sha256={q.name:sha256_file(q)for q in paths}))
            print('FIT',tag,round(time.monotonic()-began,1),flush=True)
    d=dev;t=target['dev'];budget=int(np.ceil(.1*len(d['ids'])));records=[];selections={}
    for name,score in scores['dev'].items():
        ix,priority,detail=select_supported_review(d['ids'],score,np.zeros(len(score)),np.zeros(len(score),int),budget,audit_fraction=.1,seed=281128)
        mask=np.zeros(len(score),bool);mask[ix]=True
        def recall(value,where=None):
            if where is None:where=np.ones(len(mask),bool)
            den=np.sum(value[where]);return float(np.sum(value[mask&where])/den)if den else None
        def macro(v):
            rs=[recall(v,d['site']==s)for s in np.unique(d['site'])];return float(np.mean([r for r in rs if r is not None]))
        near=t['near']['error'];far=t['far']['error'];valid=near|far
        rec=dict(name=name,rows=len(mask),budget=budget,near_error_recall=recall(near),near_error_road_macro=macro(near),
                 utility_recall=recall(t['review_utility']),utility_road_macro=macro(t['review_utility']),
                 near_fn_recall=recall(t['near']['false_negative']),near_fp_recall=recall(t['near']['false_positive']),
                 near_error_auc=float(roc_auc_score(near,score)),near_vs_far_only_auc=float(roc_auc_score(near[valid],score[valid]))if np.unique(near[valid]).size==2 else None,
                 no_error_selection_fraction=float(np.mean(t['review_utility'][ix]==0)),by_hour={str(int(h)):dict(near_errors=int(np.sum(near&(d['hour']==h))),recall=recall(near,d['hour']==h))for h in np.unique(d['hour'])})
        records.append(rec);selections[name]=d['ids'][ix].tolist();print(name,rec['near_error_recall'],rec['utility_road_macro'],flush=True)
    np.savez_compressed(a.output/'dev_scores.npz',ids=d['ids'],**scores['dev'])
    np.savez_compressed(a.output/'cal_posterior_features.npz',ids=cal['ids'],**predictions['cal'])
    np.savez_compressed(a.output/'dev_posterior_features.npz',ids=dev['ids'],**predictions['dev'])
    result=dict(protocol_sha256=sha256_file(a.output/'protocol.json'),records=records,models=models,selected_ids=selections,
                fitting_rows=len(fit['ids']),stacking_rows=len(cal['ids']),development_rows=len(dev['ids']),
                source_sha256=sha256_file(Path(__file__)),scope='Development comparison, no new independent test claim; z estimates signed supported-error contrast, not camera visibility/safety or calibrated failure probability.')
    dump(a.output/'summary.json',result);(a.output/'_SUCCESS').write_text(sha256_file(a.output/'summary.json')+'\n')
if __name__=='__main__':main()
