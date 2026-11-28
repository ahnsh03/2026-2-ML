"""Blinded development-only test of GP contrast inputs for whole-frame IoU."""
from pathlib import Path
import argparse,json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from common.bridge import sha256_file
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.review_targets import support_aware_targets,gaussian_utility
from quality_estimation.contrast_rank import standardized_contrast
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest

def read(p):return json.loads(p.read_text())
def pack(p):
    if read(p.with_suffix('.json'))['sha256']!=sha256_file(p):raise ValueError('pack changed')
    with np.load(p)as z:return{k:z[k]for k in z.files}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['parent-fit','meta-fit','meta-cal','dev','schema','bundle','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);source={n:pack(getattr(a,n))for n in ['parent_fit','meta_fit','meta_cal','dev']};schema=read(a.schema);r=SupportAwareReview(a.bundle)
    if r.error_kind!='GP':raise ValueError('GP error head required')
    values=list(source.values())
    for i,left in enumerate(values):
        for right in values[i+1:]:
            if set(left['ids'])&set(right['ids'])or set(left['pixel_sha256'])&set(right['pixel_sha256']):raise ValueError('meta/cal/parent pixel leakage')
    for n in source:
        if read(getattr(a,n).with_suffix('.json'))['perception_identity']!=r.identity:raise ValueError('model/data version differs')
    recipe=dict(schema='whole-q-gp-contrast-meta-development-1',perception_identity=r.identity,parent_bundle_manifest_sha256=sha256_file(a.bundle/'bundle_manifest.json'),source_sha256=sha256_file(Path(__file__)),parent_fit_rows=len(source['parent_fit']['ids']),meta_fit_rows=len(source['meta_fit']['ids']),meta_cal_rows=len(source['meta_cal']['ids']),dev_rows=len(source['dev']['ids']),source_pack_sha256={n:sha256_file(getattr(a,n))for n in source},confirmation_accessed=False,scope='Additional blinded quality-input hypothesis; no existing selector candidate or test membership change. Meta Cal is reserved for residual calibration, never model fitting.',fixed_hyperparameters=dict(GP=dict(length_multiplier=2.8,noise_variance=.1,task_correlation=0.,kernel_mode='isotropic'),RF=dict(n_estimators=200,max_depth=12,min_samples_leaf=4,max_features=.7,random_state=291129,n_jobs=2)))
    (a.output/'protocol.json').write_text(json.dumps(recipe,indent=2)+'\n');features={};ys={};parent={}
    for role in ['meta_fit','meta_cal','dev']:
        d=source[role];x=feature_matrix(d,schema,r.quality_columns);ex=feature_matrix(d,schema,r.error_columns)
        mean,cov=r.errors.predict(ex,observation=False);contrast,sd=gaussian_utility(mean,cov,np.array([1.,-1.,0.]));z=standardized_contrast(contrast,sd)
        features[role]=dict(raw=x,mean=np.c_[x,mean],z=np.c_[x,mean,z,sd]);ys[role]=support_aware_targets(d['counts'])['whole_frame_iou']
        parent[role]=np.clip(r.model.predict(x)[0][:,0],0,1)
    records=[];validfit=np.isfinite(ys['meta_fit']);validcal=np.isfinite(ys['meta_cal']);d=source['dev'];validdev=np.isfinite(ys['dev'])
    for kind in ['GP','RF']:
        for view in ['raw','mean','z']:
            tag=kind+'_'+view;params=recipe['fixed_hyperparameters'][kind]
            model=SpatialQualityGP(**params).fit(features['meta_fit'][view][validfit],ys['meta_fit'][validfit])if kind=='GP'else RandomForestRegressor(**params).fit(features['meta_fit'][view][validfit],ys['meta_fit'][validfit])
            def pred(role):
                value=model.predict(features[role][view]);return np.clip(value[0][:,0]if kind=='GP'else value,0,1)
            cm=pred('meta_cal');dm=pred('dev');residual=ys['meta_cal'][validcal]-cm[validcal];lower,upper=np.quantile(residual,[.05,.95])
            error=abs(dm-ys['dev']);covered=(ys['dev']>=np.clip(dm+lower,0,1))&(ys['dev']<=np.clip(dm+upper,0,1))
            columns=r.quality_columns+([]if view=='raw'else['GP.near_mean','GP.far_mean','GP.offroad_mean'])+(['GP.near_far_z','GP.near_far_latent_sd']if view=='z'else[])
            path=a.output/(tag+'.npz')
            if kind=='GP':
                model.feature_names_=np.array(columns);model.target_names_=np.array(['whole_frame_lane_iou']);model.save(path)
            else:PortableForest.from_sklearn(model,columns,dict(perception_identity=r.identity,quality_input_protocol_sha256=sha256_file(a.output/'protocol.json'))).save(path)
            rec=dict(tag=tag,kind=kind,view=view,features=columns,model_sha256=sha256_file(path),model=path.name,mae=float(error[validdev].mean()),road_macro_mae=float(np.mean([np.mean(error[(d['site']==s)&validdev])for s in np.unique(d['site'])if((d['site']==s)&validdev).any()])),residual_bounds=[float(lower),float(upper)],interval90_coverage=float(covered[validdev].mean()),interval90_mean_width=float(np.mean(np.clip(dm+upper,0,1)-np.clip(dm+lower,0,1))),fitted_rows=int(validfit.sum()),residual_calibration_rows=int(validcal.sum()))
            records.append(rec);np.savez_compressed(a.output/(tag+'_dev.npz'),ids=d['ids'],estimated_iou=dm);print(tag,rec['road_macro_mae'],rec['interval90_coverage'],flush=True)
    parenterr=abs(parent['dev']-ys['dev']);parent_mae=float(np.mean([np.mean(parenterr[(d['site']==s)&validdev])for s in np.unique(d['site'])if((d['site']==s)&validdev).any()]))
    result=dict(protocol_sha256=sha256_file(a.output/'protocol.json'),records=records,parent_GP_road_macro_mae=parent_mae,scope='Development only; new-confirmation outputs remain unopened. Do not adopt from Dev alone or fit residuals on meta-training.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output/'_SUCCESS').write_text(sha256_file(a.output/'summary.json')+'\n')
if __name__=='__main__':main()
