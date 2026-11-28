"""Matched2500-row Q-input ablation with spatially buffered GP cross-fitting.

No confirmation data are accepted. The exact raw reference is reconstructed
before accepting augmented arms; Cal remains held out from every fitted head.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from common.bridge import sha256_file
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.review_targets import support_aware_targets,gaussian_utility
from quality_estimation.contrast_rank import standardized_contrast
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment_cycle_20261127'))
from study_data import folds,macro
from fit_interval_envelope import fit_bounds

def read(p):return json.loads(p.read_text())
def pack(p):
    if read(p.with_suffix('.json'))['sha256']!=sha256_file(p):raise ValueError('pack changed')
    with np.load(p)as z:return{k:z[k]for k in z.files}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['fit','cal','dev','schema','gp-bundle','rf-bundle','model-records','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--interval-method',choices=['pooled','road_envelope'],default='road_envelope')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);data={k:pack(getattr(a,k))for k in ['fit','cal','dev']};schema=read(a.schema);gp=SupportAwareReview(a.gp_bundle);rf=SupportAwareReview(a.rf_bundle)
    if gp.error_kind!='GP'or rf.kind!='RF'or gp.quality_columns!=rf.quality_columns:raise ValueError('matched GP/RF references required')
    for role in data:
        if read(getattr(a,role).with_suffix('.json'))['perception_identity']!=gp.identity:raise ValueError('data identity differs')
    for left,right in [('fit','cal'),('fit','dev'),('cal','dev')]:
        if set(data[left]['ids'])&set(data[right]['ids'])or set(data[left]['pixel_sha256'])&set(data[right]['pixel_sha256']):raise ValueError('split overlap')
    records={kind:read(a.model_records/('env52_quality_'+kind+'_record.json'))for kind in ['GP','RF']}
    error_record=read(a.model_records/'review_support_errors_GP_record.json')
    qcols=gp.quality_columns;ecols=gp.error_columns
    if qcols!=records['GP']['columns']or qcols!=records['RF']['columns']or ecols!=error_record['columns']:raise ValueError('parent record columns differ')
    group=lambda cols:['rgb'if c in schema['rgb']else'relation'if c in schema['relations']else'prediction'for c in cols]
    target={k:support_aware_targets(d['counts'])for k,d in data.items()};cross=np.full((len(data['fit']['ids']),5),np.nan);folds_receipt=[]
    ex=feature_matrix(data['fit'],schema,ecols);et=target['fit']['error_targets']
    for i,(train,val)in enumerate(folds(data['fit']['xy'])):
        model=SpatialQualityGP(**error_record['parameters'],feature_groups=group(ecols)).fit(ex[train],et[train]);mean,cov=model.predict(ex[val],observation=False);contrast,sd=gaussian_utility(mean,cov,np.array([1.,-1.,0.]));cross[val]=np.c_[mean,standardized_contrast(contrast,sd),sd]
        folds_receipt.append(dict(fold=i,train=train.tolist(),validation=val.tolist(),buffer_m=25))
        print('OOF',i,len(train),len(val),flush=True)
    if not np.isfinite(cross).all():raise ValueError('OOF rows missing')
    x={};y={k:t['whole_frame_iou']for k,t in target.items()}
    for role,d in data.items():
        raw=feature_matrix(d,schema,qcols)
        if role=='fit':posterior=cross
        else:
            mean,cov=gp.errors.predict(feature_matrix(d,schema,ecols),observation=False);contrast,sd=gaussian_utility(mean,cov,np.array([1.,-1.,0.]));posterior=np.c_[mean,standardized_contrast(contrast,sd),sd]
        x[role]=dict(raw=raw,mean=np.c_[raw,posterior[:,:3]],z=np.c_[raw,posterior]);np.savez_compressed(a.output/(role+'_posterior.npz'),ids=d['ids'],X=posterior)
    protocol=dict(schema='spatial-oof-gp-q-input-1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha256_file(Path(__file__)),source_pack_sha256={k:sha256_file(getattr(a,k))for k in data},perception_identity=gp.identity,folds=folds_receipt,parameters={k:r['parameters']for k,r in records.items()},error_parameters=error_record['parameters'],confirmation_accessed=False,interval_method=a.interval_method,scope='Fit rows2500 common to raw and augmented heads. Training GP features are out of fold,25m buffered. All Cal835 reserved for residuals. New confirmation outputs remain unopened; new quality hypothesis is development-only until a separate frozen decision.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=[];devpred={};calpred={}
    for kind in ['GP','RF']:
        for view in ['raw','mean','z']:
            valid=np.isfinite(y['fit']);cols=qcols+([]if view=='raw'else['posterior.near','posterior.far','posterior.offroad'])+(['posterior.near_far_z','posterior.near_far_sd']if view=='z'else[])
            if kind=='GP':model=SpatialQualityGP(**records[kind]['parameters'],feature_groups=group(cols)).fit(x['fit'][view][valid],y['fit'][valid])
            else:model=RandomForestRegressor(**records[kind]['parameters'],random_state=281128,n_jobs=2).fit(x['fit'][view][valid],y['fit'][valid])
            def predict(role):
                result=model.predict(x[role][view])
                if kind=='GP':return np.clip(result[0][:,0],0,1),np.sqrt(np.maximum(result[1][:,0,0],1e-6))
                return np.clip(result,0,1),np.ones(len(x[role][view]))
            dm,ds=predict('dev');cm,cs=predict('cal');defined=np.isfinite(y['cal']);rawres=(y['cal'][defined]-cm[defined])/cs[defined];order=np.argsort(rawres);res=rawres[order];lo,hi=fit_bounds(res,np.asarray(data['cal']['site'])[defined][order],a.interval_method);validdev=np.isfinite(y['dev']);error=abs(dm-y['dev'])
            parent_difference=None
            if view=='raw':
                reference=gp.model.predict(x['dev']['raw'])[0][:,0]if kind=='GP'else rf.model.predict(x['dev']['raw']);parent_difference=float(np.max(abs(dm-np.clip(reference,0,1))))
                if parent_difference>1e-10:raise ValueError('raw parent reconstruction differs: '+kind+' '+str(parent_difference))
            tag=kind+'_'+view;path=a.output/(tag+'.npz')
            if kind=='GP':model.feature_names_=np.array(cols);model.target_names_=np.array(['whole_frame_iou']);model.save(path)
            else:PortableForest.from_sklearn(model,cols,dict(perception_identity=gp.identity,protocol_sha256=sha256_file(a.output/'protocol.json'))).save(path)
            r=dict(tag=tag,kind=kind,view=view,columns=cols,model=path.name,model_sha256=sha256_file(path),pooled_mae=float(np.mean(error[validdev])),road_macro_mae=macro(error,data['dev']['site']),interval90_coverage=float(np.mean((y['dev'][validdev]>=np.clip(dm+lo*ds,0,1)[validdev])&(y['dev'][validdev]<=np.clip(dm+hi*ds,0,1)[validdev]))),interval90_mean_width=float(np.mean(np.clip(dm+hi*ds,0,1)-np.clip(dm+lo*ds,0,1))),cal_signed_residuals=res.tolist(),cal_bounds=[float(lo),float(hi)],raw_parent_max_prediction_difference=parent_difference)
            results.append(r);devpred[tag]=dm;calpred[tag]=cm;print(tag,r['road_macro_mae'],r['interval90_coverage'],flush=True)
    np.savez_compressed(a.output/'dev_predictions.npz',ids=data['dev']['ids'],**devpred);np.savez_compressed(a.output/'cal_predictions.npz',ids=data['cal']['ids'],**calpred)
    (a.output/'summary.json').write_text(json.dumps(dict(protocol_sha256=sha256_file(a.output/'protocol.json'),records=results,confirmation_accessed=False),indent=2)+'\n')
if __name__=='__main__':main()
