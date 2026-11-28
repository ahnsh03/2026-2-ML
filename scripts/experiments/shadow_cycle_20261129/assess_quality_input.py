"""Evaluate a blinded, fixed quality-input hypothesis without retuning."""
from pathlib import Path
from datetime import datetime
import argparse,json
import numpy as np
from common.bridge import sha256_file
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.review_targets import support_aware_targets,gaussian_utility
from quality_estimation.contrast_rank import standardized_contrast
from quality_estimation.spatial_gp import SpatialQualityGP

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['freeze','model-root','pack','schema','parent-gp-bundle','reference-assessment','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);f=read(a.freeze);parent=SupportAwareReview(a.parent_gp_bundle)
    if sha256_file(a.parent_gp_bundle/'bundle_manifest.json')!=f['parent_gp_bundle_manifest_sha256']or parent.identity!=f['perception_identity']:raise ValueError('parent GP changed')
    path=a.model_root/f['model']
    if sha256_file(path)!=f['model_sha256']:raise ValueError('fixed quality model changed')
    # The additional hypothesis was fixed during collection, before test feature extraction.
    if datetime.fromtimestamp(a.pack.stat().st_mtime).timestamp()<datetime.fromisoformat(f['created_utc']).timestamp():raise ValueError('test pack existed before additional hypothesis freeze')
    with np.load(a.pack)as z:d={k:z[k]for k in z.files}
    receipt=read(a.pack.with_suffix('.json'))
    if receipt['perception_identity']!=parent.identity or receipt['sha256']!=sha256_file(a.pack):raise ValueError('quality test pack version differs')
    schema=read(a.schema);model=SpatialQualityGP.load(path);columns=f['model_ordered_columns'];rawcols=parent.quality_columns
    if model.feature_names_.tolist()!=columns or columns[:len(rawcols)]!=rawcols:raise ValueError('meta quality feature recipe differs')
    ex=feature_matrix(d,schema,parent.error_columns);mean,cov=parent.errors.predict(ex,observation=False);contrast,sd=gaussian_utility(mean,cov,np.array([1.,-1.,0.]));z=standardized_contrast(contrast,sd)
    x=np.c_[feature_matrix(d,schema,rawcols),mean,z,sd];raw,qcov=model.predict(x);q=np.clip(raw[:,0],0,1);scale=np.sqrt(np.maximum(qcov[:,0,0],1e-6));lower,upper=f['calibration']['interval_bounds'];lo=np.clip(q+lower*scale,0,1);hi=np.clip(q+upper*scale,0,1)
    target=support_aware_targets(d['counts'])['whole_frame_iou'];valid=np.isfinite(target);error=abs(q-target);roads=np.unique(d['site'])
    by_road={str(r):float(np.mean(error[(d['site']==r)&valid]))for r in roads if((d['site']==r)&valid).any()}
    old=read(a.reference_assessment/'summary.json');refs=old['quality'];delta={};improved={}
    for name in ['frozen_GP','RF_only']:
        with np.load(a.reference_assessment/(name+'_predictions.npz'))as pred:
            if pred['ids'].tolist()!=d['ids'].tolist():raise ValueError('paired quality membership differs')
            pq=pred['estimated_iou']
        olderr=abs(pq-target);diff=np.array([float(np.mean(olderr[(d['site']==r)&valid]))-by_road[str(r)]for r in roads if str(r)in by_road])
        boot=np.random.default_rng(291129).choice(diff,size=(10000,len(diff)),replace=True).mean(1);delta[name]=dict(mae_improvement=float(diff.mean()),segment_differences=diff.tolist(),paired_road_bootstrap95=np.quantile(boot,[.025,.975]).tolist());improved[name]=int((diff>0).sum())
    mae=float(error[valid].mean());macro=float(np.mean(list(by_road.values())));g=f['acceptance'];failed=[]
    if refs['frozen_GP']['quality_road_macro_mae']-macro<g['minimum_road_macro_mae_gain_vs_GP52']-1e-12:failed.append('GP52_mae_gain')
    if mae-refs['RF_only']['quality_mae']>g['maximum_pooled_mae_loss_vs_RF52']+1e-12:failed.append('RF52_quality_guard')
    if improved['frozen_GP']<g['minimum_improved_segments']:failed.append('segment_consistency')
    result=dict(schema='blinded-quality-input-assessment-1',rows=len(q),perception_identity=parent.identity,tag=f['tag'],quality_mae=mae,quality_road_macro_mae=macro,quality_mae_by_segment=by_road,interval90_coverage=float(np.mean((target[valid]>=lo[valid])&(target[valid]<=hi[valid]))),interval90_mean_width=float(np.mean(hi-lo)),by_hour={str(int(h)):dict(rows=int((d['hour']==h).sum()),mae=float(np.mean(error[(d['hour']==h)&valid])))for h in np.unique(d['hour'])},paired_road_mae_improvement=delta,gate=dict(passes=not failed,failed=failed,improved_segments_vs_GP52=improved['frozen_GP']),source_sha256=sha256_file(Path(__file__)),input_sha256=dict(freeze=sha256_file(a.freeze),pack=sha256_file(a.pack),parent_bundle=sha256_file(a.parent_gp_bundle/'bundle_manifest.json')),scope=f['hypothesis_timing']+' Point-quality improvement is separate from review selection and perception retraining. Empirical interval, no uncertain-input integration guarantee.')
    np.savez_compressed(a.output/'predictions.npz',ids=d['ids'],estimated_iou=q,scale=scale,interval90_lower=lo,interval90_upper=hi)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output/'_SUCCESS').write_text(sha256_file(a.output/'summary.json')+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
