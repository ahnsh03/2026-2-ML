"""Build a hash-bound offline candidate; this operation does not adopt/deploy it."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import numpy as np
from common.bridge import sha256_file
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.review_targets import support_aware_targets
from quality_estimation.support_aware_review import SupportAwareReview
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment_cycle_20261127'))
from fit_interval_envelope import fit_bounds


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def link(source,target):
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['models','environment-fit','cal-pack','schema','policy','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--quality-tag',required=True);parser.add_argument('--error-tag',required=True)
    parser.add_argument('--environment-key',required=True,help='representation,method,k,seed')
    parser.add_argument('--interval-method',choices=['pooled','road_envelope'],default='pooled')
    parser.add_argument('--purpose',choices=['development_prototype','frozen_confirmation_candidate'],default='development_prototype')
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('candidate output exists; never overwrite a frozen bundle')
    fit=read(args.models/'fit_complete.json');records={r['tag']:r for r in fit['records']}
    quality=records[args.quality_tag];errors=records[args.error_tag]
    if quality['task']!='quality'or errors['task']!='errors'or quality['perception_identity']!=errors['perception_identity']:
        raise ValueError('quality/error task or perception version differs')
    identity=quality['perception_identity'];schema=read(args.schema);cal_receipt=read(args.cal_pack.with_suffix('.json'))
    validate_pack_perception(cal_receipt,identity,'model_state_dict')
    if cal_receipt['sha256']!=sha256_file(args.cal_pack):raise ValueError('calibration pack changed')
    settings=read(args.policy)
    if settings.get('perception_identity',identity)!=identity:raise ValueError('policy settings belong to a different perception model')
    out.mkdir(parents=True);env_out=out/'environment';env_out.mkdir()
    for record,names in [(quality,['quality_model.npz']),
                         (errors,['error_model.npz']if errors['kind']=='GP'else['error_model_'+str(i)+'.npz'for i in range(3)])]:
        if len(record['models'])!=len(names):raise ValueError('model output inventory differs')
        for relative,name in zip(record['models'],names):
            source=(args.models/relative).resolve();source.relative_to(args.models.resolve())
            if sha256_file(source)!=record['model_sha256'][relative]:raise ValueError('fitted model changed')
            link(source,out/name)
    env_fit=read(args.environment_fit/'fit_complete.json');rep,method,k,seed=args.environment_key.split(',')
    candidate=next(r for r in env_fit['records']if(r['representation'],r['method'],r['k'],r['seed'])==(rep,method,int(k),int(seed)))
    source=args.environment_fit/candidate['model']
    if sha256_file(source)!=candidate['model_sha256']:raise ValueError('environment candidate changed')
    for name in ['weather.npz','lighting.npz','presence.npz','transforms.npz']:link(args.environment_fit/name,env_out/name)
    with np.load(source)as z:np.savez_compressed(env_out/'centers.npz',centers=z['centers'],active=z['active'])
    environment=dict(version='rgb-scene-environment-1',representation=rep,method=method,k=int(k),seed=int(seed),
        profile=schema['profile'],regions_sha256=schema['regions_sha256'],ordered_features=env_fit['environment_ordered_features'],
        classifier_ordered_features=env_fit['classifier_ordered_features'],cluster_distance_limits=candidate['cluster_distance_limits'],
        semantic_mapping=candidate['semantic_mapping'],minimum_weather_confidence=.5,minimum_presence_confidence=.6,
        training_source_sha256=sha256_file(args.environment_fit/'fit_complete.json'),
        artifact_sha256={p.name:sha256_file(p)for p in env_out.iterdir()if p.is_file()})
    write(env_out/'environment_manifest.json',environment)
    with np.load(args.cal_pack)as z:cal={k:z[k]for k in z.files}
    x=feature_matrix(cal,schema,quality['columns']);q=support_aware_targets(cal['counts'])['whole_frame_iou']
    if quality['kind']=='GP':
        mean,cov=SpatialQualityGP.load(out/'quality_model.npz').predict(x);mean=np.clip(mean[:,0],0,1);scale=np.sqrt(np.maximum(cov[:,0,0],1e-6))
    else:mean=np.clip(PortableForest.load(out/'quality_model.npz').predict(x),0,1);scale=np.ones(len(x))
    valid=np.isfinite(q);residuals=np.sort((q[valid]-mean[valid])/scale[valid])
    if len(residuals)<30:raise ValueError('insufficient defined calibration quality')
    bounds=fit_bounds(residuals,np.asarray(cal['site'])[valid][np.argsort((q[valid]-mean[valid])/scale[valid])],args.interval_method)
    hashes={p.name:sha256_file(p)for p in out.glob('*.npz')}
    calibration=dict(version='support-aware-calibration-1',perception_identity=identity,model_sha256=hashes,
        signed_quality_residuals=residuals.tolist(),interval_residual_bounds=bounds.tolist(),interval_method=args.interval_method,
        source_pack_sha256=sha256_file(args.cal_pack),defined_rows=int(valid.sum()),source_sha256=sha256_file(Path(__file__)),
        scope='Empirical signed residual distribution and interval bounds, not guaranteed conditional coverage.')
    write(out/'calibration.json',calibration)
    policy=dict(settings,perception_identity=identity,model_sha256=hashes,source_settings_sha256=sha256_file(args.policy),
                adopted=False,purpose=args.purpose)
    write(out/'policy.json',policy)
    manifest=dict(version='support-aware-review-bundle-1',purpose=args.purpose,perception_identity=identity,
        perception_weights='model_state_dict',checkpoint_sha256=identity['checkpoint_sha256'],
        profile=schema['profile'],regions_sha256=schema['regions_sha256'],quality_kind=quality['kind'],error_kind=errors['kind'],
        quality_columns=quality['columns'],error_columns=errors['columns'],ordered_features=list(dict.fromkeys(quality['columns']+errors['columns'])),
        quality_model_record=quality,error_model_record=errors,environment_selection=candidate,
        artifact_sha256={str(p.relative_to(out)):sha256_file(p)for p in out.rglob('*')if p.is_file()},
        source_sha256=sha256_file(Path(__file__)),automatic_admission=False,live_connection_completed=False)
    write(out/'bundle_manifest.json',manifest)
    SupportAwareReview(out,perception_identity=identity)
    (out/'_SUCCESS').write_text(sha256_file(out/'bundle_manifest.json')+'\n')
    print('SUPPORT-AWARE BUNDLE VERIFIED',out,quality['tag'],errors['tag'],args.environment_key,flush=True)


if __name__=='__main__':main()
