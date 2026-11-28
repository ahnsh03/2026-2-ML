"""Report fixed RGB/cluster semantics on new confirmation, without refitting maps."""
from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score
from common.bridge import sha256_file
from quality_estimation.scene_environment import SceneEnvironment
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.review_targets import support_aware_targets
from evaluate_development import environment_vectors, lighting


def read(path):
    return json.loads(Path(path).read_text())


def classification(target, prediction, classes):
    target=np.asarray(target,int);prediction=np.asarray(prediction,int)
    if target.shape!=prediction.shape or target.ndim!=1 or not len(target):
        raise ValueError('nonempty aligned environment targets required')
    if classes<1 or (target<0).any() or (prediction<0).any() or (target>=classes).any() or (prediction>=classes).any():
        raise ValueError('environment class outside fixed label order')
    # Report supported classes explicitly; absent classes are not invented.
    supported=np.unique(target)
    recalls={str(int(c)):float(np.mean(prediction[target==c]==c))for c in supported}
    return dict(balanced_accuracy=float(np.mean(list(recalls.values()))),
                accuracy=float(np.mean(target==prediction)),recall_by_supported_class=recalls,
                supported_classes=supported.tolist(),class_order=list(range(classes)),
                confusion_matrix=confusion_matrix(target,prediction,labels=list(range(classes))).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['freeze','review-assessment','schema','environment-features','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();freeze=read(a.freeze);assessment=read(a.review_assessment);schema=read(a.schema)
    if assessment['confirmation_freeze_sha256']!=sha256_file(a.freeze):
        raise ValueError('environment diagnostic requires the frozen confirmation assessment')
    if a.output.exists():raise ValueError('preserve existing environment confirmation')
    results={}
    for name in ['primary','Lloyd_environment_control']:
        candidate=next(r for r in freeze['review_candidates']if r['name']==name)
        bundle=Path(candidate['bundle']);pack=Path(candidate['pack']);receipt=read(pack.with_suffix('.json'))
        if sha256_file(bundle/'bundle_manifest.json')!=candidate['bundle_manifest_sha256']:
            raise ValueError('frozen environment source changed')
        validate_pack_perception(receipt,candidate['perception_identity'],'model_state_dict')
        if receipt['sha256']!=sha256_file(pack):raise ValueError('confirmation pack changed')
        env=SceneEnvironment(bundle/'environment')
        with np.load(pack)as z:data={k:z[k]for k in z.files}
        vectors=environment_vectors(data,a.environment_features,schema,env.hist_columns)
        pred=env.predict(vectors);groups=pred['environment_cluster'];target=support_aware_targets(data['counts'])
        truth=dict(weather=data['weather'].astype(int),lighting=lighting(data['hour']),
                   presence=(target['scene_gt_class']==2).astype(int),appearance=pred['appearance_bin'])
        direct=dict(weather=pred['predicted_weather'],lighting=pred['predicted_lighting'],
                    presence=(pred['expected_near_marking_score']>=.5).astype(int))
        classes=dict(weather=6,lighting=3,presence=2,appearance=3)
        mapped={task:np.asarray(env.spec['semantic_mapping'][task],int)[groups]for task in truth}
        result=dict(rows=len(groups),representation=env.representation,method=env.spec['method'],k=env.spec['k'],seed=env.spec['seed'],
            classifier={task:classification(truth[task],value,classes[task])for task,value in direct.items()},
            cluster_semantic={task:classification(truth[task],value,classes[task])for task,value in mapped.items()},
            cluster_ami={task:float(adjusted_mutual_info_score(value,groups))for task,value in truth.items()},
            cluster_counts=np.bincount(groups,minlength=env.spec['k']).tolist(),ood_fraction=float(pred['environment_ood'].mean()),
            weather_accuracy_by_segment={str(site):float(np.mean(direct['weather'][data['site']==site]==truth['weather'][data['site']==site]))for site in np.unique(data['site'])},
            environment_manifest_sha256=sha256_file(env.root/'environment_manifest.json'),source_pack_sha256=sha256_file(pack),
            scope='RGB classifiers use offline SDK/native-GT training labels. Cluster semantic maps were fixed on fit, not relabeled on confirmation. Appearance bins are measured image brightness, not SDK clock ground truth.')
        results[name]=result;env.verify_unchanged()
    document=dict(version='frozen-environment-context-confirmation-1',results=results,
        confirmation_freeze_sha256=sha256_file(a.freeze),review_assessment_sha256=sha256_file(a.review_assessment),
        source_sha256=sha256_file(Path(__file__)),no_confirmation_refit=True,automatic_method_replacement=False,
        scope='Semantic context diagnostics separate from the utility of review selection. A better diagnostic control is not substituted after confirmation.')
    a.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    print('ENVIRONMENT CONFIRMATION', {n:r['cluster_semantic']['weather']['balanced_accuracy']for n,r in results.items()},flush=True)


if __name__=='__main__':main()
