"""Check e-means optimization adequacy after a documented Dev stability failure."""
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from faiss.contrib.e_means import NumpyEmeans
from common.bridge import sha256_file
from quality_estimation.review_targets import support_aware_targets
from quality_estimation.scene_environment import SceneEnvironment,PHOTO_COLUMNS,photo_summary
from fit_environment import semantic_mapping
from evaluate_development import environment_vectors,environment_candidates


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['study','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();s=a.study;out=a.output;out.mkdir(exist_ok=False);base=s/'environment_fit';spec=read(base/'fit_complete.json');protocol=read(s/'protocol/emeans_pass_refinement.json');schema=read(s/'schema.json')
    for path in base.glob('*.npz'):
        try:os.link(path,out/path.name)
        except OSError:shutil.copyfile(path,out/path.name)
    with np.load(s/'fit_all_parent_features.npz')as z:data={k:z[k]for k in z.files}
    with np.load(base/'crossfit_predictions.npz')as z:
        if z['ids'].tolist()!=data['ids'].tolist():raise ValueError('cross-fit environment IDs differ')
        weather=z['weather_probabilities'];presence=z['presence_probabilities'];appearance=z['appearance']
    with np.load(base/'transforms.npz')as z:transforms={k:z[k]for k in z.files}
    raw=data['rgb'][:,[schema['rgb'].index(n)for n in PHOTO_COLUMNS]];photo=photo_summary(raw)
    xx=np.c_[np.sqrt(np.maximum(np.c_[weather,presence],0)),np.clip((photo-transforms['photo_mean'])/transforms['photo_scale'],-3,3)*.25]
    active=np.flatnonzero(xx.var(0)>1e-12);xx=xx[:,active];hour=data['hour'];light=np.where((hour>=8)&(hour<18),0,np.where(((hour>=6)&(hour<8))|((hour>=18)&(hour<20)),1,2))
    targets=dict(weather=data['weather'],lighting=light,presence=(support_aware_targets(data['counts'])['scene_gt_class']==2).astype(int),appearance=appearance)
    records=[]
    for k in protocol['k']:
        for passes in protocol['nepochs']:
            current=[];assignments=[]
            for seed in protocol['seeds']:
                began=time.monotonic();centers,_=NumpyEmeans(xx.shape[1],k,seed=seed,nepochs=passes).train(np.ascontiguousarray(xx,np.float32));seconds=time.monotonic()-began
                distance=cdist(xx,centers,'sqeuclidean');groups=distance.argmin(1);assignments.append(groups)
                path=out/f'photo12_official_e_means_ep{passes}_k{k}_s{seed}.npz';np.savez_compressed(path,centers=centers,active=active,train_assigned=groups)
                record=dict(representation='photo12',method='official_e_means_ep'+str(passes),k=k,seed=seed,nepochs=passes,model=path.name,model_sha256=sha256_file(path),
                    active_dimensions=len(active),inertia=float(distance.min(1).mean()),counts=np.bincount(groups,minlength=k).tolist(),fit_seconds=seconds,
                    semantic_mapping={n:semantic_mapping(groups,y,k)for n,y in targets.items()},train_ami={n:float(adjusted_mutual_info_score(y,groups))for n,y in targets.items()},
                    cluster_distance_limits={str(g):float(np.quantile(np.sqrt(distance[groups==g,g]),.99))if(groups==g).any()else 0. for g in range(k)})
                current.append(record);records.append(record)
            ari=float(np.mean([adjusted_rand_score(assignments[i],assignments[j])for i,j in [(0,1),(0,2),(1,2)]]))
            for record in current:record['seed_ari']=ari
            print('REFINED E-MEANS',k,passes,ari,flush=True)
    spec.update(records=spec['records']+records,base_fit_sha256=sha256_file(base/'fit_complete.json'),refinement_protocol_sha256=sha256_file(s/'protocol/emeans_pass_refinement.json'),source_sha256=sha256_file(Path(__file__)),scope='Fit-only center refinement after disclosed Dev stability failure. Same fit RGB/classifiers, default-pass and Lloyd controls retained; no confirmation access.')
    write(out/'fit_complete.json',spec)
    env=SceneEnvironment(s/'support_smoke_bundle/environment')
    with np.load(s/'parent_cohort_packs/development.npz')as z:dev={k:z[k]for k in z.files}
    vectors=environment_vectors(dev,s/'development_environment_features.npz',schema,env.hist_columns);spec['_directory']=out
    selected,record,results,classifier=environment_candidates(spec,env,vectors,dev)
    write(out/'development_refinement.json',dict(selected=selected,selected_record=record,records=results,classifier_scores=classifier,
        old_winner=read(s/'parent_development_selection/selection.json')['environment_selected'],independent_confirmation_accessed=False,
        source_sha256=sha256_file(Path(__file__)),scope='Dev-only context selection; the previously selected global utility policy and bank scores are not changed.'))
    (out/'_SUCCESS').write_text(sha256_file(out/'fit_complete.json')+'\n');print('REFINED ENVIRONMENT CHOICE',selected,flush=True)


if __name__=='__main__':main()
