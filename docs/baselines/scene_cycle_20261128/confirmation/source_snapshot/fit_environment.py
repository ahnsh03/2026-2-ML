"""Fit RGB-only environment/presence representations with geographic cross-fitting."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score,adjusted_rand_score,adjusted_mutual_info_score
from faiss.contrib.e_means import NumpyEmeans
from quality_estimation.review_targets import support_aware_targets
from quality_estimation.portable_forest import PortableForest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment_cycle_20261127'))
from study_data import read,dump,sha,folds,lighting

PHOTO_COLUMNS=['hood.intensity','upper_top_left.intensity','upper_top_right.intensity',
               'upper_top_left.intensity_std','upper_top_right.intensity_std','hood.glare_fraction',
               'hood.dark_fraction','global.dark_fraction']


def photo_summary(raw):
    return np.column_stack([.5*raw[:,0]+.25*(raw[:,1]+raw[:,2]),
                            .5*(raw[:,3]+raw[:,4]),raw[:,5],.5*(raw[:,6]+raw[:,7])])


def semantic_mapping(groups,target,k):
    classes=np.unique(target);freq={c:max(int((target==c).sum()),1)for c in classes}
    return [int(max(classes,key=lambda c:((groups==g)&(target==c)).sum()/freq[c]))for g in range(k)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();study=args.study;out=study/'environment_fit';out.mkdir(exist_ok=False)
    with np.load(study/'fit_environment_rgb400.npz')as z:hist=z['X'];ids=z['ids'];hcols=z['columns'].tolist()
    with np.load(study/'fit_all_parent_features.npz')as z:data={k:z[k]for k in z.files}
    if ids.tolist()!=data['ids'].tolist():raise ValueError('RGB environment IDs differ')
    schema=read(study/'schema.json');raw=data['rgb'][:,[schema['rgb'].index(n)for n in PHOTO_COLUMNS]]
    photo=photo_summary(raw);photo_scaler=StandardScaler().fit(photo)
    illumination_edges=np.quantile(photo[:,0],[.2,.6]);appearance=np.digitize(photo[:,0],illumination_edges)
    targets=dict(weather=data['weather'].astype(int),lighting=lighting(data['hour']),presence=(support_aware_targets(data['counts'])['scene_gt_class']==2).astype(int))
    oof={};full={};scores={};splits=folds(data['xy'])
    for name,target in targets.items():
        classes=np.unique(target);values=np.zeros((len(ids),len(classes)));fold_scores=[]
        for train,val in splits:
            model=RandomForestClassifier(n_estimators=160,max_depth=16,min_samples_leaf=2,max_features=.5,
                class_weight='balanced_subsample',random_state=281128,n_jobs=2).fit(hist[train],target[train])
            if model.classes_.tolist()!=classes.tolist():raise ValueError('missing environment class in buffered fold')
            values[val]=model.predict_proba(hist[val]);fold_scores.append(float(balanced_accuracy_score(target[val],model.predict(hist[val]))))
        oof[name]=values
        model=RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=2,max_features=.5,
            class_weight='balanced_subsample',random_state=281128,n_jobs=2).fit(hist,target)
        full[name]=model.predict_proba(hist)
        portable=PortableForest.from_sklearn(model,hcols,dict(profile=schema['profile'],regions_sha256=schema['regions_sha256'],
            target=name,role='RGB-only; independent of perception weights; training labels are offline metadata/native-GT presence'))
        portable.save(out/(name+'.npz'));scores[name]=dict(fold_balanced_accuracy=fold_scores,mean=float(np.mean(fold_scores)),
            crossfit_balanced_accuracy=float(balanced_accuracy_score(target,classes[values.argmax(1)])))
        print('RGB CLASSIFIER',name,scores[name],flush=True)
    scaler=StandardScaler().fit(hist);scaled=scaler.transform(hist);pca=PCA(16,whiten=True,random_state=281128).fit(scaled)
    hist_repr=pca.transform(scaled)
    def reps(prob):
        return dict(rgb_hist_pca16=hist_repr,
                    prob11=np.sqrt(np.maximum(np.c_[prob['weather'],prob['lighting'],prob['presence']],0)),
                    photo12=np.c_[np.sqrt(np.maximum(np.c_[prob['weather'],prob['presence']],0)),np.clip(photo_scaler.transform(photo),-3,3)*.25])
    training=reps(oof);in_sample=reps(full)
    np.savez_compressed(out/'transforms.npz',hist_mean=scaler.mean_,hist_scale=scaler.scale_,pca_mean=pca.mean_,
        pca_components=pca.components_,pca_variance=pca.explained_variance_,photo_mean=photo_scaler.mean_,photo_scale=photo_scaler.scale_,
        illumination_edges=illumination_edges)
    np.savez_compressed(out/'crossfit_predictions.npz',ids=ids,**{n+'_probabilities':x for n,x in oof.items()},illumination=photo[:,0],appearance=appearance)
    records=[]
    for name,xx in training.items():
        active=np.flatnonzero(xx.var(0)>1e-12);xx=xx[:,active]
        for k in [8,12,16]:
            for method in ['official_e_means','lloyd']:
                assignments=[];current=[]
                for seed in [281128,281129,281130]:
                    began=time.monotonic()
                    if method=='official_e_means':centers,_=NumpyEmeans(xx.shape[1],k,seed=seed,nepochs=5).train(np.ascontiguousarray(xx,np.float32))
                    else:centers=KMeans(k,n_init=3,max_iter=100,random_state=seed).fit(xx).cluster_centers_
                    elapsed=time.monotonic()-began;dist=cdist(xx,centers,'sqeuclidean');assigned=dist.argmin(1);assignments.append(assigned)
                    mappings={n:semantic_mapping(assigned,t,k)for n,t in dict(targets,appearance=appearance).items()}
                    path=out/f'{name}_{method}_k{k}_s{seed}.npz';np.savez_compressed(path,centers=centers,active=active,train_assigned=assigned)
                    counts=np.bincount(assigned,minlength=k)
                    record=dict(representation=name,method=method,k=k,seed=seed,model=path.name,model_sha256=sha(path),
                        active_dimensions=len(active),inertia=float(dist.min(1).mean()),counts=counts.tolist(),fit_seconds=elapsed,
                        semantic_mapping=mappings,train_ami={n:float(adjusted_mutual_info_score(t,assigned))for n,t in targets.items()},
                        cluster_distance_limits={str(g):float(np.quantile(np.sqrt(dist[assigned==g,g]),.99))if (assigned==g).any()else 0. for g in range(k)})
                    records.append(record);current.append(record)
                ari=float(np.mean([adjusted_rand_score(assignments[i],assignments[j])for i,j in [(0,1),(0,2),(1,2)]]))
                for record in current:record['seed_ari']=ari
                print('CLUSTER FIT',name,k,method,'ARI',ari,flush=True)
    document=dict(classifier_scores=scores,records=records,classifier_ordered_features=hcols,
        environment_ordered_features=hcols+PHOTO_COLUMNS,photo_columns=PHOTO_COLUMNS,illumination_edges=illumination_edges.tolist(),
        cross_fitted_centroids=True,fit_rows=len(ids),source_sha256=sha(__file__),histogram_source_sha256=sha(study/'fit_environment_rgb400.npz'),
        scope='Fit only. Select grouping on new Dev; do not open independent confirmation. Observed illumination bins are image measurements, not SDK clock prediction accuracy.')
    dump(out/'fit_complete.json',document);(out/'_SUCCESS').write_text(sha(out/'fit_complete.json')+'\n')


if __name__=='__main__':main()
