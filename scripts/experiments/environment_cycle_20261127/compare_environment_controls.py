"""Matched learned-representation clustering controls; never open fresh images."""
from pathlib import Path
import argparse,time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,adjusted_rand_score
from quality_estimation.portable_forest import PortableForest
from study_data import read,dump,sha,load_sources,memberships,lighting


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study
 data,_,rows=load_sources(a.parent);member=memberships(S,data['ids']);pool=np.flatnonzero(data['role']=='candidate_pool');fit=np.unique(np.r_[pool,member['quality_fit']]);dev=member['development'];dev=dev[np.array([rows[i]['source_key'].startswith('drive_development_') for i in dev])]
 hz=[np.load(S/name) for name in ['environment_features.npz','environment_promoted_features.npz']];ids=np.concatenate([h['ids'] for h in hz]);x=np.concatenate([h['X'] for h in hz]);lookup={s:i for i,s in enumerate(ids)};env=read(S/'environment/environment_selection_frozen.json');chosen=env['selected_classifier'];hcols=hz[0]['columns'].tolist();cols=[hcols.index(n) for n in chosen['columns']]
 X=x[[lookup[s] for s in data['ids'][fit]]][:,cols];XD=x[[lookup[s] for s in data['ids'][dev]]][:,cols];oob=[];observed=[]
 targets=[data['weather'].astype(int),lighting(data['hour'])]
 for name,y in zip(['weather','lighting'],targets):
  m=RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=2,max_features=.5,class_weight='balanced_subsample',oob_score=True,random_state=271127,n_jobs=2).fit(X,y[fit]);oob.append(m.oob_decision_function_);observed.append(PortableForest.load(S/'environment'/f"{chosen['view']}_{name}.npz").predict_proba(XD))
 mapping={int(i):j for j,i in enumerate(fit)};train=np.sqrt(np.maximum(np.c_[oob[0],oob[1]],0))[[mapping[int(i)] for i in pool]];test=np.sqrt(np.maximum(np.c_[observed[0],observed[1]],0));saved=np.load(S/env['selected_cluster_artifact']);active=saved['active'];train=train[:,active];test=test[:,active];official=cdist(train,saved['centers'],'sqeuclidean').argmin(1);assert np.array_equal(official,saved['train_assigned']);records=[]
 def measures(method,seed,assigned,predicted,seconds):
  scores=[]
  for y in targets:
   values=np.unique(y[pool]);meaning={g:int(max(values,key=lambda v:((assigned==g)&(y[pool]==v)).sum()/max((y[pool]==v).sum(),1))) for g in range(16)};scores.append(float(balanced_accuracy_score(y[dev],[meaning[int(g)] for g in predicted])))
  records.append(dict(method=method,seed=seed,weather_balanced_accuracy=scores[0],lighting_balanced_accuracy=scores[1],minimum_cluster_rows=int(np.bincount(assigned,minlength=16).min()),seconds=seconds));return assigned
 measures('official_e_means',271127,official,cdist(test,saved['centers'],'sqeuclidean').argmin(1),None)
 stability={}
 for method in ['lloyd','diagonal_gmm']:
  assignments=[]
  for seed in [271127,271128,271129]:
   model=KMeans(16,n_init=3,max_iter=100,random_state=seed) if method=='lloyd' else GaussianMixture(16,covariance_type='diag',max_iter=100,reg_covar=1e-4,random_state=seed)
   start=time.monotonic();z=model.fit_predict(train);pred=model.predict(test);assignments.append(measures(method,seed,z,pred,time.monotonic()-start))
  stability[method]=float(np.mean([adjusted_rand_score(assignments[i],assignments[j]) for i,j in [(0,1),(0,2),(1,2)]]))
 stability['official_e_means']=env['selected_cluster']['seed_ari']
 dump(S/'reports/learned_representation_cluster_controls.json',dict(training_rows=len(pool),development_rows=len(dev),dimensions=len(active),clusters=16,representation='Same supervised RGB OOB probability embedding for training, full-forest predictions for Dev; square-root transform.',records=records,seed_stability_ari=stability,selected_runtime_unchanged=True,source_sha256=sha(__file__),scope='Diagnostic on existing Dev only. Unequal iteration/initialization budgets, no fair speedup claim. Improvements over raw RGB do not isolate a clustering-algorithm effect. No fresh confirmation or IoU labels accessed.'))
 print('MATCHED LEARNED CLUSTER CONTROLS',[(r['method'],r['weather_balanced_accuracy'],r['lighting_balanced_accuracy']) for r in records],flush=True)
if __name__=='__main__':main()
