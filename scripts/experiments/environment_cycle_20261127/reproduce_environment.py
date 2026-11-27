"""Reproduce the selected learned RGB representation and official e-means recipe."""
from pathlib import Path
import argparse,inspect,json,hashlib
import faiss,numpy as np
from faiss.contrib.e_means import NumpyEmeans
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
from study_data import read,dump,sha,memberships,lighting

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;z=np.load(a.parent/'source_packs_v2.npz');data={k:z[k] for k in ['ids','role','weather','hour']};members=memberships(S,data['ids']);pool=np.flatnonzero(data['role']=='candidate_pool');fit=np.unique(np.r_[pool,members['quality_fit']]);hz=[np.load(S/name) for name in ['environment_features.npz','environment_promoted_features.npz']];ids=np.concatenate([h['ids'] for h in hz]);x=np.concatenate([h['X'] for h in hz]);lookup={s:i for i,s in enumerate(ids)};env=read(S/'environment/environment_selection_frozen.json');chosen=env['selected_classifier'];hcols=hz[0]['columns'].tolist();X=x[[lookup[s] for s in data['ids'][fit]]][:,[hcols.index(n) for n in chosen['columns']]];probs=[]
 for y in [data['weather'].astype(int),lighting(data['hour'])]:
  m=RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=2,max_features=.5,class_weight='balanced_subsample',oob_score=True,random_state=271127,n_jobs=2).fit(X,y[fit]);probs.append(m.oob_decision_function_)
 mapping={int(i):j for j,i in enumerate(fit)};raw=np.sqrt(np.maximum(np.c_[probs[0],probs[1]],0));train=raw[[mapping[int(i)] for i in pool]];active=np.flatnonzero(train.var(0)>1e-12);train=np.ascontiguousarray(train[:,active],np.float32);faiss.omp_set_num_threads(2);model=NumpyEmeans(train.shape[1],env['selected_cluster']['k'],seed=271127,nepochs=5);centers,last_epoch=model.train(train);old=np.load(S/env['selected_cluster_artifact']);delta=float(abs(centers-old['centers']).max());assigned=cdist(train,centers,'sqeuclidean').argmin(1);equal=bool(np.array_equal(assigned,old['train_assigned']));source=Path(inspect.getfile(NumpyEmeans));recipe=getattr(model,'recipe',None)
 def serial(value):
  if isinstance(value,np.ndarray):return value.tolist()
  if isinstance(value,np.generic):return value.item()
  return str(value)
 report=dict(package='faiss-cpu',version=faiss.__version__,implementation_source_sha256=sha(source),official_source='https://github.com/facebookresearch/faiss/blob/main/contrib/e_means/README.md',paper='https://arxiv.org/abs/2611.09501',training_rows=len(train),dimensions=train.shape[1],clusters=len(centers),passes=5,cpu_threads=2,recipe=json.loads(json.dumps(recipe,default=serial)),oob_probability_embedding_sha256=hashlib.sha256(train.tobytes()).hexdigest(),reproduced_centroid_max_difference=delta,reproduced_nearest_assignments_equal=equal,source_sha256=sha(__file__),scope='Re-fitted exactly the fixed supervised RGB classifier/OOB representation. No candidate IoU or fresh data accessed. Final nearest-centroid reassignment matches documented official label semantics; no10x speedup claim.')
 dump(S/'reports/clustering_implementation_reproduction.json',report);assert delta<1e-6 and equal;print('ENVIRONMENT REPRODUCED',delta,equal,flush=True)
if __name__=='__main__':main()
