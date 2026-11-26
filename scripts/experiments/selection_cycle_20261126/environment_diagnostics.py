"""Appearance/position confounds and shared-initialization e-means/Lloyd diagnostics."""
from pathlib import Path
import argparse,json,time,collections
import faiss,numpy as np
from faiss.contrib.e_means import NumpyEmeans
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from quality_estimation.region_feature_views import feature_matrix
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;faiss.omp_set_num_threads(2)
 z=np.load(S/'source_packs_v2.npz');d={k:z[k] for k in z.files};schema=json.loads((S/'schema.json').read_text());chosen=json.loads((S/'protocol/policy_frozen.json').read_text())['cluster'];archive=np.load(S/chosen['path']);x=feature_matrix(d,schema,chosen['columns']);scaled=(x[:,archive['selected']]-archive['mean'])/archive['scale'];pool=d['role']=='candidate_pool';labels=archive['assignment'];blocks=np.floor(d['xy']/50).astype(int);geo=np.array([str(tuple(p)) for p in blocks]);report={}
 for role in ['candidate_pool','development','calibration']:
  mask=d['role']==role;report[role]=dict(rows=int(mask.sum()),weather_nmi=float(normalized_mutual_info_score(d['weather'][mask],labels[mask])),hour_nmi=float(normalized_mutual_info_score(d['hour'][mask],labels[mask])),geography50m_nmi=float(normalized_mutual_info_score(geo[mask],labels[mask])),counts=np.bincount(labels[mask],minlength=8).tolist())
 conditioned=[]
 for weather,hour in sorted(set(zip(d['weather'][pool].tolist(),d['hour'][pool].tolist()))):
  mask=pool&(d['weather']==weather)&(d['hour']==hour);counts=np.bincount(labels[mask],minlength=8);conditioned.append(dict(weather=int(weather),hour=int(hour),rows=int(mask.sum()),occupied=int((counts>0).sum()),largest_fraction=float(counts.max()/mask.sum())))
 common=[];values=np.ascontiguousarray(scaled[pool],np.float32)
 for seed in [261126,261127,261128]:
  initial=values[np.random.default_rng(seed).choice(len(values),8,replace=False)].copy();out={};assignments={}
  for name,passes in [('emeans5',5),('lloyd5',5),('lloyd25',25)]:
   start=time.monotonic()
   if name.startswith('emeans'):
    model=NumpyEmeans(values.shape[1],8,seed=seed,nepochs=passes);centers,_=model.train(values,init_centroids=initial.copy());iterations=passes
   else:
    model=KMeans(n_clusters=8,init=initial.copy(),n_init=1,max_iter=passes,tol=0.,random_state=seed,algorithm='lloyd').fit(values);centers=model.cluster_centers_;iterations=int(model.n_iter_)
   elapsed=time.monotonic()-start;distance=cdist(values,centers,'sqeuclidean');assignment=distance.argmin(1);assignments[name]=assignment
   out[name]=dict(seconds=elapsed,actual_iterations=iterations,mean_squared_distance=float(distance.min(1).mean()),occupied=len(set(assignment)))
  common.append(dict(seed=seed,results=out,emeans5_vs_lloyd25_ari=float(adjusted_rand_score(assignments['emeans5'],assignments['lloyd25']))))
 dump(S/'reports/environment_diagnostics.json',dict(assignment=report,within_condition=conditioned,shared_initialization_comparison=common,
      scope='Frozen candidate-selected centers only; no recipe or training admissions changed by these diagnostics. Small7442x24/27 CPU problem, not a reproduction of paper-scale speedups. Geography/metadata are diagnostics, never online quality inputs.'))
 print(report,common,flush=True)
if __name__=='__main__':main()
