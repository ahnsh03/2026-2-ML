"""Official e-means recipe study and dev-only equal-budget BO review search."""
from pathlib import Path
import argparse,json,time,itertools,hashlib,inspect
import faiss
from faiss.contrib.e_means import NumpyEmeans
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,silhouette_score
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.cluster_review import review_scores,select_review
from quality_estimation.bounded_bo import next_expected_improvement
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;faiss.omp_set_num_threads(2)
 schema=json.loads((S/'schema.json').read_text());z=np.load(S/'source_packs.npz');data={k:z[k] for k in z.files};rows=[json.loads(s) for s in (S/'protocol/all_sources.jsonl').read_text().splitlines()]
 pool=np.flatnonzero(data['role']=='candidate_pool');dev=np.flatnonzero(data['role']=='development');cal=np.flatnonzero(data['role']=='calibration')
 rgb=schema['rgb'];top=[n for n in rgb if n.startswith(('upper_top_','hood.'))];photometric=['intensity','intensity_std','dark_fraction','glare_fraction','intensity_p90_minus_p10','saturation_median','blue_minus_red','hue_cos','hue_sin']
 views={'top_hood54':top,'top_hood27':[n for n in top if n.split('.')[1] in photometric],
        'upper_hood90':[n for n in rgb if n.startswith(('upper_top_','upper_bottom_','hood.'))],
        'all_rgb180':rgb}
 output=S/'clusters';output.mkdir(exist_ok=True);records=[];models={};seeds=[261126,261127,261128]
 for view,columns in views.items():
  raw=feature_matrix(data,schema,columns);selected=np.flatnonzero(raw[pool].var(0)>1e-12);mu=raw[pool][:,selected].mean(0);sd=raw[pool][:,selected].std(0);x=(raw[:,selected]-mu)/sd
  for k in [8,16,32]:
   trials=[];labels=[]
   for seed in seeds:
    t=time.monotonic();centers,_=NumpyEmeans(x.shape[1],k,seed=seed).train(np.ascontiguousarray(x[pool],np.float32))
    # Official train labels are last-pass assignments, not final nearest centers.
    d=cdist(x,centers,'sqeuclidean');assignment=d.argmin(1);labels.append(assignment[pool]);counts=np.bincount(assignment[pool],minlength=k)
    score=float(silhouette_score(x[pool],assignment[pool],sample_size=1200,random_state=261126))
    path=output/f'{view}_k{k}_s{seed}.npz';np.savez_compressed(path,centers=centers,selected=selected,mean=mu,scale=sd,ordered_features=columns,assignment=assignment,distance=np.sqrt(d.min(1)))
    trial=dict(view=view,k=k,seed=seed,columns=columns,dimensions=len(columns),active_dimensions=len(selected),seconds=time.monotonic()-t,
               occupied=int((counts>0).sum()),min_cluster=int(counts.min()),largest_fraction=float(counts.max()/len(pool)),silhouette=score,
               weather_nmi=float(normalized_mutual_info_score(data['weather'][pool],assignment[pool])),hour_nmi=float(normalized_mutual_info_score(data['hour'][pool],assignment[pool])),
               path=str(path.relative_to(S)),sha256=hashlib.sha256(path.read_bytes()).hexdigest());trials.append(trial);models[str(path.relative_to(S))]=assignment
   ari=float(np.mean([adjusted_rand_score(labels[i],labels[j]) for i,j in [(0,1),(0,2),(1,2)]]))
   rec=dict(view=view,k=k,trials=trials,seed_ari=ari,mean_weather_nmi=float(np.mean([t['weather_nmi'] for t in trials])),mean_hour_nmi=float(np.mean([t['hour_nmi'] for t in trials])),mean_silhouette=float(np.mean([t['silhouette'] for t in trials])))
   # Prefer environment information/stability; avoid very small unsupported groups.
   rec['recipe_score']=.5*(rec['mean_weather_nmi']+rec['mean_hour_nmi'])+.2*ari+.1*rec['mean_silhouette']-.05*np.log2(k/8)
   records.append(rec);print('clustered',view,k,'score',rec['recipe_score'],flush=True)
 eligible=[r for r in records if all(t['min_cluster']>=20 for t in r['trials'])]
 winner=max(eligible,key=lambda r:r['recipe_score']);chosen=winner['trials'][0];labels=models[chosen['path']]
 source=Path(inspect.getfile(NumpyEmeans))
 dump(output/'study.json',dict(package_version=faiss.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
      official_source='https://github.com/facebookresearch/faiss/blob/main/contrib/e_means/README.md',records=records,selected=chosen,
      recipe_selection='Train-candidate RGB and offline environment metadata only; no review/test or perception-update result used. First seed of winning recipe, not best seed.'))
 # Evaluate review policies exclusively on reserved development labels.
 near=data['counts'][:,6:8].sum(1);support=near[:,0]+near[:,2]>=50
 nr=np.divide(near[:,0],near[:,0]+near[:,2],out=np.full(len(rows),np.nan),where=near[:,0]+near[:,2]>0)
 np_=np.divide(near[:,0],near[:,0]+near[:,1],out=np.full(len(rows),np.nan),where=near[:,0]+near[:,1]>0)
 harm=support&((nr<.8)|(np_<.8));normal=support&(nr>=.9)&(np_>=.9)
 fraction=3072/len(pool);budget=int(np.ceil(len(dev)*fraction));ids=data['ids'];mean=data['g0_mean'][:,0];var=data['g0_covariance'][:,0,0]
 params=[dict(uncertainty_weight=u,quota_exponent=q,audit_fraction=au) for u,q,au in itertools.product([0.,.1,.25],[0.,.5,1.],[0.,.1,.25])]
 coordinates=np.array([[v['uncertainty_weight']/.25,v['quota_exponent'],v['audit_fraction']/.25] for v in params]);objectives={};details={}
 def evaluate(index):
  if index in objectives:return objectives[index]
  v=params[index];score=review_scores(mean[dev],var[dev],uncertainty_weight=v['uncertainty_weight']);ix=dev[select_review(ids[dev],score,labels[dev],budget,quota_exponent=v['quota_exponent'],audit_fraction=v['audit_fraction'])];picked=np.isin(dev,ix)
  def recalls(groups):
   vals=[]
   for g in np.unique(groups):
    sel=(groups==g)&harm[dev]
    if sel.sum()>=5:vals.append(float(picked[sel].mean()))
   return float(np.mean(vals)) if vals else 0.
  geo=recalls(data['site'][dev]);env=recalls(labels[dev]);normal_fraction=float(normal[ix].mean());loss=.5*(1-geo)+.3*(1-env)+.2*normal_fraction
  details[index]=dict(parameters=v,macro_road_harm_recall=geo,macro_cluster_harm_recall=env,selected_normal_fraction=normal_fraction,loss=loss)
  objectives[index]=loss;return loss
 initial=[0,8,18,26];hist=[];losses=[]
 for _ in range(16):
  i=next_expected_improvement(coordinates,hist,losses,initial);hist.append(i);losses.append(evaluate(i))
 random=list(initial)+np.random.default_rng(261126).permutation([i for i in range(len(params)) if i not in initial]).tolist()[:12];rloss=[evaluate(int(i)) for i in random]
 chosen_index=min(hist,key=lambda i:(objectives[i],params[i]['uncertainty_weight'],params[i]['audit_fraction']));best=params[chosen_index]
 # Equal-cost random search is reported; BO's best is not presumed superior.
 dump(S/'protocol/policy_frozen.json',dict(cluster=chosen,selection_parameters=best,bo_history=hist,bo_losses=losses,random_history=random,random_losses=rloss,
      trials={str(k):v for k,v in details.items()},objective='Development-only macro-road/cluster near precision-or-recall deficit retrieval plus normal-review cost; NOT retraining benefit.',
      initial_shared=initial,budget=len(hist),review_fraction=fraction,development_review_budget=budget,
      failure_definition='Absolute IoU<0.7 remains separate from near FP/FN diagnostic. Far-only recall weakness is not operational failure truth.',
      threshold_rule='No threshold lowered without supported normal visibility/precision references; cluster quotas affect review priority only.'))
 score=review_scores(mean[pool],var[pool],uncertainty_weight=best['uncertainty_weight'])
 selections={'Random':np.random.default_rng(261126).choice(pool,3072,replace=False),
             'QualityRank':pool[np.lexsort((ids[pool],-review_scores(mean[pool],var[pool])))[:3072]],
             'ClusterBO':pool[select_review(ids[pool],score,labels[pool],3072,quota_exponent=best['quota_exponent'],audit_fraction=best['audit_fraction'])]}
 for arm,indices in selections.items():
  admitted=[dict(rows[i],selection={'arm':arm,'rank':j+1,'environment_cluster':int(labels[i]),'policy_sha256':hashlib.sha256((S/'protocol/policy_frozen.json').read_bytes()).hexdigest()},
                 label_admission='User-authorized offline native Semantic review; mask validity/exposure checks required') for j,i in enumerate(indices)]
  jsonl(S/f'protocol/{arm}_sources.jsonl',admitted)
 dump(S/'reports/admission_summary.json',{arm:dict(new_rows=len(ix),conditions=dict(collections_counter(zip(data['weather'][ix].tolist(),data['hour'][ix].tolist()))),clusters=np.bincount(labels[ix],minlength=chosen['k']).tolist()) for arm,ix in selections.items()})
 print('FROZEN POLICY',chosen['view'],chosen['k'],best,'BO',min(losses),'random',min(rloss),flush=True)
def collections_counter(values):
 from collections import Counter
 return {str(k):v for k,v in Counter(values).items()}
if __name__=='__main__':main()
