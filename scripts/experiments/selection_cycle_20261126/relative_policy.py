"""Cal-only environment references and dev-only BO; absolute quality stays 0.7."""
from pathlib import Path
import argparse,json,itertools
import numpy as np
from scipy.spatial.distance import cdist
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.cluster_review import review_scores,select_review
from quality_estimation.environment_references import fit_relative_references,reference_thresholds
from quality_estimation.bounded_bo import next_expected_improvement
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--source-packs',default='source_packs.npz');p.add_argument('--output-name',default='relative_policy_frozen.json');a=p.parse_args();S=a.study
 z=np.load(S/a.source_packs);d={k:z[k] for k in z.files};schema=json.loads((S/'schema.json').read_text());policy=json.loads((S/'protocol/policy_frozen.json').read_text());cluster=policy['cluster'];model=np.load(S/cluster['path']);labels=model['assignment']
 pool=np.flatnonzero(d['role']=='candidate_pool');dev=np.flatnonzero(d['role']=='development');cal=np.flatnonzero(d['role']=='calibration')
 near=d['counts'][:,6:8].sum(1);ng=near[:,0]+near[:,2];nr=np.divide(near[:,0],ng,out=np.full(len(ng),np.nan),where=ng>0);np_=np.divide(near[:,0],near[:,0]+near[:,1],out=np.full(len(ng),np.nan),where=near[:,0]+near[:,1]>0)
 harm=(ng>=50)&((nr<.8)|(np_<.8));normal=(ng>=50)&(nr>=.9)&(np_>=.9)
 # F0 morphology counts support does not depend on prediction for GT count.
 cal=cal[(ng[cal]>=50)&np.isfinite(d['y'][cal])]
 features=feature_matrix(d,schema,cluster['columns'])[:,model['selected']];scaled=(features-model['mean'])/model['scale'];distance=np.sqrt(cdist(scaled,model['centers'],'sqeuclidean').min(1))
 limits={int(c):float(np.quantile(distance[pool][labels[pool]==c],.99)) for c in np.unique(labels[pool])};ood=np.array([distance[i]>limits[int(labels[i])] for i in range(len(labels))])
 budget=int(np.ceil(len(dev)*3072/len(pool)));base=policy['selection_parameters'];ids=d['ids'];mu=d['g0_mean'][:,0];var=d['g0_covariance'][:,0,0]
 params=[dict(quantile=q,shrinkage=s,quota_exponent=e,uncertainty_weight=u) for q,s,e,u in itertools.product([.1,.25,.4],[25.,100.,400.],[0.,.5,1.],[0.,.25])]
 coords=np.clip(np.array([[(v['quantile']-.1)/.3,(np.log(v['shrinkage'])-np.log(25))/np.log(16),v['quota_exponent'],v['uncertainty_weight']/.25] for v in params]),0.,1.);cache={};trials={}
 def objective(index):
  if index in cache:return cache[index]
  v=params[index];ref=fit_relative_references(d['y'][cal],labels[cal],d['site'][cal],d['xy'][cal],quantile=v['quantile'],shrinkage=v['shrinkage'])
  tau=reference_thresholds(ref,labels[dev],ood=ood[dev]);score=review_scores(mu[dev],var[dev],threshold=tau,uncertainty_weight=v['uncertainty_weight']);picked=select_review(ids[dev],score,labels[dev],budget,quota_exponent=v['quota_exponent']);selected=np.isin(np.arange(len(dev)),picked)
  def recall(groups):
   values=[]
   for g in np.unique(groups):
    ix=(groups==g)&harm[dev]
    if ix.sum()>=5:values.append(float(selected[ix].mean()))
   return float(np.mean(values)) if values else 0.
  road=recall(d['site'][dev]);env=recall(labels[dev]);normal_cost=float(normal[dev[picked]].mean());loss=.5*(1-road)+.3*(1-env)+.2*normal_cost
  cache[index]=loss;trials[index]=dict(parameters=v,loss=loss,macro_road_harm_recall=road,macro_cluster_harm_recall=env,selected_normal_fraction=normal_cost,reference=ref);return loss
 initial=[0,17,36,53];history=[];losses=[]
 for _ in range(16):
  i=next_expected_improvement(coords,history,losses,initial);history.append(i);losses.append(objective(i))
 random=initial+np.random.default_rng(261126).permutation([i for i in range(len(params)) if i not in initial]).tolist()[:12];randomloss=[objective(int(i)) for i in random]
 win=min(history,key=lambda i:(cache[i],-params[i]['shrinkage']));selected=trials[win];basebest=min(policy['bo_losses']);improved=bool(selected['loss']<=basebest-.01)
 receipt=dict(selected=selected,bo_history=history,bo_losses=losses,random_history=random,random_losses=randomloss,trials={str(k):v for k,v in trials.items()},
      common_reference_loss=basebest,relative_training_gate_passed=improved,training_gate='Add an equal-budget paired arm only if dev objective improves by >=0.01; all threshold tuning before any f1 evaluation.',
      absolute_threshold=.7,interpretation='Relative review reference, never a weather-dependent pass criterion. Unsupported or OOD clusters fall back to 0.7.')
 dump(S/'protocol'/a.output_name,receipt)
 if improved:
  v=selected['parameters'];tau=reference_thresholds(selected['reference'],labels[pool],ood=ood[pool]);score=review_scores(mu[pool],var[pool],threshold=tau,uncertainty_weight=v['uncertainty_weight']);indices=pool[select_review(ids[pool],score,labels[pool],3072,quota_exponent=v['quota_exponent'])]
  rows=[json.loads(s) for s in (S/'protocol/all_sources.jsonl').read_text().splitlines()];jsonl(S/'protocol/RelativeBO_sources.jsonl',[dict(rows[i],selection={'arm':'RelativeBO','rank':j+1,'environment_cluster':int(labels[i])}) for j,i in enumerate(indices)])
 print('RELATIVE BO',selected['loss'],'common',basebest,'add_training',improved,flush=True)
if __name__=='__main__':main()
