"""Refit f1 relative review references on Cal and search Dev before fresh opening."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
from scipy.spatial.distance import cdist
from quality_estimation.portable_forest import PortableForest
from quality_estimation.environment_selection import fit_environment_reference,thresholds,residual_distribution,calibrated_priority,select_environment_review
from quality_estimation.bounded_bo import next_expected_improvement
from study_data import read,dump,sha,load_sources,lighting,labels

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;index=read(S/'protocol/quality_index_frozen.json');data={};parts={};offset=0
 for key in ['quality_fit','development','calibration']:
  z=np.load(S/'g1_packs'/(key+'.npz'));parts[key]=np.arange(offset,offset+len(z['ids']));offset+=len(z['ids'])
  for name in z.files:data.setdefault(name,[]).append(z[name])
 data={k:np.concatenate(v) for k,v in data.items()};q,harm,normal=labels(data);dev=parts['development'];dev=dev[np.array([r['source_key'].startswith('drive_development_') for r in index['development']['rows']])];cal=parts['calibration'];cal=cal[np.isfinite(q[cal])]
 quality=read(S/'g1_quality/quality_selection_frozen.json');kind=quality['policy_predictor'];rec=quality['selected'][kind];pred=np.load(S/rec['prediction']);assert pred['ids'].tolist()==data['ids'].tolist();mean=pred['mean'];scale=np.maximum(pred['sd'],1e-3) if kind=='GP' else np.ones(len(mean));resid=residual_distribution(q[cal],mean[cal],scale[cal]);env=read(S/'environment/environment_selection_frozen.json');classifier=env['selected_classifier'];z=np.load(S/env['selected_cluster_artifact']);hz=[np.load(S/n) for n in ['environment_features.npz','environment_promoted_features.npz']];hids=np.concatenate([h['ids'] for h in hz]);hx=np.concatenate([h['X'] for h in hz]);hlookup={s:i for i,s in enumerate(hids)};columns=hz[0]['columns'].tolist();x=hx[[hlookup[s] for s in data['ids']]][:,[columns.index(n) for n in classifier['columns']]];wp=PortableForest.load(S/'environment'/f"{classifier['view']}_weather.npz").predict_proba(x);lp=PortableForest.load(S/'environment'/f"{classifier['view']}_lighting.npz").predict_proba(x);embedding=np.sqrt(np.maximum(np.c_[wp,lp],0))[:,z['active']];dist=cdist(embedding,z['centers'],'sqeuclidean');cluster=dist.argmin(1);rgb_tag=wp.argmax(1)*3+lp.argmax(1);sim_tag=data['weather'].astype(int)*3+lighting(data['hour']);confidence=(wp.max(1)>=.6)&(lp.max(1)>=.6)
 def metrics(picked):
  chosen=np.isin(dev,picked);rr=[];ee=[]
  for group,dest in [(data['site'][dev],rr),(sim_tag[dev],ee)]:
   for g in np.unique(group):
    valid=(group==g)&harm[dev]
    if valid.sum()>=5:dest.append(float(chosen[valid].mean()))
  road=float(np.mean(rr)) if rr else 0.;environment=float(np.mean(ee)) if ee else 0.;cost=float(normal[picked].mean());return dict(loss=.45*(1-road)+.35*(1-environment)+.2*cost,road_near_deficit_recall=road,environment_near_deficit_recall=environment,near_retained_selected_fraction=cost)
 fractions=[.1,.25,3072/7442]
 def score_policy(groups,ref,params):
  tau=thresholds(ref,groups[dev],unsupported=~confidence[dev]) if ref else .7;priority=calibrated_priority(mean[dev],resid,threshold=tau,scale=scale[dev],uncertainty_weight=params.get('uncertainty_weight',0.));result=[]
  for f in fractions:
   ix=dev[select_environment_review(data['ids'][dev],priority,groups[dev],int(np.ceil(len(dev)*f)),balanced_floor=params.get('balanced_floor',0.))];result.append(dict(fraction=f,**metrics(ix)))
  return float(np.mean([r['loss'] for r in result])),result
 common,common_metrics=score_policy(np.zeros(len(mean),int),None,{})
 parameters=[dict(quantile=qq,shrinkage=s,balanced_floor=f,uncertainty_weight=u,retained_only=r) for qq,s,f,u,r in itertools.product([.1,.25,.4],[20.,100.],[0.,.25,.5],[0.,.1],[False,True])];coords=np.clip(np.array([[(v['quantile']-.1)/.3,np.log(v['shrinkage']/20)/np.log(5),v['balanced_floor']/.5,v['uncertainty_weight']/.1,float(v['retained_only'])] for v in parameters]),0.,1.);studies={}
 for name,groups in [('sim_oracle',sim_tag),('rgb_tag',rgb_tag),('rgb_cluster',cluster)]:
  trials={}
  def evaluate(j):
   if j in trials:return trials[j]['loss']
   v=parameters[j];eligible=normal[cal] if v['retained_only'] else None
   if eligible is not None and not eligible.any():eligible=None
   reference=fit_environment_reference(q[cal],groups[cal],data['site'][cal],data['xy'][cal],quantile=v['quantile'],shrinkage=v['shrinkage'],eligible=eligible);loss,detail=score_policy(groups,reference,v);trials[j]=dict(parameters=v,reference=reference,loss=loss,metrics=detail);return loss
  initial=[0,len(parameters)//3,2*len(parameters)//3,len(parameters)-1];history=[];losses=[]
  for _ in range(16):
   j=next_expected_improvement(coords,history,losses,initial);history.append(j);losses.append(evaluate(j))
  random=initial+np.random.default_rng(271127).permutation([i for i in range(len(parameters)) if i not in initial]).tolist()[:12];random_losses=[evaluate(int(j)) for j in random];j=min(history,key=lambda j:(trials[j]['loss'],parameters[j]['balanced_floor'],parameters[j]['uncertainty_weight']));studies[name]=dict(selected=trials[j],bo_indices=history,bo_losses=losses,random_indices=random,random_losses=random_losses,trials=trials)
 positive_support=bool(harm[dev].sum()>=20 and sum(harm[dev][data['site'][dev]==site].sum()>=5 for site in np.unique(data['site'][dev]))>=2)
 eligible=[(name,v['selected']) for name,v in studies.items() if positive_support and name!='sim_oracle' and v['selected']['loss']<=common-.01];winner=min(eligible,key=lambda nv:nv[1]['loss']) if eligible else None
 if winner:
  name,v=winner;runtime=dict(grouping=name,reference=v['reference'],balanced_floor=v['parameters']['balanced_floor'],uncertainty_weight=v['parameters']['uncertainty_weight'],audit_fraction=.1,environment_thresholds_adopted=True,decision='Version-specific relative policy improved mean Dev retrieval proxy by>=0.01; frozen before fresh confirmation.')
 else:runtime=dict(grouping='global',reference={'absolute_threshold':.7,'groups':{}},balanced_floor=0.,uncertainty_weight=0.,audit_fraction=.1,environment_thresholds_adopted=False,decision='No RGB-only group policy improved mean Dev retrieval proxy by>=0.01; retain common calibrated probability plus10%audit.')
 dump(S/'g1_quality/review_policy_frozen.json',dict(quality_model_kind=kind,perception_identity=quality['perception_identity'],quality_selection_sha256=sha(S/'g1_quality/quality_selection_frozen.json'),calibration_rows=len(cal),primary_development_rows=len(dev),supported_near_deficit_rows=int(harm[dev].sum()),positive_support_passes=positive_support,common_objective=common,common_metrics=common_metrics,studies=studies,runtime_policy=runtime,source_sha256=sha(__file__),scope='f1 labels and features only. Cal reference/empirical residual distribution; Dev BO budget16 plus matched random16. Simulator oracle is analysis only, runtime groups use RGB. No f2 admission or further perception training on this policy yet.'))
 print('REFIT REVIEW POLICY',runtime['grouping'],common,[(n,v['selected']['loss']) for n,v in studies.items()],flush=True)
if __name__=='__main__':main()
