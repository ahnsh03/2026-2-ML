"""Freeze simulator-tag and RGB-cluster policies before neural Dev evaluation."""
from pathlib import Path
import argparse,itertools,json
from collections import Counter
import numpy as np
from quality_estimation.environment_selection import fit_environment_reference,thresholds,residual_distribution,calibrated_priority,select_environment_review
from quality_estimation.bounded_bo import next_expected_improvement
from study_data import read,dump,sha,load_sources,memberships,labels,lighting,macro


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;P=a.parent
 data,schema,rows=load_sources(P);members=memberships(S,data['ids']);dev=members['development'];dev=dev[np.array([rows[i]['source_key'].startswith('drive_development_') for i in dev])];cal=members['calibration'];pool=np.flatnonzero(data['role']=='candidate_pool');assert len(pool)==7442
 q,harm,normal=labels(data);weather=data['weather'].astype(int);sim=weather*3+lighting(data['hour']);rgb=np.full(len(rows),-1,int)
 env=read(S/'environment/environment_selection_frozen.json');cluster=np.load(S/env['selected_cluster_artifact']);rgb[cluster['all_source_indices']]=cluster['assigned'];assert (rgb[np.r_[pool,dev,cal]]>=0).all()
 quality=read(S/'f0_quality/quality_selection_frozen.json');winner=quality['policy_predictor'];predictions={};models={};fraction=3072/len(pool);budget=int(np.ceil(len(dev)*fraction));primary=np.array([rows[i]['source_key'].startswith('drive_development_') for i in dev])
 for kind in ['GP','RF']:
  rec=quality['selected'][kind];z=np.load(S/rec['prediction']);assert z['ids'].tolist()==data['ids'].tolist();mu=z['mean'];scale=np.maximum(z['sd'],1e-3) if kind=='GP' else np.ones(len(rows));resid=residual_distribution(q[cal],mu[cal],scale[cal]);predictions[kind]=(mu,scale,resid)
  models[kind]=dict(record=rec,residuals=resid.tolist(),calibration_rows=len(cal),scale='posterior_sd' if kind=='GP' else 'unit_absolute_residual',probability='empirical signed calibration-residual CDF; no assumed RF epistemic variance')
 def metrics(chosen):
  selected=np.isin(dev,chosen)
  def recall(group,target):
   vals=[float(selected[(group==g)&target[dev]].mean()) for g in np.unique(group) if ((group==g)&target[dev]).sum()>=5]
   return float(np.mean(vals)) if vals else 0.
  rr=recall(data['site'][dev],harm);er=recall(sim[dev],harm);nc=float(normal[chosen].mean());low=np.isfinite(q)&(q<.7)
  return dict(loss=.45*(1-rr)+.35*(1-er)+.2*nc,macro_road_near_deficit_recall=rr,macro_sim_environment_near_deficit_recall=er,selected_near_retained_fraction=nc,macro_road_absolute_low_iou_recall=recall(data['site'][dev],low),selected_rows=len(chosen),primary_selected_rows=int(selected[primary].sum()))
 params=[dict(quantile=q_,shrinkage=s,balanced_floor=f,uncertainty_weight=u,retained_only=r) for q_,s,f,u,r in itertools.product([.1,.25,.4],[20.,100.],[0.,.25,.5],[0.,.1],[False,True])]
 coordinates=np.clip(np.array([[(v['quantile']-.1)/.3,np.log(v['shrinkage']/20)/np.log(5),v['balanced_floor']/.5,v['uncertainty_weight']/.1,float(v['retained_only'])] for v in params]),0.,1.);studies={};frozen={};indices={}
 for tag,groups in [('SimTagFull',sim),('RGBTagFull',rgb)]:
  mu,scale,resid=predictions[winner];cache={};trials={}
  def evaluate(j):
   if j in cache:return cache[j]
   v=params[j];eligible=normal[cal] if v['retained_only'] else None
   if eligible is not None and not eligible.any():eligible=None
   ref=fit_environment_reference(q[cal],groups[cal],data['site'][cal],data['xy'][cal],quantile=v['quantile'],shrinkage=v['shrinkage'],eligible=eligible)
   risk=calibrated_priority(mu[dev],resid,threshold=thresholds(ref,groups[dev]),scale=scale[dev],uncertainty_weight=v['uncertainty_weight']);results=[]
   for fraction_ in [.1,.25,fraction]:
    n=int(np.ceil(len(dev)*fraction_));chosen=dev[select_environment_review(data['ids'][dev],risk,groups[dev],n,balanced_floor=v['balanced_floor'])];results.append(dict(fraction=fraction_,**metrics(chosen)))
   result=metrics(dev[select_environment_review(data['ids'][dev],risk,groups[dev],budget,balanced_floor=v['balanced_floor'])]);cache[j]=float(np.mean([r['loss'] for r in results]));trials[j]=dict(parameters=v,reference=ref,metrics=result,multibudget_metrics=results,objective=cache[j]);return cache[j]
  initial=[0,len(params)//3,2*len(params)//3,len(params)-1];history=[];losses=[]
  for _ in range(16):
   j=next_expected_improvement(coordinates,history,losses,initial);history.append(j);losses.append(evaluate(j))
  random=initial+np.random.default_rng(271127).permutation([j for j in range(len(params)) if j not in initial]).tolist()[:12];random_losses=[evaluate(int(j)) for j in random]
  j=min(history,key=lambda j:(cache[j],params[j]['balanced_floor'],params[j]['uncertainty_weight']));frozen[tag]=trials[j];studies[tag]=dict(bo_history=history,bo_losses=losses,random_history=random,random_losses=random_losses,trials=trials,selected_index=j,selected=trials[j]);v=trials[j]['parameters'];risk=calibrated_priority(mu[pool],resid,threshold=thresholds(trials[j]['reference'],groups[pool]),scale=scale[pool],uncertainty_weight=v['uncertainty_weight']);indices[tag]=pool[select_environment_review(data['ids'][pool],risk,groups[pool],3072,balanced_floor=v['balanced_floor'])]
 for kind in ['GP','RF']:
  mu,scale,resid=predictions[kind];risk=calibrated_priority(mu[pool],resid,scale=scale[pool]);indices[kind+'GlobalFull']=pool[select_environment_review(data['ids'][pool],risk,sim[pool],3072)]
 rng=np.random.default_rng(271127);indices['TagBalancedFull']=pool[select_environment_review(data['ids'][pool],rng.random(len(pool)),sim[pool],3072,balanced_floor=1.,audit_fraction=0.)]
 # Preserve previously frozen Random bank rather than rerolling its seed.
 random_rows=[json.loads(x) for x in (S/'protocol/RandomFull_cached.jsonl').read_text().splitlines()];lookup={s:i for i,s in enumerate(data['ids'])};indices['RandomFull']=np.array([lookup[r['sample_id']] for r in random_rows])
 comparisons={}
 for kind in ['GP','RF']:
  mu,scale,resid=predictions[kind];risk=calibrated_priority(mu[dev],resid,scale=scale[dev]);ix=dev[select_environment_review(data['ids'][dev],risk,sim[dev],budget)];comparisons[kind+'GlobalFull']=metrics(ix)
 frozen_policy=dict(quality_predictor=winner,quality_models=models,groups={'sim':'verified simulator weather code times3 plus day/transition/night','rgb':env['selected_cluster_artifact'],'rgb_semantic_gate':env['semantic_cluster_gate_passes']},policies=frozen,search=studies,development_global_controls=comparisons,absolute_quality_benchmark=.7,selection_budget=3072,candidate_rows=len(pool),source_sha256=sha(__file__),development_rows=len(dev),development_objective='Equal average of10%,25%,41.3% fixed review budgets on three moving Dev roads; historical static cohort reported separately, not dominating policy fit.',scope='Cal-only relative references; dev-only BO surrogate retrieval. Six fixed-v6 neural trials evaluate downstream benefit. RGB representation is supervised then e-means, not unsupervised weather discovery.')
 dump(S/'protocol/selection_policy_frozen.json',frozen_policy);digest=sha(S/'protocol/selection_policy_frozen.json')
 cached={r['sample_id']:r for r in [json.loads(s) for s in (P/'protocol/cached_sources.jsonl').read_text().splitlines()]};summary={}
 for arm,ix in indices.items():
  assert len(ix)==3072 and len(set(ix))==3072;admitted=[]
  for rank,i in enumerate(ix):
   r=cached[data['ids'][i]];admitted.append(dict(r,selection=dict(arm=arm,rank=rank+1,sim_environment=int(sim[i]),rgb_cluster=int(rgb[i]),policy_sha256=digest),label_admission='Offline native Semantic verified; adaptation-only spatial membership'))
  if arm!='RandomFull':(S/f'protocol/{arm}_cached.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in admitted))
  summary[arm]=dict(rows=len(ix),condition_counts={str(k):int(v) for k,v in Counter(sim[ix]).items()},weather_counts={str(k):int(v) for k,v in Counter(weather[ix]).items()},lighting_counts={str(k):int(v) for k,v in Counter(lighting(data['hour'][ix])).items()},unique_5m_cells=len(np.unique(np.floor(data['xy'][ix]/5),axis=0)),absolute_low_iou_fraction=float((q[ix]<.7).mean()),near_deficit_fraction=float(harm[ix].mean()),near_retained_fraction=float(normal[ix].mean()))
 dump(S/'reports/admission_summary.json',summary)
 queue=read(S/'protocol/training_plan.json');names={x['name'] for x in queue['queue']}
 for arm in ['TagBalancedFull','RFGlobalFull','GPGlobalFull','SimTagFull','RGBTagFull']:
  name=arm+'_s271127'
  if name not in names:queue['queue'].append(dict(name=name,arm=arm,seed=271127,manifest=str(S/f'protocol/{arm}_cached.jsonl'),run=str(S/'runs'/name)))
 dump(S/'protocol/training_plan.json',queue);print('POLICY FROZEN',winner,{k:v['parameters'] for k,v in frozen.items()},flush=True)
if __name__=='__main__':main()
