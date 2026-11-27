"""Measure empirical-CDF saturation and compare two Dev-only tie recipes."""
from pathlib import Path
import argparse
import numpy as np
from quality_estimation.environment_selection import calibrated_priority,residual_distribution,select_environment_review
from study_data import read,dump,sha,load_sources,memberships,labels,lighting


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--quality-dir',required=True);a=p.parse_args();S=a.study;root=S/a.quality_dir;choice=read(root/'quality_selection_frozen.json');index=read(S/'protocol/quality_index_frozen.json')
 if a.quality_dir=='f0_quality':
  data,_,rows=load_sources(a.parent);members=memberships(S,data['ids']);pool=np.flatnonzero(data['role']=='candidate_pool')
 else:
  data={};rows=[];members={};offset=0
  for key in ['quality_fit','development','calibration']:
   with np.load(S/'g1_packs'/(key+'.npz')) as z:
    for name in z.files:data.setdefault(name,[]).append(z[name])
    members[key]=np.arange(offset,offset+len(z['ids']));offset+=len(z['ids'])
   rows+=index[key]['rows']
  data={name:np.concatenate(value) for name,value in data.items()};pool=np.array([],int)
 dev=members['development'];dev=dev[[rows[i]['source_key'].startswith('drive_development_') for i in dev]];q,harm,normal=labels(data);cal=members['calibration'];cal=cal[np.isfinite(q[cal])];sim=data['weather'].astype(int)*3+lighting(data['hour']);positive_support=bool(harm[dev].sum()>=20 and sum(harm[dev][data['site'][dev]==s].sum()>=5 for s in np.unique(data['site'][dev]))>=2);models={}
 for kind in ['GP','RF']:
  rec=choice['selected'][kind]
  with np.load(S/rec['prediction']) as z:
   if z['ids'].tolist()!=data['ids'].tolist():raise ValueError('prediction ordering differs')
   mean=z['mean'];scale=np.maximum(z['sd'],1e-3) if kind=='GP' else np.ones(len(mean))
  residuals=residual_distribution(q[cal],mean[cal],scale[cal]);score=calibrated_priority(mean[dev],residuals,scale=scale[dev]);margin=(.7-mean[dev])/scale[dev];trials={}
  for tie in ['identifier','standardized_margin']:
   budgets=[]
   for fraction in [.1,.25,3072/7442]:
    picked=dev[select_environment_review(data['ids'][dev],score,sim[dev],int(np.ceil(len(dev)*fraction)),secondary_scores=margin if tie=='standardized_margin' else None)];selected=np.isin(dev,picked)
    def recall(groups,target):
     values=[float(selected[(groups==g)&target[dev]].mean()) for g in np.unique(groups) if ((groups==g)&target[dev]).sum()>=5]
     return float(np.mean(values)) if values else 0.
    road=recall(data['site'][dev],harm);environment=recall(sim[dev],harm);cost=float(normal[picked].mean());budgets.append(dict(fraction=fraction,rows=len(picked),loss=.45*(1-road)+.35*(1-environment)+.2*cost,near_deficit_road_recall=road,near_deficit_environment_recall=environment,near_retained_selected_fraction=cost,precision=float((q[picked]<.7).mean()),absolute_low_iou_road_recall=recall(data['site'][dev],np.isfinite(q)&(q<.7))))
   trials[tie]=dict(loss=float(np.mean([r['loss'] for r in budgets])),budgets=budgets)
  adopt=positive_support and trials['standardized_margin']['loss']<=trials['identifier']['loss']-.005;diagnostic={}
  for name,ix in [('primary_development',dev),('candidate_pool',pool)]:
   if not len(ix):continue
   p_=calibrated_priority(mean[ix],residuals,scale=scale[ix]);n=min(len(ix),int(np.ceil(len(ix)*3072/7442)));n-=int(np.floor(n*.1));order=np.argsort(-p_);boundary=p_[order[max(n-1,0)]];diagnostic[name]=dict(rows=len(ix),unique_scores=len(np.unique(p_)),fraction_at_max=float((p_==p_.max()).mean()),maximum_score=float(p_.max()),priority_budget=n,boundary_score=float(boundary),boundary_ties=int((p_==boundary).sum()),strictly_above_boundary=int((p_>boundary).sum()))
  models[kind]=dict(selected_tie_break='standardized_margin' if adopt else 'identifier',trials=trials,saturation=diagnostic,model_sha256=rec['model_sha256'])
 result=dict(models=models,selected_model=choice['policy_predictor'],runtime_tie_break=models[choice['policy_predictor']]['selected_tie_break'],positive_support=positive_support,quality_selection_sha256=sha(root/'quality_selection_frozen.json'),source_sha256=sha(__file__),rule='Compare only identifier and standardized (threshold-mean)/scale within exactly equal primary scores. Require Dev multibudget objective improvement>=0.005 and positive support; no change to primary probabilities, fitted means, or any already-frozen admission bank.',scope='Cal/Dev-only diagnostic and g1 policy option. f0 experiment is retrospective; its existing banks remain unchanged. Mean-retrieval proxy is not operational night normality, and this is not a new downstream neural comparison.')
 dump(root/'review_tie_study.json',result);print('REVIEW TIES',a.quality_dir,{k:v['selected_tie_break'] for k,v in models.items()},flush=True)

if __name__=='__main__':main()
