"""Descriptive budget/temporal/spatial trade-offs of frozen test predictions.

No model, threshold, quota or production policy is selected from these results.
"""
from pathlib import Path
import argparse,json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from feature_study import read,save,sha
from review_policy import evaluate,labels

def select_quota(data,global_score,far_score,budget,fraction=.7):
 selected=[]
 for road in np.unique(data['roads']):
  ix=np.flatnonzero(data['roads']==road);B=int(np.ceil(len(ix)*budget));first=int(np.ceil(B*fraction))
  chosen=ix[np.lexsort((data['ids'][ix],-global_score[ix]))[:first]];remaining=np.setdiff1d(ix,chosen)
  rest=remaining[np.lexsort((data['ids'][remaining],-far_score[remaining]))[:B-len(chosen)]];selected.extend(np.r_[chosen,rest].tolist())
 # Represent the fixed selected set as a score solely to reuse the same evaluator.
 score=np.zeros(len(data['ids']));score[selected]=1.;return score
def totals(result,data):
 targets=labels(data);return dict(budget=result['budget'],actual_budget_fraction=result['actual_budget_fraction'],counts=result['counts'],recall={name:result['counts'][name]/int(mask.sum()) if mask.any() else None for name,mask in targets.items()})
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--predictions',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.study;predictions=a.predictions or n/'prospective_test_v1/predictions.npz'
 with np.load(predictions,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 data=dict(truth=d['truth'],counts=d['counts'],near_gt=d['near_gt'],far_gt=d['far_gt'],roads=d['roads'],ids=d['ids'])
 sigma=np.sqrt(d['raw_covariance'][:,0,0]);scores=dict(frozen_policy=d['priority'],calibrated_global=d['global_probability'],low_GP_mean=-d['raw_mean'][:,0],low_RF_mean=-d['rf_mean'][:,0],high_GP_observation_std=sigma,far_only_diagnostic=d['spatial'][:,2]);results={}
 for budget in [.05,.1,.2]:
  candidates=dict(scores)
  for fraction in [.5,.7,.9]:candidates['split_global_far_'+str(fraction)]=select_quota(data,d['global_probability'],d['spatial'][:,2],budget,fraction)
  records={}
  for name,score in candidates.items():
   result=evaluate(score,data,budget);out=totals(result,data);selected=np.isin(d['ids'],result['selected_ids']);out['whole_iou_event_sensitivity']={str(t):dict(total=int((d['truth'][:,0]<t).sum()),selected=int(((d['truth'][:,0]<t)&selected).sum())) for t in [.65,.7,.75]};records[name]=out
  results[str(budget)]=records
 temporal=dict(data,roads=np.array([str(sid).rsplit('_f',1)[0] for sid in d['ids']]))
 temporal_records={name:totals(evaluate(score,temporal,.1),temporal) for name,score in scores.items()}
 valid=np.isin(d['labels'],[2,3]);auc=float(roc_auc_score(d['labels'][valid]==3,sigma[valid])) if len(set(d['labels'][valid]))==2 else None
 class_summary={str(label):dict(rows=int((d['labels']==label).sum()),median_whole_iou_std=float(np.median(sigma[d['labels']==label])) if (d['labels']==label).any() else None) for label in [-1,0,1,2,3]}
 corr=spearmanr(sigma,d['truth'][:,0],nan_policy='omit');correlation=float(corr.statistic) if np.isfinite(corr.statistic) else None
 environment={str(int(group)):dict(rows=int((d['environment']==group).sum()),weather_counts={str(int(w)):int(((d['environment']==group)&(d['weather']==w)).sum()) for w in np.unique(d['weather'])},hour_counts={str(int(h)):int(((d['environment']==group)&(d['hour']==h)).sum()) for h in np.unique(d['hour'])}) for group in np.unique(d['environment'])}
 save(a.output/'results.json',dict(predictions_sha256=sha(predictions),budget_comparisons=results,temporal_short_session=temporal_records,sigma_diagnostic=dict(by_spatial_class=class_summary,both_weak_vs_far_only_AUC=auc,Spearman_with_whole_iou=correlation),environment_groups=environment,scope=__doc__+' Short sessions have ceil-budget inflation. Near-healthy low-IoU is a proxy, not a normal/failure label. Spatial class-1 claims require actual class-1 support.'))
 print('descriptive sensitivity complete',flush=True)
if __name__=='__main__':main()
