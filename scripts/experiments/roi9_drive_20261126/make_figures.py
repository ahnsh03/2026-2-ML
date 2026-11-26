"""Presentation figures from frozen drive results; no fitting or policy tuning."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr

def read(path):return json.loads(path.read_text())
def savefig(fig,root,name):
 fig.savefig(root/(name+'.png'),dpi=160);fig.savefig(root/(name+'.svg'));plt.close(fig)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.study
 plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 selection=read(n/'feature_study/selection_frozen.json');records=[r for r in selection['results'] if r['kind']=='GP'];names=[r['view'].replace('_','\n') for r in records];mae=np.array([r['development_macro_mae'] for r in records]);x=np.arange(len(names))
 fig,ax=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[2,1]})
 ax[0].bar(x,mae[:,0],color=['#06346a' if r['view']=='prior_frozen28' else '#a6bdcf' for r in records]);ax[0].set_xticks(x,names,fontsize=9);ax[0].set_ylabel('Whole-frame lane IoU MAE');ax[0].set_title('Whole-image input selection: three development segments')
 d=selection['auxiliary_diagnostic'];by=list(d['byroad'].values());x=np.arange(len(by));ax[1].bar(x-.17,[v['primary'] for v in by],.34,label='Primary 28 inputs',color='#a6bdcf');ax[1].bar(x+.17,[v['auxiliary'] for v in by],.34,label='Far auxiliary 31 inputs',color='#ef9b46');ax[1].set_xticks(x,['Dev A','Dev B','Dev C']);ax[1].set_ylabel('Far-band recall MAE');ax[1].legend(fontsize=10);ax[1].set_title('Separate target, separate feature choice')
 fig.suptitle('9 fixed masks; keep 28 primary inputs + a separate far GP',fontsize=20);fig.text(.05,.04,'Old fixed 2,480 training frames. New development labels select inputs; calibration/test labels excluded.\nNear/far are image bands; targets use GT only offline. These are development results.',fontsize=11);fig.subplots_adjust(bottom=.22,top=.83,wspace=.35);savefig(fig,a.output,'01_feature_selection')
 if not (n/'prospective_test_v1/results.json').exists():return
 result=read(n/'prospective_test_v1/results.json')
 with np.load(n/'prospective_test_v1/predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 colors=np.array(['#2171b5','#dc4b50','#e9a23b','#7b4ba0']);valid=d['labels']>=0
 fig,axes=plt.subplots(1,3,figsize=(16,9))
 for j,ax in enumerate(axes):
  ix=np.isfinite(d['truth'][:,j]);
  if j:ix&=d['near_gt' if j==1 else 'far_gt']>=50
  ax.scatter(d['truth'][ix,j],np.clip(d['raw_mean'][ix,j],0,1),s=18,c=colors[np.maximum(d['labels'][ix],0)],alpha=.65);ax.plot([0,1],[0,1],'k--',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='GT (offline)',ylabel='Raw GP point estimate',title=['Whole-frame lane IoU','Near-band recall','Far-band recall'][j]);ax.set_aspect('equal')
 fig.suptitle('Prospective evaluation on three new moving segments',fontsize=20);fig.text(.05,.08,'Features, GP weights, calibration and review policy were frozen before all test drives.\n5 weather settings x 3 times per segment; distance-thinned candidates. Same-map evidence, not independent-map validation.',fontsize=12);fig.subplots_adjust(bottom=.2,top=.84,wspace=.28);savefig(fig,a.output,'02_prospective_regression')
 keys=['raw_composed','global_moments','selected_calibration'];names=['Raw composed GP','Global moment correction','Selected calibration'];metrics=result['calibration']
 fig,axes=plt.subplots(1,3,figsize=(16,9));x=np.arange(3)
 for ax,field,title in zip(axes,['brier','nll','coverage90'],['Four-state Brier (lower is better)','Four-state NLL (lower is better)','Far recall 90% interval coverage']):
  vals=[metrics[k]['spatial'][field] if field!='coverage90' else metrics[k]['marginals']['far_recall'][field] for k in keys];ax.bar(x,vals,color=['#a6bdcf','#5d8eae','#06346a']);ax.set_xticks(x,names,rotation=18,ha='right',fontsize=10);ax.set_title(title,fontsize=13)
  if field=='coverage90':ax.axhline(.9,color='#d45c4b',ls='--');ax.set_ylim(0,1)
  for j,v in enumerate(vals):ax.text(j,v+.01,f'{v:.3f}',ha='center',fontsize=11)
 fig.suptitle('Spatial uncertainty: calibrated on different segments',fontsize=20);fig.text(.05,.05,'Four mutually exclusive states: both good / near weak only / far weak only / both weak.\nReference events: near recall < 0.8, far recall < 0.5; each needs >=50 GT pixels offline. Empirical correction gives no OOD coverage guarantee.',fontsize=11);fig.subplots_adjust(bottom=.25,top=.83,wspace=.3);savefig(fig,a.output,'03_spatial_calibration')
 policy_keys=['low_GP_mean','low_RF_mean','calibrated_global_probability','frozen_policy'];labels=['Low GP mean','Low RF mean','Calibrated global','Frozen candidate'];review=result['review'];targets=['global_low','near_weak','far_only'];target_names=['Whole-IoU < 0.7','Near recall < 0.8','Far-only weak'];x=np.arange(4)
 fig,axes=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[2,1]})
 for j,(target,label) in enumerate(zip(targets,target_names)):
  vals=[review[k]['counts'][target] for k in policy_keys];axes[0].bar(x+(j-1)*.24,vals,.24,label=label,color=['#06346a','#dc4b50','#e9a23b'][j])
 axes[0].set_xticks(x,labels,fontsize=10);axes[0].set_ylabel('Selected frames (event sets can overlap)');axes[0].legend();axes[0].set_title('Same candidate pool and same per-segment budget')
 vals=[review[k]['utility'] for k in policy_keys];axes[1].bar(x,vals,color='#5d8eae');axes[1].axhline(review['random100']['mean_utility'],ls='--',color='#777',label='Random 100-seed mean');axes[1].set_xticks(x,labels,rotation=25,ha='right',fontsize=10);axes[1].legend(fontsize=10);axes[1].set_title('Frozen multi-objective utility')
 fig.suptitle('Review selection: prospective evidence, not retraining gain',fontsize=20);fig.text(.05,.05,f'Budget = {review[policy_keys[0]]["budget"]} frames, ceil(10% per segment). Utility and weights fixed using calibration OOF.\nFar-only weakness is a separate diagnostic event. Near-healthy/low-IoU is a proxy, not operational normality GT.',fontsize=11);fig.subplots_adjust(bottom=.23,top=.83,wspace=.3);savefig(fig,a.output,'04_review_budget')
 fig,axes=plt.subplots(1,2,figsize=(16,9));sigma=np.sqrt(d['raw_covariance'][:,0,0]);gap=d['raw_mean'][:,1]-d['raw_mean'][:,2]
 for label,name in enumerate(['Both good','Near weak only','Far weak only','Both weak']):
  ix=valid&(d['labels']==label);axes[0].scatter(sigma[ix],gap[ix],s=23,alpha=.65,label=f'{name} (n={ix.sum()})',color=colors[label])
 axes[0].set(xlabel='Raw whole-IoU posterior observation std',ylabel='Predicted near recall - far recall',title='A scalar uncertainty alone does not define spatial failure');axes[0].legend(fontsize=10)
 reliability=[]
 for label in range(4):
  prob=d['spatial'][valid,label];truth=d['labels'][valid]==label;xs=[];ys=[]
  for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
   ix=(prob>=lo)&(prob<=hi if hi==1 else prob<hi)
   if ix.sum()>=5:xs.append(float(prob[ix].mean()));ys.append(float(truth[ix].mean()))
  axes[1].plot(xs,ys,'o-',label=['Both good','Near only','Far only','Both weak'][label],color=colors[label]);reliability.append(dict(label=label,mean_predicted=xs,observed_fraction=ys))
 axes[1].plot([0,1],[0,1],'k--');axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed fraction',title='Four-state reliability (pooled descriptive bins)');axes[1].legend(fontsize=10)
 fig.suptitle('Use joint predictions and calibrated probabilities',fontsize=20);fig.text(.05,.05,'Masks are fixed. Online inputs are RGB + perception outputs; offline GT only checks these diagnostic event definitions.\nReliability bins pool correlated frames; three segments do not establish general calibration.',fontsize=11);fig.subplots_adjust(bottom=.2,top=.83,wspace=.3);savefig(fig,a.output,'05_uncertainty_and_reliability')
 (a.output/'reliability_bins.json').write_text(json.dumps(reliability,indent=2)+'\n')
 print('figures saved',a.output,flush=True)
if __name__=='__main__':main()
