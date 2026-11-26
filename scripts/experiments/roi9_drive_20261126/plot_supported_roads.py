"""Plot the explicitly GT-supported conditional followup without replacing main results."""
from pathlib import Path
import argparse,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def export(fig,root,name):fig.savefig(root/(name+'.png'),dpi=160);fig.savefig(root/(name+'.svg'));plt.close(fig)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=read(a.evaluation/'results.json')
 with np.load(a.evaluation/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False});colors=['#2471a3','#dd8c27','#839192'];fig,axes=plt.subplots(1,3,figsize=(16,9))
 for j,ax in enumerate(axes):
  valid=np.isfinite(d['truth'][:,j])
  if j:valid&=d['near_gt' if j==1 else 'far_gt']>=50
  for color,road in zip(colors,np.unique(d['roads'])):
   ix=valid&(d['roads']==road);ax.scatter(d['truth'][ix,j],np.clip(d['raw_mean'][ix,j],0,1),s=18,c=color,alpha=.65,label=str(road))
  ax.plot([0,1],[0,1],'k--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Offline GT',ylabel='Raw GP point estimate',title=['Whole-frame IoU','Near-band recall','Far-band recall'][j]);ax.set_aspect('equal')
 axes[0].legend(fontsize=8);fig.suptitle('Fixed GP: GT-supported new-road followup',fontsize=20);fig.text(.03,.055,'Post-main-result cohort chosen only by annotation support at three fixed positions. Predictions/errors were not used to select routes.\nAll estimators and calibration unchanged. This conditional domain does not replace the original sparse-marking test.',fontsize=11);fig.subplots_adjust(bottom=.2,top=.83,wspace=.28);export(fig,a.output,'01_supported_regression')
 keys=['raw_composed','global_moments','selected_calibration'];available=[key for key in keys if key in r['calibration']]
 if not available:available=list(r['calibration'])
 names=[key.replace('_','\n') for key in available];x=np.arange(len(available));fig,axes=plt.subplots(2,2,figsize=(16,9))
 for j,target in enumerate(['near_recall','far_recall']):
  axes[0,0].bar(x+(j-.5)*.32,[r['calibration'][key]['marginals'][target]['coverage90'] for key in available],.32,label=target.replace('_',' '));axes[0,1].bar(x+(j-.5)*.32,[r['calibration'][key]['marginals'][target]['unbounded_width90'] for key in available],.32,label=target.replace('_',' '))
 axes[0,0].axhline(.9,color='#444',ls='--',label='Nominal90%');axes[0,0].set(xticks=x,xticklabels=names,ylabel='Empirical coverage',ylim=(0,1.05),title='Near/far interval coverage');axes[0,0].legend(fontsize=9);axes[0,1].set(xticks=x,xticklabels=names,ylabel='Unbounded interval width',title='Coverage must be read with width');axes[0,1].legend(fontsize=9)
 b=[r['calibration'][key]['spatial']['brier'] for key in available];axes[1,0].bar(x,b,color='#5b8fa8');axes[1,0].set(xticks=x,xticklabels=names,ylabel='Four-state Brier (sum over4 classes)',title='Joint spatial probabilities')
 for i,key in enumerate(available):axes[1,0].text(i,b[i],f"NLL {r['calibration'][key]['spatial']['nll']:.3f}",ha='center',va='bottom',fontsize=9)
 policies=[key for key in ['raw_GP_low_probability','calibrated_global','far_only','frozen_policy'] if key in r['review']];x=np.arange(len(policies))
 for j,event in enumerate(['global_low','near_weak','far_only']):axes[1,1].bar(x+(j-1)*.23,[r['review'][key]['counts'][event] for key in policies],.23,label=event.replace('_',' '))
 axes[1,1].set(xticks=x,xticklabels=[key.replace('_','\n') for key in policies],ylabel='Selected frames (events overlap)',title='Same candidates and per-road review budget');axes[1,1].legend(fontsize=8)
 fig.suptitle('Unchanged calibration on supported road geometry',fontsize=20);fig.text(.03,.015,'Image-band diagnostic targets; not operational failure labels. Same-map three-road followup; no refitting or automatic admission.',fontsize=10);fig.tight_layout(rect=(0,.05,1,.94));export(fig,a.output,'02_supported_calibration_review')
 print('supported-domain figures saved',flush=True)
if __name__=='__main__':main()
