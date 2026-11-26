"""Plot fixed v2 calibration on the fresh confirmation, without fitting."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    r=json.loads((a.evaluation/'results.json').read_text())
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    keys=['raw_composed','previous_v1','selected_calibration']
    names=['Raw composed GP','Previous v1','Fixed v2']
    targets=['IoU','near_recall','far_recall'];target_names=['Whole IoU','Near recall','Far recall']
    colors=['#143f6b','#e2a543','#4c9b91'];x=np.arange(3)
    fig,axes=plt.subplots(2,2,figsize=(16,9))
    models=['GP28_primary','GP28_plus_far31','matched_RF']
    for i,key in enumerate(models):
        axes[0,0].bar(x+(i-1)*.24,r['regression'][key]['macro_mae'],.24,label=key,color=colors[i])
    axes[0,0].set(xticks=x,xticklabels=target_names,ylabel='Road-macro MAE',title='Point estimates: fixed training membership')
    axes[0,0].legend(fontsize=9)
    for i,key in enumerate(keys):
        m=r['calibration'][key]['marginals']
        axes[0,1].bar(x+(i-1)*.24,[m[t]['coverage90'] for t in targets],.24,label=names[i],color=colors[i])
        axes[1,0].bar(x+(i-1)*.24,[m[t]['unbounded_width90'] for t in targets],.24,label=names[i],color=colors[i])
    axes[0,1].axhline(.9,ls='--',color='#777');axes[0,1].set(xticks=x,xticklabels=target_names,ylim=(0,1.05),ylabel='Observed 90% interval coverage',title='Coverage on new road segments');axes[0,1].legend(fontsize=9)
    axes[1,0].set(xticks=x,xticklabels=target_names,ylabel='Unbounded 90% interval width',title='Read coverage together with interval width')
    values=[r['calibration'][k]['spatial']['brier'] for k in keys]
    counts=r['calibration']['selected_calibration']['spatial']['class_counts']
    axes[1,1].bar(x,values,color=colors)
    for i,k in enumerate(keys):
        axes[1,1].text(i,values[i],f"NLL {r['calibration'][k]['spatial']['nll']:.3f}",ha='center',va='bottom',fontsize=10)
    axes[1,1].set(xticks=x,xticklabels=names,ylabel='Four-state Brier (sum over classes)',title='Joint near/far probabilities: lower is better')
    counts=r['calibration']['selected_calibration']['spatial']['class_counts']
    fig.suptitle(f"Fresh confirmation: {r['rows']} thinned frames / {r['roads']} road segments",fontsize=20)
    fig.text(.04,.015,f'Prior supported cohort was reassigned to calibration. All v2 parameters frozen before these drives.\nClass counts [both good, near only, far only, both weak] = {counts}. Empirical same-map evidence; no OOD guarantee.',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.94))
    for suffix in ['png','svg']:fig.savefig(a.output/('01_fresh_confirmation.'+suffix),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
