"""Plot measured equal-budget search traces without changing model selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads((a.models/'fit_complete.json').read_text());a.output.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(2,4,figsize=(16,8));fig.subplots_adjust(left=.05,right=.99,bottom=.08,top=.86,hspace=.4,wspace=.3)
    colors=dict(random='#606776',fixed_ei='#D48428',botorch_logei='#236CA8')
    labels=dict(random='Random',fixed_ei='Fixed Matern EI',botorch_logei='BoTorch LogEI')
    effects=[]
    for ax,(tag,search)in zip(axes.flat,data['search'].items()):
        histories=search['histories'];b=search['query_budget_per_method_per_seed']
        for method in colors:
            values=np.array([h['incumbent']for h in histories if h['method']==method])
            if not len(values):continue
            mean=values.mean(0);sd=values.std(0);x=np.arange(1,b+1)
            ax.plot(x,mean,label=labels[method],color=colors[method],linewidth=2)
            ax.fill_between(x,mean-sd,mean+sd,color=colors[method],alpha=.10)
        ax.axvline(4,color='#BFC2C8',linestyle=':',linewidth=1)
        ax.set_title(tag.replace('review_support','Support74').replace('env52','RGB+output52').replace('_',' / '),fontsize=10)
        ax.set_xlabel('CV queries');ax.set_ylabel('Best CV loss (lower is better)');ax.set_xticks([1,4,b]);ax.grid(alpha=.12)
        by={(h['method'],h['seed']):h for h in histories};seeds=search['search_seeds']
        effects.append((tag,[by[('random',s)]['incumbent'][-1]-by[('botorch_logei',s)]['incumbent'][-1]for s in seeds]))
    handles,legend=axes.flat[0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.suptitle('BO versus random: same initial four queries, eight-query budget, three seeds',fontsize=14,y=.99)
    for ext in ['png','svg']:fig.savefig(a.output/('01_equal_budget_bo.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig);fig,ax=plt.subplots(figsize=(11,5),layout='constrained');means=np.array([np.mean(v)for _,v in effects]);x=np.arange(len(effects))
    ax.bar(x,means,color='#236CA8');ax.axhline(.001,color='#B2493D',linestyle='--',label='Required mean gain: 0.001');ax.axhline(0,color='#606776',linewidth=.8)
    for i,(_,values)in enumerate(effects):ax.scatter(np.full(len(values),i),values,color='#273346',s=20,zorder=3)
    ax.set_xticks(x,[t.replace('review_support','Support74').replace('env52','52').replace('_','\n')for t,_ in effects]);ax.set_ylabel('Final random loss minus BoTorch loss');ax.set_title('Measured BO benefit does not reach the preregistered adoption gate');ax.legend(frameon=False)
    for ext in ['png','svg']:fig.savefig(a.output/('02_bo_adoption_gate.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
