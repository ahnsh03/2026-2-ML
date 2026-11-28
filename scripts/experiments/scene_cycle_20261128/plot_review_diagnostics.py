"""Audit GP uncertainty and supported error types using offline GT only."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import sha256_file
from quality_estimation.review_targets import support_aware_targets


def number(value):return float(value)if np.isfinite(value)else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['selection','pack','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);selection=json.loads((a.selection/'selection.json').read_text())
    with np.load(a.pack)as z:data={k:z[k]for k in z.files}
    with np.load(a.selection/'development_predictions.npz')as z:pred={k:z[k]for k in z.files}
    if data['ids'].tolist()!=pred['ids'].tolist():raise ValueError('diagnostic prediction/GT membership differs')
    t=support_aware_targets(data['counts']);report=[];plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,tag in zip(axes,['env52_errors_GP','review_support_errors_GP']):
        sd=pred[tag+'_sd'];mean=pred[tag+'_error_mean'];valid=t['near']['supported']&t['far']['supported']
        both=t['near']['error']&t['far']['error'];far_only=(~t['near']['error'])&t['far']['error']&(t['near']['gt_pixels']>=50)&(t['near']['predicted_pixels']>=50)
        labels=np.full(len(sd),3);labels[valid&~both&~far_only]=0;labels[valid&far_only]=1;labels[valid&both]=2
        positions=np.array_split(np.argsort(sd),4);values=[]
        for indices in positions:values.append([float(np.mean(both[indices])),float(np.mean(far_only[indices])),float(np.mean(t['near']['error'][indices]))])
        values=np.array(values);x=np.arange(4)
        for j,(name,color)in enumerate([('Near and far errors','#B24B45'),('Near retained, far error','#357BAC'),('Any near error','#628B53')]):ax.plot(x,values[:,j],marker='o',label=name,color=color)
        ax.set_xticks(x,['Lowest SD','Q2','Q3','Highest SD']);ax.set_ylim(0,1);ax.set_ylabel('Observed fraction in SD quartile');ax.set_title(tag.replace('_',' / '),fontsize=11);ax.grid(alpha=.15)
        def auc(y,score,use):return number(roc_auc_score(y[use],score[use]))if len(np.unique(y[use]))==2 else None
        contrast=mean[:,0]-mean[:,1];contrast_sd=pred[tag+'_near_minus_far_sd']
        negative_contrast=(contrast+1.645*contrast_sd<0)&valid
        report.append(dict(model=tag,rows=len(sd),supported_both_rows=int(valid.sum()),
            utility_sd_spearman=number(spearmanr(sd,t['review_utility']).statistic),
            sd_auc_near_and_far_errors=auc(both,sd,valid),sd_auc_near_retained_far_error=auc(far_only,sd,valid),
            error_mean_auc_near=auc(t['near']['error'],mean[:,0],np.ones(len(sd),bool)),
            negative_gaussian_contrast_rows=int(negative_contrast.sum()),
            negative_contrast_far_only_precision=float(far_only[negative_contrast].mean())if negative_contrast.any()else None,
            negative_contrast_far_only_recall=float((negative_contrast&far_only).sum()/far_only.sum())if far_only.any()else None,
            contrast_scope='Uncalibrated Gaussian score contrast excludes zero; not a calibrated90% categorical failure interval.',
            quartile_sd_means=[float(sd[i].mean())for i in positions],quartile_outcome_fractions=values.tolist(),
            interpretation='SD measures model uncertainty. No monotone physical failure/near-versus-far guarantee; no safety label from absent support.'))
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Does GP uncertainty separate failure types? New development segments only')
    for ext in ['png','svg']:fig.savefig(a.output/('03_uncertainty_failure_types.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    tags=list(selection['quality_scores']);mae=[selection['quality_scores'][k]['mae']for k in tags]
    axes[0].bar(np.arange(len(tags)),mae,color=['#357BAC'if k.endswith('GP')else'#B38145'for k in tags]);axes[0].set_xticks(np.arange(len(tags)),[k.replace('review_support','Support74').replace('env52','52').replace('_','\n')for k in tags]);axes[0].set_ylabel('Road-macro whole-frame IoU MAE');axes[0].set_title('Whole-frame quality regression')
    cases=[]
    for kind in ['GP','RF']:
        rs=[r for r in selection['candidates']if r['error_tag'].endswith(kind)and r['policy']['mode']=='supported_utility'and r['policy']['beta']==0 and r['policy']['information_fraction']==0]
        best=max(rs,key=lambda r:r['metrics']['utility_road_macro']);cases.append((kind,best['metrics']['utility_road_macro']))
    cases+=[('Random',selection['random100_mean']['utility_road_macro']),('Selected',selection['selected']['metrics']['utility_road_macro'])]
    axes[1].bar([n for n,_ in cases],[v for _,v in cases],color=['#357BAC','#B38145','#888E98','#528459']);axes[1].set_ylabel('Road-macro supported-error utility recall');axes[1].set_title('Same 10% review budget; Dev selection')
    for ext in ['png','svg']:fig.savefig(a.output/('04_review_model_development.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    document=dict(records=report,perception_identity=selection['perception_identity'],source_pack_sha256=sha256_file(a.pack),selection_sha256=sha256_file(a.selection/'selection.json'),
        source_sha256=sha256_file(Path(__file__)),scope='Offline diagnostic on opened development. Cannot substitute for frozen independent confirmation.')
    (a.output/'uncertainty_diagnostic.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
