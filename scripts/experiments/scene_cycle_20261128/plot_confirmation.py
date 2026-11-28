"""Create shareable scientific figures from the already scored fixed confirmation."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import sha256_file
from quality_estimation.review_targets import support_aware_targets


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['freeze','review-assessment','neural-report','environment-report','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();freeze=read(a.freeze);review=read(a.review_assessment);neural=read(a.neural_report);environment=read(a.environment_report)
    if any(doc.get('confirmation_freeze_sha256',doc.get('source_freeze_sha256'))!=sha256_file(a.freeze)for doc in [review,neural,environment]):
        raise ValueError('figure inputs belong to different fixed confirmations')
    a.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    files=[]
    def save(fig,name):
        for ext in ['png','svg']:
            path=a.output/(name+'.'+ext);fig.savefig(path,dpi=200,bbox_inches='tight');files.append(path)
        plt.close(fig)
    scores=neural['scores'];sites=list(scores['r1']['marked_by_segment']);x=np.arange(len(sites));width=.24
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for i,(name,label,color)in enumerate([('v6','Original v6','#9299A2'),('r1','Parent r1','#3F7EA6'),('selected','Frozen candidate','#BA7447')]):
        axes[0].bar(x+(i-1)*width,[scores[name]['marked_by_segment'][s]for s in sites],width,label=label,color=color)
    axes[0].set_xticks(x,[s.replace('scene28_','')for s in sites]);axes[0].set_ylim(0,1);axes[0].set_ylabel('Mean whole-frame marking IoU, GT >= 50 px');axes[0].set_title('Three new marked segments');axes[0].legend(frameon=False,fontsize=9)
    heads=['lane','drivable','white','yellow','stopline'];xx=np.arange(len(heads))
    for i,(name,label,color)in enumerate([('r1','Parent r1','#3F7EA6'),('selected','Frozen candidate','#BA7447')]):
        gains=[scores[name]['all_frame_pooled'][h]-scores['v6']['all_frame_pooled'][h]if scores[name]['all_frame_pooled'][h]is not None and scores['v6']['all_frame_pooled'][h]is not None else np.nan for h in heads]
        axes[1].bar(xx+(i-.5)*.34,gains,.34,label=label,color=color)
    axes[1].axhline(0,color='#777777',linewidth=.8);axes[1].set_xticks(xx,heads);axes[1].set_ylabel('Pooled IoU change from original v6');axes[1].set_title('New confirmation, all frames and heads');axes[1].legend(frameon=False,fontsize=9)
    gain=neural['marked_gain_vs_parent'];ci=neural['paired_condition_bootstrap']['gain95_interval']
    fig.suptitle(f'Frozen perception comparison | candidate - parent = {gain:+.4f} (conditional block CI {ci[0]:+.4f}, {ci[1]:+.4f})\nSame map, new qualified segments; original-retention check is separate',fontsize=12)
    save(fig,'08_perception_fresh_confirmation')
    results=review['results'];names=['RF_only','GP_only','primary','parent_r1'];labels=['RF quality','GP quality','Primary pair','Parent pair']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors=['#B38145','#357BAC','#528459','#9299A2']
    axes[0].bar(labels,[results[n]['gt_supported_quality_segment_macro_mae']for n in names],color=colors)
    axes[0].set_ylabel('Segment-macro whole-frame IoU MAE');axes[0].set_title('Identical GT-supported population')
    axes[1].bar(labels,[results[n]['gt_supported_interval90_segment_macro_coverage']for n in names],color=colors);axes[1].axhline(.9,color='#555555',linestyle='--',label='Nominal 90%');axes[1].set_ylim(0,1.04);axes[1].set_ylabel('Segment-macro interval coverage');axes[1].set_title('Empirical residual intervals');axes[1].legend(frameon=False)
    fig.suptitle('Quality regression and coverage on frozen new confirmation\nParent pair uses its own perception target; GP/RF comparison uses the same frozen candidate',fontsize=12)
    save(fig,'09_quality_fresh_confirmation')
    names=['primary','RF_only','GP_only','GP_information_control','global_q07_control','parent_r1'];labels=['Primary','RF only','GP only','GP info','Q < 0.7','Parent']
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained');xx=np.arange(len(names));primary=results['primary'];random=primary['random100_mean']
    for ax,key,title in [(axes[0],'near_error_road_macro','Near-error segment-macro recall'),(axes[1],'utility_recall','Weighted supported-error recall')]:
        values=[results[n]['review'][key]for n in names];ax.bar(xx,[np.nan if v is None else v for v in values],color=['#528459','#B38145','#357BAC','#7285AD','#9299A2','#BDC4CE']);ax.set_xticks(xx,labels,rotation=15);ax.set_ylim(0,1);ax.set_ylabel('Recall at the fixed 10% budget');ax.set_title(title)
        if random[key]is not None:ax.axhline(random[key],color='#444444',linestyle='--',label='Random100 mean (candidate target)');ax.legend(frameon=False,fontsize=8)
    m=primary['review'];fig.suptitle(f"Fixed review policies | {m['selected']} selected / {m['rows']} frames, {m['near_error_positives']} supported near errors\nFN/FP support and policy acceptance are reported separately; zero event is not a safety label",fontsize=12)
    save(fig,'10_review_fresh_confirmation')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    env=environment['results']
    cases=[('primary','classifier','Direct RGB weather'),('primary','cluster_semantic','e-means cluster semantics'),('Lloyd_environment_control','cluster_semantic','Lloyd cluster semantics')]
    for ax,(name,kind,title)in zip(axes,cases):
        value=env[name][kind]['weather'];counts=np.array(value['confusion_matrix']);den=counts.sum(1,keepdims=True);norm=np.divide(counts,den,out=np.zeros_like(counts,dtype=float),where=den>0)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(6),[f'W{i}'for i in range(6)]);ax.set_yticks(range(6),[f'W{i}'for i in range(6)]);ax.set_xlabel('Predicted SDK weather preset');ax.set_ylabel('Collected SDK preset');ax.set_title(title+'\nBA '+f"{value['balanced_accuracy']:.3f}",fontsize=10)
        for i in range(6):
            for j in range(6):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white'if norm[i,j]>.6 else 'black',fontsize=8)
    fig.suptitle('RGB context and fixed cluster interpretation on new segments\nSemantic maps learned only on fit; clustering quality does not establish review benefit',fontsize=12)
    save(fig,'11_environment_fresh_confirmation')
    primary_spec=next(r for r in freeze['review_candidates']if r['name']=='primary');pack=Path(primary_spec['pack'])
    with np.load(pack)as z:data={k:z[k]for k in z.files}
    truth=support_aware_targets(data['counts'])['whole_frame_iou'];valid=np.isfinite(truth)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,name,label in zip(axes,['RF_only','GP_only'],['RF','GP']):
        with np.load(a.review_assessment.parent/(name+'_predictions.npz'))as z:
            if z['ids'].tolist()!=data['ids'].tolist():raise ValueError('quality scatter membership differs')
            mean=z['estimated_iou'];scale=z['quality_predictive_scale']
        points=ax.scatter(truth[valid],mean[valid],c=data['hour'][valid],s=12,alpha=.65,cmap='viridis',vmin=7,vmax=20);ax.plot([0,1],[0,1],'--',color='#555555');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_aspect('equal');ax.set_xlabel('True whole-frame marking IoU');ax.set_ylabel('Predicted whole-frame IoU');ax.set_title(label+' quality on identical images')
    fig.colorbar(points,ax=axes,label='Collected SDK hour',ticks=[7,13,20],shrink=.75)
    fig.suptitle('Frozen RF/GP predictions; includes all defined IoU labels\nNear FP/FN review uses separate support-aware targets',fontsize=12)
    save(fig,'12_quality_scatter_fresh_confirmation')
    report=dict(files={path.name:sha256_file(path)for path in files},source_sha256=sha256_file(Path(__file__)),
        inputs={str(path.name):sha256_file(path)for path in [a.freeze,a.review_assessment,a.neural_report,a.environment_report]},
        scope='Already scored fixed confirmation, no model selection or threshold tuning. SVG/PNG figures, keep original panel proportions.')
    (a.output/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CONFIRMATION FIGURES',len(files),flush=True)


if __name__=='__main__':main()
