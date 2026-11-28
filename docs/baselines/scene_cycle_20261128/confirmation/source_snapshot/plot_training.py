"""Plot verified completed training logs; training loss is not an adoption metric."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import sha256_file


def read(path):return json.loads(Path(path).read_text())


def smooth(values,window=25):
    values=np.asarray(values,float);prefix=np.r_[0.,np.cumsum(values)]
    left=np.maximum(0,np.arange(len(values))+1-window)
    return (prefix[1:]-prefix[left])/(np.arange(len(values))+1-left)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[];reference=None
    styles=[('Coverage2048','Coverage24k diagnostic','#9299A2','--'),('Risk2048','Risk18k','#528459','-'),('Random2048','Matched random18k','#3F7EA6','-')]
    for arm,label,color,linestyle in styles:
        root=a.study/'runs'/(arm+'_s281128');run=read(root/'run.json')
        if(run['status']!='complete'or(root/'_SUCCESS').read_text().strip()!=sha256_file(root/'run.json')
                or run['artifact_sha256']['training_steps.jsonl']!=sha256_file(root/'training_steps.jsonl')):
            raise ValueError('training trace is incomplete or changed')
        if reference is not None and run['parent_perception_identity']!=reference:raise ValueError('training parents differ')
        reference=run['parent_perception_identity'];history=[json.loads(line)for line in (root/'training_steps.jsonl').read_text().splitlines()]
        steps=np.asarray([r['step']for r in history])
        if steps[0]!=1 or steps[-1]!=run['steps']or(np.diff(steps)<=0).any():raise ValueError('training update log differs from endpoint')
        records.append((label,color,linestyle,run,history,root))
    a.output.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for label,color,linestyle,run,history,root in records:
        x=np.asarray([r['step']for r in history]);total=[r['losses']['loss']for r in history];distill=[r['losses']['replay_distillation']for r in history];lr=[r['learning_rate']*1e6 for r in history]
        for ax,values in zip(axes,[smooth(total),smooth(distill),lr]):ax.plot(x,values,label=label,color=color,linestyle=linestyle,linewidth=1.5);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.15)
    for ax,title,ylabel in zip(axes,['Total training objective','Replay-teacher distillation','Fixed learning-rate schedules'],['Trailing mean over up to25log points','Trailing mean over up to25log points','Learning rate (x 1e-6)']):ax.set_title(title);ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Verified training traces | common parent and full replay\nOne optimizer seed; Coverage24k is diagnostic, only Risk/Random18k are matched controls; no validation/adoption claim from these curves',fontsize=11)
    for ext in ['png','svg']:fig.savefig(a.output/('15_training_traces.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    document=dict(records=[dict(arm=run['arm'],steps=run['steps'],seed=run['seed'],new_rows=run['new_rows'],replay_rows=run['replay_rows'],
        parent_perception_identity=run['parent_perception_identity'],checkpoint_sha256=run['checkpoint_sha256'],run_sha256=sha256_file(root/'run.json'),training_steps_sha256=sha256_file(root/'training_steps.jsonl'),
        elapsed_seconds_in_run=read(root/'summary.json')['elapsed_seconds'])for _,_,_,run,_,root in records],
        files={path.name:sha256_file(path)for path in a.output.iterdir()if path.is_file()},source_sha256=sha256_file(Path(__file__)),
        scope='Optimizer-update axis avoids mixing the55minute simulator-collection pause into training progress. Logged wall elapsed is preserved separately. Equal training loss does not imply equal held-out perception quality.')
    (a.output/'training_trace_manifest.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n');print('VERIFIED TRAINING FIGURE READY',flush=True)


if __name__=='__main__':main()
