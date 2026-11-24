"""Publish compact metrics and a paired-error figure without distributing local frame data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate import HERE, PLAN, digest, dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    def load(name):
        return json.loads((args.out/(name+'.json')).read_text())
    lock, follow = load('lock'),load('followup_lock')
    result, second, audit = load('results'),load('followup_results'),load('audit')
    receipt=load('receipt')
    avgs={k:{key:v[key] for key in ('n','cv','dev')} for k,v in dict(lock['models'],**follow['models']).items()}
    regions={}
    for k in PLAN['cohorts']:
        row=result['cohorts'][k]
        models=dict(row['models'],**second['cohorts'][k])
        arms={}
        for arm in avgs:
            s=models[arm+'_23411']
            arms[arm]=dict(mae=s['mae'],mae_seeds=[models[f'{arm}_{seed}']['mae'] for seed in PLAN['seeds']],
                           delta_ci95=(s.get('paired') or row['paired'][arm])['ci95'],budget=s['budget'],
                           within=s['within'],window=s.get('window'),auroc=s['auroc'],boundary_mae=s['boundary_mae'],
                           calibrated_alarm=s['alarm'])
        regions[k]=dict(n=row['n'],low=row['low'],boundary=row['boundary'],under25=row['under25'],arms=arms,
                        gp_mae={a:models[a+'_gp']['mae'] for a in lock['gp']})
    dest=args.destination
    dest.mkdir(parents=True,exist_ok=True)
    summary=dict(commit=PLAN['commit'],roi_mask=receipt['schema']['regions_sha256'],
                 target='whole-frame white+yellow lane IoU',samples=sum(v['rows'] for v in receipt['sources'].values()),
                 decisions=result['decisions'],primary_selected=result['selected'],
                 followup={k:second[k] for k in ('candidate','dev_ok','seeds','passed')},
                 development=avgs,primary=PLAN['primary'],cohorts=regions,public_mae_parity=audit['public_mae_parity'],
                 prior_packs=audit['prior_packs'],archive_maps=dict(checked=sum(v['n'] for v in audit['archived_map_checks'].values()),
                 max_diff=max(v['max_abs_feature_diff'] for v in audit['archived_map_checks'].values())),
                 fingerprints={n:digest(args.out/(n+'.json')) for n in
                              ('lock','results','followup_lock','followup_results','audit','receipt')},
                 source_sha256={p.name:digest(p) for p in sorted(HERE.iterdir()) if p.is_file()})
    transfer=json.loads((args.out/'reserve/transfer.json').read_text())
    summary['transfer']=transfer
    summary['all_frames']=summary['samples']+sum(r['n'] for r in transfer['cohorts'].values())
    summary['final_candidate']=transfer['selected']
    summary['fingerprints']['reserve_transfer']=digest(args.out/'reserve/transfer.json')
    dump(dest/'summary.json',summary)
    winner=transfer['selected']
    model_path=args.out/'models'/f'{winner}_23411.joblib'
    cols=(lock['arms']['uncertainty'] if winner=='uncertainty' else
          lock['arms']['base']+receipt['columns'][96:99]+receipt['columns'][111:114])
    dump(dest/'candidate.json',dict(status='offline research candidate',reference_commit=PLAN['commit'],
         region_profile=receipt['schema']['profile'],region_mask_sha256=summary['roi_mask'],
         perception_checkpoint_sha256=receipt['checkpoint'],reference_seed=23411,
         estimator='RandomForestRegressor',model_file=model_path.name,model_sha256=digest(model_path),
         dimensions=len(cols),ordered_features=cols,target=summary['target']))
    panels=[('global_shape','Whole-frame omission cues (+3)'),('uncertainty','Uncertainty (+3)'),
            ('margin','Lane probability margin (+1)'), ('global_only','Whole-frame baseline (12 total)')]
    names=['18 static sites','40 spatial sites','12 later sites','Repeated drive','Separate road','Extra hours','Night challenge']
    keys=list(PLAN['cohorts'])
    fig,axes=plt.subplots(1,4,figsize=(13,4.4),sharey=True,sharex=True)
    for ax,(a,title) in zip(axes,panels):
        for i,k in enumerate(keys):
            r=regions[k]['arms']
            delta=r[a]['mae']-r['base']['mae']
            lo,hi=r[a]['delta_ci95']
            col='#00695c' if hi<0 else '#ad482f' if lo>0 else '#535c65'
            ax.plot([lo,hi],[i,i],color=col,lw=2)
            ax.scatter(delta,i,color=col,s=26,zorder=3)
        ax.axvline(0,color='#555',lw=.8,ls='--')
        ax.grid(axis='x',alpha=.15)
        ax.set_title(title,fontsize=10,fontweight='bold')
        ax.set_yticks(range(len(names)),names)
    axes[0].invert_yaxis()
    fig.suptitle('Change in lane-IoU prediction error versus the 96-feature ROI7 baseline',fontsize=12,fontweight='bold')
    fig.supxlabel('MAE difference on the same frames (lower is better). Lines show 95% grouped bootstrap intervals',fontsize=9)
    fig.tight_layout()
    fig.savefig(dest/'paired_mae.png',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
