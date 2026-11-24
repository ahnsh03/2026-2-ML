"""Export the frozen feature order and portable aggregate evidence."""
import argparse
from pathlib import Path

import numpy as np

from prepare import HERE, PLAN, sha, read, dump

ROOT=HERE.parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    ap.add_argument('destination',type=Path)
    args=ap.parse_args()
    dest=args.destination
    dest.mkdir(parents=True,exist_ok=True)
    p=args.out
    result=read(p/'results.json')
    pairs=read(p/'pairs_lock.json')
    registration=read(p/'lock.json')
    chosen=result['selected']
    schema=read(p/'features.json')['schema']
    base=schema[:96]
    columns={'96':base,'97':base+[schema[-1]],'99':base+schema[-3:],'102':schema,
             **{k:base+s for k,s in read(HERE/'minimality.json')['pairs'].items()}}
    if chosen in columns:
        arm=PLAN['candidates'].get(chosen,chosen)
        model=arm+'_23411'
        h=(registration['frozen_models'][model] if chosen in PLAN['candidates'] else
           pairs['results'][chosen]['seeds']['23411']['model_sha256'])
        dump(dest/'selected_features.json',dict(status='selected for offline quality estimation on archived KATRI front-camera data',
            reference_commit=PLAN['reference_commit'],selection=chosen,dimensions=len(columns[chosen]),
            ordered_features=columns[chosen],roi='roi7-front-v1',estimator='RandomForestRegressor',
            seed=23411,model_file=model+'.joblib',model_sha256=h,roi_mask_sha256=read(p/'features.json')['roi_sha'],
            f0_sha256=read(p/'features.json')['checkpoint'],target='whole-frame white+yellow lane IoU'))
    info={}
    for name,g in result['cohorts'].items():
        if name not in ('links','far_links','all_static','repeat'):continue
        v={k:r for k,r in g.items() if k not in ('models',)}
        models={}
        for arm in columns:
            measured=[g['models'][f'{arm}_{s}'] for s in PLAN['seeds']]
            models[arm]=dict(mean_macro=float(np.mean([r['macro'] for r in measured])),
                             per_seed_macro=[r['macro'] for r in measured],reference=measured[0])
        v['models']=models
        info[name]=v
    parity=result['data_counts_parity']
    summary=dict(selected=chosen,registered_rows=registration['total'],prior_frames=registration['prior_unique_frames'],
        static_telemetry=result['telemetry'],superiority=result['superiority'],spatial_harm=result['new_spatial_harm'],
        smaller=result['simplification_passed'],fallback_order=read(p/'fallback_lock.json')['order'],
        development={k:{m:r[m] for m in ('cv','dev')} for k,r in pairs['results'].items()},
        label_check=dict(max_abs=max(v['max_abs'] for v in parity.values()),
            rows_over_float_tolerance=sum(v['above_float_round'] for v in parity.values()),
            failure_class_changed=sum(v['failure_class_changed'] for v in parity.values())),
        cutoffs=result['calibration_cutoffs'],cohorts=info,
        evidence_hashes={file:sha(p/file) for file in ('lock.json','pairs_lock.json','fallback_lock.json','features.json','results.json','predictions.npz')})
    dump(dest/'summary.json',summary)


if __name__=='__main__':
    main()
