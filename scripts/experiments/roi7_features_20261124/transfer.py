"""Score previously frozen RFs on all archived fixed-condition and switching sessions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree

from evaluate import HERE, PLAN, digest, dump, pack, paired, stats
from followup import FOLLOW


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    out=args.out
    folder=out/'reserve'
    protocol=json.loads((HERE/'transfer_protocol.json').read_text())
    # The recorded inputs must still be the ones fixed before reading these outcomes.
    for line in (folder/'frozen.sha256').read_text().splitlines():
        expected,path=line.split(maxsplit=1)
        assert digest(Path(path))==expected
    receipts=json.loads((folder/'receipt.json').read_text())['sources']
    roster=json.loads((args.data/'ml_roi10_20261123/afternoon/protocols/roster.json').read_text())
    locked=json.loads((out/'lock.json').read_text())
    extra=json.loads((out/'followup_lock.json').read_text())
    train=pack(out,PLAN['train'])
    cal=pack(out,PLAN['cal'])
    prior=json.loads((out/'receipt.json').read_text())['sources']
    seen={key:set() for key in ('pixel_sha256','sample_id')}
    for name in prior:
        with np.load(out/(name+'.npz'),allow_pickle=False) as p:
            for key in seen:
                seen[key].update(p[key])
    output={}
    indices={'base':list(range(96)), 'uncertainty':list(range(96))+[111,112,113]}
    for name in ('margin','uncertainty_and_shape'):
        indices[name]=list(range(96))+FOLLOW['arms'][name]
    for role in protocol['groups']:
        keys=[r['key'] for r in roster if r['role']==role]
        for k in keys:
            assert digest(folder/(k+'.npz'))==receipts[k]['pack_sha256']
        d=pack(folder,keys)
        n=len(d['y'])
        for key in seen:
            unique=set(d[key])
            assert len(unique)==n and not seen[key].intersection(unique)
            seen[key].update(unique)
        d['frame_id']=np.array([int(k.rsplit('_',1)[1]) for k in d['sample_id']])
        for k in keys:
            assert np.all(np.diff(d['frame_id'][d['session']==k])>0)
        row=dict(n=n,low=int((d['y']<.7).sum()),runs=len(keys),
                 within25=int((cKDTree(train['xy']).query(d['xy'])[0]<25).sum()),
                 models={},paired={})
        preds={}
        for arm in protocol['candidates']:
            ref=locked if arm in locked['models'] else extra
            cols=indices[arm]
            for seed in PLAN['seeds']:
                tag=f'{arm}_{seed}'
                path=out/'models'/(tag+'.joblib')
                assert digest(path)==ref['models'][arm]['seeds'][str(seed)]['model_sha256']
                model=joblib.load(path)
                p=model.predict(d['x'][:,cols])
                normal=cal['y']>=.7
                cutoff=float(np.quantile(model.predict(cal['x'][normal][:,cols]),.1,method='lower'))
                row['models'][tag]=stats(d,p,cutoff)
                preds[tag]=p
        for seed in PLAN['seeds']:
            for a,b in [('uncertainty','base'),('margin','base'),('uncertainty_and_shape','uncertainty')]:
                row['paired'][f'{a}_vs_{b}_{seed}']=paired(d,preds[f'{a}_{seed}'],preds[f'{b}_{seed}'])
        np.savez_compressed(folder/('predictions_'+role+'.npz'),sample_id=d['sample_id'],iou=d['y'],**preds)
        output[role]=row
    checks={}
    for role,row in output.items():
        checks[role]={}
        for seed in PLAN['seeds']:
            a,b=[row['models'][f'{x}_{seed}'] for x in ('uncertainty_and_shape','uncertainty')]
            delta=a['mae']-b['mae']
            global_delta=a['budget']['0.1']['hits']-b['budget']['0.1']['hits']
            window_delta=a['window']['hits']-b['window']['hits']
            checks[role][str(seed)]=dict(delta=delta,global_hit_delta=global_delta,window_hit_delta=window_delta,
                                          passed=bool(delta<=-.003 and global_delta>=0 and window_delta>=0))
    passed=all(r['passed'] for role in checks.values() for r in role.values())
    dump(folder/'transfer.json',dict(cohorts=output,checks=checks,passed=passed,
                                     selected='uncertainty_and_shape' if passed else 'uncertainty'))
    print('transfer',checks,flush=True)


if __name__=='__main__':
    main()
