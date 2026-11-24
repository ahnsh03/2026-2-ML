"""Fit the two remaining uncertainty pairs with the original model recipe and dev split."""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from prepare import HERE, PLAN, sha, read, dump
sys.path.insert(0,str(HERE.parent/'roi7_features_20261124'))
from evaluate import PLAN as OLD, pack, rf, mae


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('models',type=Path)
    ap.add_argument('out',type=Path)
    a=ap.parse_args()
    spec=read(HERE/'minimality.json')
    train,dev=[pack(a.models,OLD[key]) for key in ('train','dev')]
    prior=read(a.models/'lock.json')['models']
    orig97=read(a.models/'followup_lock.json')['models']['margin']
    results={'97':orig97,'99':prior['uncertainty']}
    for name,fields in spec['pairs'].items():
        indices=list(range(96))+[train['names'].index(s) for s in fields]
        tx=train['x'][:,indices]
        dx=dev['x'][:,indices]
        rec={}
        for seed in PLAN['seeds']:
            cv=[]
            for i in range(3):
                held=np.isin(train['session'],OLD['train'][i::3])
                fitted=rf(seed).fit(tx[~held],train['y'][~held].astype('float32'))
                cv.append(mae(train['y'][held],fitted.predict(tx[held])))
            fitted=rf(seed).fit(tx,train['y'].astype('float32'))
            file=a.out/f'{name}_{seed}.joblib'
            joblib.dump(fitted,file,compress=1)
            rec[str(seed)]=dict(cv=float(np.mean(cv)),dev=mae(dev['y'],fitted.predict(dx)),model_sha256=sha(file))
        results[name]=dict(n=len(indices),seeds=rec,**{k:float(np.mean([s[k] for s in rec.values()])) for k in ('cv','dev')})
    eligible=[k for k,v in results.items() if k!='99' and all(v[m]<=results['99'][m]+.001 for m in ('cv','dev'))]
    nominee=min(eligible,key=lambda k:(results[k]['n'],results[k]['dev'])) if eligible else None
    dump(a.out/'pairs_lock.json',dict(protocol=sha(HERE/'minimality.json'),main_lock=sha(a.out/'lock.json'),
         nominee=nominee,results=results))
    fallback=[key for key in eligible if key.startswith('98')]
    order=[nominee] if nominee else []
    if fallback:
        order=list(dict.fromkeys(order+[min(fallback,key=lambda k:results[k]['dev'])]))
    dump(a.out/'fallback_lock.json',dict(order=order,pair_lock=sha(a.out/'pairs_lock.json'),
                                        plan=sha(HERE/'size_fallback.json')))
    print(nominee,[(k,s['cv'],s['dev']) for k,s in results.items()],flush=True)


if __name__=='__main__':
    main()
