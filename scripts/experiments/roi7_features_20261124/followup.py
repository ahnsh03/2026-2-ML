"""Resolve the additional uncertainty features using development data only."""
import argparse
import json

import joblib
import numpy as np

from evaluate import HERE, PLAN, Path, digest, dump, mae, pack, paired, rf, stats

FOLLOW=json.loads((HERE/'followup_protocol.json').read_text())


def matrix(d,arm):
    return d['x'][:,list(range(96))+FOLLOW['arms'][arm]]


def develop(out):
    train=pack(out, PLAN['train'])
    dev=pack(out, PLAN['dev'])
    report={}
    for arm,cols in FOLLOW['arms'].items():
        x=matrix(train,arm)
        vals=matrix(dev,arm)
        per_seed={}
        for seed in PLAN['seeds']:
            folds=[]
            for i in range(3):
                hold=np.isin(train['session'],PLAN['train'][i::3])
                model=rf(seed).fit(x[~hold],train['y'][~hold].astype(np.float32))
                folds.append(mae(train['y'][hold],model.predict(x[hold])))
            model=rf(seed).fit(x,train['y'].astype(np.float32))
            path=out/'models'/f'{arm}_{seed}.joblib'
            joblib.dump(model,path,compress=1)
            per_seed[str(seed)]=dict(folds=folds,cv=float(np.mean(folds)),dev=mae(dev['y'],model.predict(vals)),
                                     model_sha256=digest(path))
        report[arm]=dict(n=96+len(cols),seeds=per_seed,
                         **{k:float(np.mean([r[k] for r in per_seed.values()])) for k in ('cv','dev')})
        print(arm,report[arm]['dev'],flush=True)
    primary=json.loads((out/'lock.json').read_text())
    comparison=dict(report,uncertainty=primary['models']['uncertainty'])
    best=min(row['dev'] for row in comparison.values())
    eligible=[k for k,v in comparison.items() if v['dev']<=best+.001]
    candidate=min(eligible,key=lambda k:(comparison[k]['n'],comparison[k]['dev']))
    dump(out/'followup_lock.json',dict(candidate=candidate,models=report,
         protocol_sha256=digest(HERE/'followup_protocol.json'),first_lock_sha256=digest(out/'lock.json')))


def diagnose(out):
    lock=json.loads((out/'followup_lock.json').read_text())
    assert digest(HERE/'followup_protocol.json')==lock['protocol_sha256']
    assert digest(out/'lock.json')==lock['first_lock_sha256']
    primary=json.loads((out/'results.json').read_text())
    first_lock=json.loads((out/'lock.json').read_text())
    cal=pack(out,PLAN['cal'])
    cohorts={k:pack(out,v) for k,v in PLAN['cohorts'].items()}
    result={k:{} for k in cohorts}
    for arm,item in lock['models'].items():
        for seed,h in item['seeds'].items():
            path=out/'models'/f'{arm}_{seed}.joblib'
            assert digest(path)==h['model_sha256']
            model=joblib.load(path)
            normal=cal['y']>=.7
            cutoff=float(np.quantile(model.predict(matrix(cal,arm)[normal]),.1,method='lower'))
            for k,d in cohorts.items():
                p=model.predict(matrix(d,arm))
                with np.load(out/('predictions_'+k+'.npz'),allow_pickle=False) as previous:
                    assert np.array_equal(previous['sample_id'],d['sample_id'])
                    baseline=previous['base_'+seed]
                values=stats(d,p,cutoff)
                if seed=='23411':
                    values['paired']=paired(d,p,baseline)
                result[k][f'{arm}_{seed}']=values
    a=lock['candidate']
    merged=dict(first_lock['models'],**lock['models'])
    cv_dev=all(merged[a][k]<merged['base'][k] for k in ('cv','dev'))
    seeds={}
    for seed in PLAN['seeds']:
        ds=[]
        hits=0
        for k in PLAN['primary']:
            options=dict(primary['cohorts'][k]['models'],**result[k])
            cand,base=options[f'{a}_{seed}'],options[f'base_{seed}']
            ds.append(cand['mae']-base['mae'])
            hits+=cand['budget']['0.1']['hits']-base['budget']['0.1']['hits']
        seeds[str(seed)]=dict(macro_delta=float(np.mean(ds)),worst_delta=max(ds),hit_delta=hits,
                              passed=bool(np.mean(ds)<=-.003 and max(ds)<=.003 and hits>=0))
    dump(out/'followup_results.json',dict(cohorts=result,candidate=a,dev_ok=cv_dev,seeds=seeds,
                                         passed=bool(cv_dev and all(r['passed'] for r in seeds.values()))))
    print('followup',a,seeds,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['develop','diagnose'])
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    (develop if args.mode=='develop' else diagnose)(args.out)
