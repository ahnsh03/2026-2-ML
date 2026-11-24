"""Paired fixed-ROI ablations. Run develop before diagnose."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

from quality_estimation.gaussian_process import fit_quality_gp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = json.loads((HERE/'protocol.json').read_text())
BASELINE = json.loads((ROOT/'config/spatial_quality_baseline_v1.json').read_text())
ADD = ['global_shape','road_shape','all_shape','uncertainty','relations','coverage','gap','soft']
REMOVE = ['only_rgb','only_heads','drop_hood','drop_upper','drop_global','global_only','road_global']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def pack(out, keys):
    data = {}
    expected = None
    for key in keys:
        with np.load(out/(key+'.npz'), allow_pickle=False) as f:
            names = f['names'].tolist()
            if expected is None:
                expected = names
            assert expected == names
            for name in f.files:
                if name != 'names':
                    data.setdefault(name, []).append(f[name])
            data.setdefault('session', []).append(np.repeat(key,len(f['x'])))
    d = {k:np.concatenate(v) for k,v in data.items()}
    d['names'] = expected
    d['y'] = d['labels'][:,0]
    counts = d['labels'][:,1:]
    assert np.all(counts.sum(1)>0), 'modify review handling explicitly for undefined targets'
    np.testing.assert_allclose(d['y'], counts[:,0]/counts.sum(1), atol=1e-15, rtol=0)
    return d


def features(d):
    x, names = d['x'], d['names']
    schema = json.loads((ROOT/'docs/baselines/roi_selection_20261124/selected_schema.json').read_text())
    base_names = schema['ordered_features']
    assert names[:96] == base_names
    b, g, r = base_names, names[96:99], names[99:111]
    rel_names, rel_values = [], []
    for stat in ('intensity','lane_fraction','lane_probability'):
        fl, fr, nl, nr = [x[:,names.index(region+'.'+stat)] for region in
                          ('far_left','far_right','near_left','near_right')]
        for label, values in [('far_lr',fl-fr),('near_lr',nl-nr),('near_far',((nl+nr)-(fl+fr))/2)]:
            rel_names.append(label+'.'+stat)
            rel_values.append(values)
    full_names = names + rel_names
    full = np.column_stack([x,*rel_values])
    head_stats = {s.split('.')[1] for s in b[8:12]}
    arms = dict(base=b, global_shape=b+g, road_shape=b+r, all_shape=b+g+r,
                uncertainty=b+names[111:114], relations=b+rel_names,
                coverage=b+g[:1], gap=b+g[1:2], soft=b+g[2:],
                only_rgb=[s for s in b if s.split('.')[1] not in head_stats],
                only_heads=[s for s in b if s.split('.')[1] in head_stats],
                drop_hood=[s for s in b if not s.startswith('hood.')],
                drop_upper=[s for s in b if not s.startswith('upper_')],
                drop_global=[s for s in b if not s.startswith('global.')],
                global_only=[s for s in b if s.startswith('global.')],
                road_global=[s for s in b if s.startswith(('far_','near_','global.'))])
    assert set(arms) == set(PLAN['arms'])
    return {k:full[:,[full_names.index(s) for s in cols]] for k,cols in arms.items()}, arms


def mae(y, s):
    return float(np.abs(s-y).mean())


def rf(seed):
    return RandomForestRegressor(**BASELINE['rf_parameters'], random_state=seed, n_jobs=5)


def develop(out):
    t, d = pack(out,PLAN['train']), pack(out,PLAN['dev'])
    tx, arms = features(t)
    dx, _ = features(d)
    assert not np.intersect1d(t['pixel_sha256'],d['pixel_sha256']).size
    folds = [np.isin(t['session'],PLAN['train'][i::3]) for i in range(3)]
    distances = [float(cKDTree(t['xy'][~f]).query(t['xy'][f])[0].min()) for f in folds]
    report = {}
    model_dir = out/'models'
    model_dir.mkdir(exist_ok=True)
    for arm in PLAN['arms']:
        seeds = {}
        for seed in PLAN['seeds']:
            cv = []
            for f in folds:
                model = rf(seed).fit(tx[arm][~f],t['y'][~f].astype(np.float32))
                cv.append(mae(t['y'][f],model.predict(tx[arm][f])))
            model = rf(seed).fit(tx[arm],t['y'].astype(np.float32))
            path = model_dir / f'{arm}_{seed}.joblib'
            joblib.dump(model,path,compress=1)
            seeds[str(seed)] = dict(folds=cv,cv=float(np.mean(cv)),dev=mae(d['y'],model.predict(dx[arm])),
                                    model_sha256=digest(path))
        report[arm] = dict(n=len(arms[arm]),seeds=seeds,
                           **{k:float(np.mean([row[k] for row in seeds.values()])) for k in ('cv','dev')})
        print(arm,round(report[arm]['dev'],5),flush=True)
    add = min(ADD,key=lambda a:(report[a]['dev'],report[a]['n']))
    base = report['base']
    eligible = [k for k in REMOVE if all(report[k][v]<=base[v]+.002 for v in ('dev','cv'))]
    prune = min(eligible,key=lambda a:(report[a]['n'],report[a]['dev'])) if eligible else None
    gps = sorted(set(['base','global_shape',add]+([prune] if prune else [])))
    gp_hashes = {}
    for arm in gps:
        m = fit_quality_gp(tx[arm],t['y'].astype(np.float32),**BASELINE['gp_parameters'])
        path = model_dir/('gp_'+arm+'.joblib')
        joblib.dump(m,path,compress=1)
        gp_hashes[arm]=digest(path)
    dump(out/'lock.json',dict(arms=arms,models=report,add=add,prune=prune,gp=gp_hashes,
                              train_dev_min_m=float(cKDTree(t['xy']).query(d['xy'])[0].min()),
                              fold_min_m=distances,plan_sha256=digest(HERE/'protocol.json'),
                              fit_packs={k:digest(out/(k+'.npz')) for k in PLAN['train']+PLAN['dev']}))


def review(y,p,ids,fraction=.1,threshold=.7):
    order = np.lexsort((ids,p))
    n = int(np.ceil(len(y)*fraction))
    return dict(n=n,hits=int((y[order[:n]]<threshold).sum()))


def stats(d,p,cutoff):
    y, ids = d['y'], d['sample_id']
    low = y < .7
    band = (y>=.65)&(y<.75)
    budget = {str(v):review(y,p,ids,v) for v in PLAN['review_fractions']}
    cells = np.unique(np.c_[d['weather'],d['hour']],axis=0)
    hits, count = 0, 0
    for weather,hour in cells:
        sel = (d['weather']==weather)&(d['hour']==hour)
        row = review(y[sel],p[sel],ids[sel])
        hits += row['hits']
        count += row['n']
    result = dict(mae=mae(y,p), bias=float((p-y).mean()), budget=budget,
                  auroc=float(roc_auc_score(low,-p)) if np.unique(low).size==2 else None,
                  within=dict(hits=hits,n=count), boundary_mae=mae(y[band],p[band]) if band.any() else None,
                  alarm=dict(tp=int(((p<cutoff)&low).sum()),fp=int(((p<cutoff)&~low).sum())))
    if np.all(d['frame_id']>=0):
        hits, count = 0, 0
        for s in np.unique(d['session']):
            idx = np.flatnonzero(d['session']==s)
            idx = idx[np.argsort(d['frame_id'][idx])]
            for start in range(0,len(idx),64):
                w=idx[start:start+64]
                v=review(y[w],p[w],ids[w])
                hits+=v['hits']
                count+=v['n']
        result['window']=dict(hits=hits,n=count)
    return result


def paired(d,p,ref):
    if np.all(d['frame_id']>=0):
        _,group=np.unique(np.floor(d['xy']/50).astype(int),axis=0,return_inverse=True)
    else:
        _,group=np.unique(d['pose'],return_inverse=True)
    delta = np.abs(d['y']-p)-np.abs(d['y']-ref)
    count=np.bincount(group)
    total=np.bincount(group,weights=delta)
    rng=np.random.default_rng(23411)
    draw=rng.integers(0,len(count),size=(2000,len(count)))
    boot=total[draw].sum(1)/count[draw].sum(1)
    return dict(groups=len(count),delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


def diagnose(out):
    lock=json.loads((out/'lock.json').read_text())
    assert digest(HERE/'protocol.json')==lock['plan_sha256']
    receipt=json.loads((out/'receipt.json').read_text())
    for k,row in receipt['sources'].items():
        assert digest(out/(k+'.npz'))==row['pack_sha256']
        if k in lock['fit_packs']:
            assert row['pack_sha256']==lock['fit_packs'][k]
    keys=dict(calibration=PLAN['cal'], **PLAN['cohorts'])
    cal=pack(out,PLAN['cal'])
    cx,_=features(cal)
    train=pack(out,PLAN['train'])
    ds={k:pack(out,v) for k,v in keys.items()}
    xs={k:features(d)[0] for k,d in ds.items()}
    for field in ('pixel_sha256','sample_id'):
        merged=np.concatenate([train[field],pack(out,PLAN['dev'])[field],*[d[field] for d in ds.values()]])
        assert np.unique(merged).size==merged.size
    output={}
    pred={k:{} for k in ds}
    for key,d in ds.items():
        y=d['y']
        output[key]=dict(n=len(y),low=int((y<.7).sum()),boundary=int(((y>=.65)&(y<.75)).sum()),
                         under25=int((cKDTree(train['xy']).query(d['xy'])[0]<25).sum()),models={},paired={})
    jobs=[(arm,str(seed),row['model_sha256']) for arm,item in lock['models'].items() for seed,row in item['seeds'].items()]
    jobs += [(arm,'gp',h) for arm,h in lock['gp'].items()]
    for arm,seed,h in jobs:
        file=(f'{arm}_{seed}' if seed!='gp' else 'gp_'+arm)+'.joblib'
        path=out/'models'/file
        assert digest(path)==h
        model=joblib.load(path)
        normal=cal['y']>=.7
        cutoff=float(np.quantile(np.clip(model.predict(cx[arm][normal]),0,1),.1,method='lower'))
        tag=f'{arm}_{seed}'
        for k,d in ds.items():
            p=np.clip(model.predict(xs[k][arm]),0,1)
            pred[k][tag]=p
            output[k]['models'][tag]=dict(cutoff=cutoff,**stats(d,p,cutoff))
    for k,d in ds.items():
        for arm in PLAN['arms']:
            output[k]['paired'][arm]=paired(d,pred[k][arm+'_23411'],pred[k]['base_23411'])
        np.savez_compressed(out/('predictions_'+k+'.npz'),sample_id=d['sample_id'],iou=d['y'],**pred[k])
    base=lock['models']['base']
    decisions={}
    for kind in ('add','prune'):
        arm=lock[kind]
        if arm is None:
            decisions[kind]=dict(candidate=None,passed=False)
            continue
        item=lock['models'][arm]
        development_ok=all(item[k]<base[k] for k in ('dev','cv')) if kind=='add' else True
        checks={}
        for seed in PLAN['seeds']:
            delta, hits=[],[]
            for k in PLAN['primary']:
                m=output[k]['models']
                a,b=m[f'{arm}_{seed}'],m[f'base_{seed}']
                delta.append(a['mae']-b['mae'])
                hits.append(a['budget']['0.1']['hits']-b['budget']['0.1']['hits'])
            ok=(max(delta)<=.003 and sum(hits)>=0 and (kind=='prune' or np.mean(delta)<=-.003))
            checks[str(seed)]=dict(macro_delta=float(np.mean(delta)),worst_delta=max(delta),hit_delta=sum(hits),passed=bool(ok))
        decisions[kind]=dict(candidate=arm,dev_ok=development_ok,seeds=checks,
                             passed=bool(development_ok and all(c['passed'] for c in checks.values())))
    adopted=next((v['candidate'] for v in decisions.values() if v['passed']),'base')
    dump(out/'results.json',dict(cohorts=output,decisions=decisions,selected=adopted))
    print('selection',adopted,decisions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['develop','diagnose'])
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    (develop if args.mode=='develop' else diagnose)(args.out)
