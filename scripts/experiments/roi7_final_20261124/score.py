"""Score locked feature sets, audit independent labels, and apply registered decision rules."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from prepare import HERE, PLAN, read, rows, sha, dump


def rate(n,d):
    return float(n/d) if d else None


def confusion(y,p,cut):
    valid=np.isfinite(y)
    low=valid & (y<.7)
    normal=valid & ~low
    return dict(fn=int((low & (p>=cut)).sum()),failures=int(low.sum()),
                fp=int((normal & (p<cut)).sum()),normals=int(normal.sum()))


def pick(ids,p,idx):
    n=int(np.ceil(.1*len(idx)))
    return idx[np.lexsort((ids[idx],p[idx]))[:n]]


def hits(y,selected):
    return dict(found=int((y[selected]<.7).sum()),slots=len(selected),unknown=int(np.isnan(y[selected]).sum()))


def metrics(y,p,ids,group,session,frame,threshold,moving):
    valid=np.isfinite(y)
    band=valid & (y>=.65) & (y<.75)
    idx=np.arange(len(y))
    group_mae=[abs(p[valid & (group==g)]-y[valid & (group==g)]).mean() for g in np.unique(group[valid])]
    review=hits(y,pick(ids,p,idx))
    window=None
    if moving:
        chosen=[]
        for s in np.unique(session):
            ix=idx[session==s]
            ix=ix[np.argsort(frame[ix])]
            assert np.all(np.diff(frame[ix])>0)
            for i in range(0,len(ix),64):
                chosen.extend(pick(ids,p,ix[i:i+64]))
        window=hits(y,np.asarray(chosen))
    result=dict(macro=float(np.mean(group_mae)),mae=float(abs(p[valid]-y[valid]).mean()),
                review=review,window=window,boundary_mae=float(abs(p[band]-y[band]).mean()) if band.any() else None,
                fixed=confusion(y,p,.7),calibrated=confusion(y,p,threshold),
                boundary=confusion(y[band],p[band],.7),boundary_calibrated=confusion(y[band],p[band],threshold))
    return result


def bootstrap(y,p,ref,groups,macro):
    ok=np.isfinite(y)
    if not ok.any():
        return None
    _,g=np.unique(groups[ok],return_inverse=True)
    counts=np.bincount(g)
    total=np.bincount(g,weights=abs(p[ok]-y[ok])-abs(ref[ok]-y[ok]))
    rng=np.random.default_rng(23411)
    draw=rng.integers(0,len(counts),size=(2000,len(counts)))
    if macro:
        draw_delta=(total/counts)[draw].mean(1)
        point=(total/counts).mean()
    else:
        draw_delta=total[draw].sum(1)/counts[draw].sum(1)
        point=total.sum()/counts.sum()
    return dict(groups=len(counts),point=float(point),ci=np.quantile(draw_delta,[.025,.975]).tolist(),upper_adjusted=float(np.quantile(draw_delta,.9875)))


def boundary_error_ci(y,p,ref,group):
    result={}
    for name,keep in [('false_negative',(y>=.65)&(y<.7)),('false_positive',(y>=.7)&(y<.75))]:
        if not keep.any():
            result[name]=None
            continue
        a,b=(p[keep]>=.7),(ref[keep]>=.7)
        if name=='false_positive':
            a,b=~a,~b
        _,g=np.unique(group[keep],return_inverse=True)
        count=np.bincount(g)
        err=np.bincount(g,weights=a.astype(int)-b.astype(int))
        draws=np.random.default_rng(23411).integers(0,len(count),size=(2000,len(count)))
        rates=err[draws].sum(1)/count[draws].sum(1)
        result[name]=dict(denominator=int(keep.sum()),groups=len(count),delta=float(err.sum()/count.sum()),
                          ci=np.quantile(rates,[.025,.975]).tolist())
    return result


def score(a):
    first=read(a.out/'lock.json')
    pair=read(a.out/'pairs_lock.json')
    receipt=read(a.out/'features.json')
    assert first['protocol_sha']==sha(HERE/'protocol.json') and first['manifest_sha']==sha(a.out/'registered.jsonl')
    assert pair['protocol']==sha(HERE/'minimality.json') and pair['main_lock']==receipt['registration']==sha(a.out/'lock.json')
    specs=read(HERE/'minimality.json')['pairs']
    registered=rows(a.out/'registered.jsonl')
    matrices=[]; truths=[]; audit={}; records=[]
    for key,digest in receipt['packs'].items():
        source=a.out/(key+'.npz')
        assert sha(source)==digest
        rs=[r for r in registered if r['session']==key]
        with np.load(source) as pack:
            ids=pack['ids']
            np.testing.assert_array_equal(ids,[r['sample_id'] for r in rs])
            assert pack['names'].tolist()==receipt['schema']
            matrices.append(pack['x'])
            tp,fp,fn=pack['counts'].T
        den=tp+fp+fn
        y=tp/np.where(den>0,den,np.nan)
        with np.load(a.data/'ml_roi10_20261123/afternoon/packs'/(key+'_base.npz')) as original:
            np.testing.assert_array_equal(ids,original['ids'])
            yy=original['y']
            assert np.array_equal(np.isfinite(y),np.isfinite(yy))
            err=np.abs(y-yy)
            audit[key]=dict(max_abs=float(np.nanmax(err)),above_float_round=int((err>1e-6).sum()),
                            failure_class_changed=int(((y<.7)!=(yy<.7)).sum()))
        truths.extend(y); records.extend(rs)
    X=np.vstack(matrices); y=np.array(truths)
    ids=np.asarray([r['sample_id'] for r in records])
    session=np.asarray([r['session'] for r in records])
    static=np.asarray([r['static'] for r in records])
    links=np.asarray([r['link'] for r in records])
    frame=np.asarray([r.get('frame_id',-1) for r in records])
    distance=np.asarray([r['position_distance'] for r in records])
    link_distance=np.asarray([r['link_distance'] for r in records])
    xy=np.array([r['xy'] for r in records])
    _,cells=np.unique(np.floor(xy/50).astype(int),axis=0,return_inverse=True)
    separated=static & (distance>=25) & (link_distance>=25)
    telemetry={}
    for key,rel in [('extra_cal','real_g1_extension/training'),('newroads','real_g1_extension/held'),
                    ('shapes','road_shape_transfer/held')]:
        source=a.data/'ml_quality_study_20261123/runs'/rel/'links.jsonl'
        live={r['id']:r for r in rows(source)}
        diffs=[]
        for i in np.flatnonzero(session==key):
            rec=records[i]; measured=live[int(rec['pose'].rsplit('_',1)[1])]
            assert measured['requested']==rec['link']
            if measured['actual']['id'].get('value')!=rec['link']:
                diffs.append(i)
        assert not separated[np.asarray(diffs,dtype=int)].any()
        telemetry[key]=dict(source_sha=sha(source),different_requested_actual=len(diffs))
    with np.load(a.models/'z03.npz') as f:
        original_names=f['names'].tolist()
    prior=read(a.models/'receipt.json')['sources']
    cal_x=[];cal_y=[]
    for k in ('z03','z12'):
        file=a.models/(k+'.npz')
        assert sha(file)==prior[k]['pack_sha256']
        with np.load(file) as d:
            cal_x.extend(d['x']);cal_y.extend(d['labels'][:,0])
    cx=np.array(cal_x); normal=np.asarray(cal_y)>=.7
    names=receipt['schema']
    base=names[:96]
    fields={'96':base,'97':base+['global.lane_margin'],'99':base+names[-3:],'102':names,
            **{k:base+v for k,v in specs.items()}}
    pred={}; cut={}
    for arm,cols in fields.items():
        for seed in PLAN['seeds']:
            if arm in specs:
                file=a.out/f'{arm}_{seed}.joblib'
                h=pair['results'][arm]['seeds'][str(seed)]['model_sha256']
            else:
                name=PLAN['candidates'][arm]+f'_{seed}'
                file=a.models/'models'/(name+'.joblib')
                h=first['frozen_models'][name]
            assert sha(file)==h
            m=joblib.load(file)
            key=f'{arm}_{seed}'
            pred[key]=m.predict(X[:,[names.index(s) for s in cols]])
            cut[key]=float(np.quantile(m.predict(cx[normal][:,[original_names.index(s) for s in cols]]),.1,method='lower'))
    sets=dict(links=separated,far_links=separated & (distance>=50),all_static=static,repeat=~static)
    # Also break down each collection without choosing among those subgroups.
    for s in np.unique(session):
        sets[s]=session==s
    result={}
    for key,sel in sets.items():
        moving=not static[sel].any()
        unit=session[sel] if moving else links[sel]
        if not sel.any():continue
        ys=y[sel];bd=(ys>=.65)&(ys<.75)
        rec=dict(n=int(sel.sum()),defined=int(np.isfinite(ys).sum()),failures=int((ys<.7).sum()),
                 groups=len(np.unique(unit)),boundary=int(bd.sum()),boundary_groups=len(np.unique(unit[bd])),
                 min_position_m=float(distance[sel].min()),min_link_m=float(link_distance[sel].min()) if not moving else None,
                 models={},diff={})
        for model,scores in pred.items():
            rec['models'][model]=metrics(ys,scores[sel],ids[sel],unit,session[sel],frame[sel],cut[model],moving)
        if key in ('links','repeat','far_links','all_static'):
            for arm in fields:
                p=pred[arm+'_23411'][sel]
                ref=pred['99_23411'][sel]
                rec['diff'][arm]=dict(macro=bootstrap(ys,p,ref,unit,True),
                    location_weighted=bootstrap(ys,p,ref,cells[sel] if moving else unit,False),
                    boundary_rates=boundary_error_ci(ys,p,ref,cells[sel] if moving else unit))
        result[key]=rec
    fallback=read(a.out/'fallback_lock.json')
    assert fallback['pair_lock']==sha(a.out/'pairs_lock.json') and fallback['plan']==sha(HERE/'size_fallback.json')
    nominees=fallback['order']
    decisions={}
    for arm in dict.fromkeys(['96','97','102']+nominees):
        if arm is None:continue
        ds=[]; changes=[]
        for name in ('links','repeat'):
            table=result[name]['models']
            budget='review' if name=='links' else 'window'
            for seed in PLAN['seeds']:
                cand,ref=[table[f'{k}_{seed}'] for k in (arm,'99')]
                ds.append(cand['macro']-ref['macro'])
                changes.append(cand[budget]['found']-ref[budget]['found'])
        decisions[arm]=dict(max_delta=max(ds),min_review_delta=min(changes),superior=bool(max(ds)<=-.003 and min(changes)>=0))
    spatial_harm=all(result['links']['models'][f'99_{s}']['macro']-result['links']['models'][f'96_{s}']['macro']>=.003 for s in PLAN['seeds'])
    spatial_harm=spatial_harm and result['links']['diff']['96']['macro']['ci'][1]<0
    eligible=[k for k in ('96','97','102') if decisions[k]['superior']]
    nominee=pair['nominee']
    simplifications={}
    for arm in nominees:
        simplifications[arm]=bool(decisions[arm]['max_delta']<=.001 and decisions[arm]['min_review_delta']>=0 and
            all(result[k]['diff'][arm]['macro']['upper_adjusted']<=.001 for k in ('links','repeat')))
    replacement=next((arm for arm in nominees if simplifications[arm]),None)
    if eligible:
        selected=min(eligible,key=lambda k:sum(result[name]['models'][f'{k}_{s}']['macro'] for name in ('links','repeat') for s in PLAN['seeds']))
    elif spatial_harm:
        selected='unconfirmed'
    else:
        selected=replacement or '99'
    np.savez_compressed(a.out/'predictions.npz',ids=ids,y=y,session=session,isolated=separated,**pred)
    dump(a.out/'results.json',dict(selected=selected,new_spatial_harm=bool(spatial_harm),superiority=decisions,
         simpler_nominee=nominee,simplification_passed=simplifications,calibration_cutoffs=cut,
         cohorts=result,data_counts_parity=audit,telemetry=telemetry,predictions_sha=sha(a.out/'predictions.npz')))
    print('selected',selected,'decisions',decisions,'simplify',simplifications,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','models','out'):
        parser.add_argument(key,type=Path)
    score(parser.parse_args())
