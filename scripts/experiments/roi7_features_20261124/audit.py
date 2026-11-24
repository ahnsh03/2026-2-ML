"""Independent archived-map parity, release-baseline parity, and matched decision diagnostics."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from common.bridge import DEFAULT_CHECKPOINT_SHA256
from quality_estimation.spatial_features import ROI7_PROFILE_DIR, SpatialProfile, extract_spatial_features
from evaluate import PLAN, ROOT, dump, pack, review


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--previous',type=Path,help='Optional whole-frame reference packs')
    args=ap.parse_args()
    output=args.out
    data=args.data/'ml_roi10_20261123/afternoon'
    receipt=json.loads((output/'receipt.json').read_text())
    lock=json.loads((output/'lock.json').read_text())
    result=json.loads((output/'results.json').read_text())
    profile=SpatialProfile.load(ROI7_PROFILE_DIR)
    assert profile.regions_sha256==receipt['schema']['regions_sha256']
    archive={}
    for group in receipt['sources']:
        path=data/'maps'/(group+'.npz')
        rows=[json.loads(line) for line in (data/'protocols'/(group+'.jsonl')).read_text().splitlines()]
        d=pack(output,[group])
        with np.load(path,allow_pickle=False) as stored:
            assert stored['checkpoint_sha'].item()==DEFAULT_CHECKPOINT_SHA256
            assert np.array_equal(d['sample_id'],stored['ids'])
            maps=stored['maps']
            idx=np.unique(np.linspace(0,len(rows)-1,3).astype(int))
            max_diff=[]
            for i in idx:
                img=cv2.imread(str(args.data/'ml_quality_study_20261123'/rows[i]['image']))
                feat=extract_spatial_features(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),maps[i],profile,include_global=True)
                max_diff.append(float(abs(feat-d['x'][i,:96]).max()))
        archive[group]=dict(n=len(idx),max_abs_feature_diff=max(max_diff))
    reference=json.loads((ROOT/'docs/baselines/roi_selection_20261124/results.json').read_text())['cohorts']
    mapped=dict(test18='test18',geography40='geography40',known12='prospective_known',stream='stream_new',
                other='other_road',unseen_hours='prospective_unseen',night='night_challenge')
    parity={}
    for key,original in mapped.items():
        item=reference[original]
        obs=result['cohorts'][key]
        assert (obs['n'],obs['low'])==(item['defined'],item['low'])
        ref=item['RF']['7']['mae']
        observed=obs['models']['base_23411']['mae']
        parity[key]=dict(reference=ref,measured=observed,delta=observed-ref)
    prior_audit={}
    if args.previous:
        for scope in ['development','held']:
            with np.load(args.previous/(scope+'.npz'),allow_pickle=False) as previous:
                lookup={k:i for i,k in enumerate(previous['sample_id'])}
                rows_checked=0
                maxdiff=0.
                for group in receipt['sources']:
                    d=pack(output,[group])
                    valid=np.asarray([k in lookup for k in d['sample_id']])
                    if not valid.any():
                        continue
                    idx=[lookup[k] for k in d['sample_id'][valid]]
                    assert np.array_equal(previous['pixel_sha256'][idx],d['pixel_sha256'][valid])
                    np.testing.assert_array_equal(previous['labels'][idx],d['labels'][valid])
                    delta=float(abs(previous['x'][idx,11:14]-d['x'][valid,96:99]).max())
                    maxdiff=max(maxdiff,delta)
                    rows_checked+=len(idx)
                prior_audit[scope]=dict(n=rows_checked,max_global_shape_diff=maxdiff)
    # A closer look at selection under the same environment and near the threshold.
    train=pack(output,PLAN['train'])
    arms=list(dict.fromkeys(['base','global_shape','road_shape',lock['add']]+([lock['prune']] if lock['prune'] else [])))
    details={}
    for key,keys in PLAN['cohorts'].items():
        d=pack(output,keys)
        p=np.load(output/('predictions_'+key+'.npz'))
        assert np.array_equal(d['sample_id'],p['sample_id'])
        dist=cKDTree(train['xy']).query(d['xy'])[0]
        details[key]={}
        for arm in arms:
            s=p[arm+'_23411']
            by={}
            support=dict(boundary=(d['y']>=.65)&(d['y']<.75),away25=dist>=25,
                         night=np.isin(d['hour'],[20,21,22,23,0,1,2,3,4,5]),
                         other_hours=~np.isin(d['hour'],[20,21,22,23,0,1,2,3,4,5]))
            for tag,sel in support.items():
                y=d['y'][sel]
                if len(y):
                    by[tag]=dict(n=len(y),low=int((y<.7).sum()),mae=float(abs(y-s[sel]).mean()),
                                 review=review(y,s[sel],d['sample_id'][sel]))
            details[key][arm]=by
    dump(output/'audit.json',dict(archived_map_checks=archive,public_mae_parity=parity,
                                  prior_packs=prior_audit,subgroups=details))
    print('archived maps',sum(r['n'] for r in archive.values()),max(r['max_abs_feature_diff'] for r in archive.values()))
    print('max release MAE diff',max(abs(row['delta']) for row in parity.values()))
    print('prior',prior_audit)


if __name__=='__main__':
    main()
