"""Check the numerical conclusions against stored labels and omission of any single evaluation group."""
import argparse
from pathlib import Path

import numpy as np

from prepare import PLAN, rows, read, sha, dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('data','out','destination'):
        ap.add_argument(key,type=Path)
    a=ap.parse_args()
    scored=read(a.out/'results.json')
    assert sha(a.out/'predictions.npz')==scored['predictions_sha']
    reg=rows(a.out/'registered.jsonl')
    ids=[]; archived=[]; links=[]
    for k in read(a.out/'features.json')['packs']:
        records=[r for r in reg if r['session']==k]
        links.extend(r['link'] for r in records)
        with np.load(a.data/'ml_roi10_20261123/afternoon/packs'/(k+'_base.npz')) as temp:
            archived.extend(temp['y']);ids.extend(temp['ids'])
    links=np.array(links);archived=np.array(archived)
    sensitivity={}
    with np.load(a.out/'predictions.npz') as p:
        np.testing.assert_array_equal(ids,p['ids'])
        for subset,use,g in [('links',p['isolated'],links),('repeat',links=='',p['session'])]:
            summary={}
            for dim in ('96','97','98top','98patch','102'):
                changes=[]; delta_archive=[]
                for seed in PLAN['seeds']:
                    pred=p[f'{dim}_{seed}'][use];ref=p[f'99_{seed}'][use]
                    yy=p['y'][use];old=archived[use];group=g[use]
                    unique=np.unique(group)
                    delta=abs(pred-yy)-abs(ref-yy)
                    prior_delta=abs(pred-old)-abs(ref-old)
                    means=np.array([np.nanmean(delta[group==k]) for k in unique])
                    archive_mean=np.mean([np.nanmean(prior_delta[group==k]) for k in unique])
                    delta_archive.append(float(archive_mean-means.mean()))
                    changes.extend((means.sum()-means)/(len(means)-1))
                summary[dim]=dict(leave_one_out_range_across_seeds=[min(changes),max(changes)],
                                  archive_minus_native_delta_max_abs=max(map(abs,delta_archive)))
            sensitivity[subset]=summary
    dump(a.destination/'sensitivity.json',sensitivity)
    print(sensitivity['links']['96'],sensitivity['repeat']['96'])


if __name__=='__main__':
    main()
