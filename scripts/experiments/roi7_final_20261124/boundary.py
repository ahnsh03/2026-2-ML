"""Measure alarm tradeoffs at cutoffs chosen solely from the original calibration split."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from prepare import PLAN, read, rows, sha, dump
from score import confusion


def measure(y, p, cutoff):
    band = (y >= .65) & (y < .75)
    return dict(all=confusion(y, p, cutoff), boundary=confusion(y[band], p[band], cutoff))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'final', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    final = args.final
    locked = read(final/'lock.json')
    scored = read(final/'results.json')
    assert sha(final/'predictions.npz') == scored['predictions_sha']
    registered = {r['sample_id']: r for r in rows(final/'registered.jsonl')}
    with np.load(final/'predictions.npz') as pack:
        last = dict(pack)
    prior = read(args.old/'receipt.json')['sources']
    inputs, labels = [], []
    for scene in ('z03', 'z12'):
        pack = args.old/(scene+'.npz')
        assert sha(pack) == prior[scene]['pack_sha256']
        with np.load(pack) as data:
            inputs.append(data['x'])
            labels.append(data['labels'][:, 0])
            names = data['names'].tolist()
    y_cal = np.concatenate(labels)
    x_cal = np.concatenate(inputs)
    index = {'96': list(range(96)),
             '99': list(range(96)) + [names.index('global.'+s) for s in
                                        ('top_entropy','patch_entropy','lane_margin')]}
    is_driving = np.array([not registered[k]['static'] for k in last['ids']])
    cohorts = {'separate_links':last['isolated'], 'repeat_driving':is_driving}
    result = {}
    for count in ('96', '99'):
        result[count] = {}
        for seed in PLAN['seeds']:
            arm = PLAN['candidates'][count]+f'_{seed}'
            path = args.old/'models'/(arm+'.joblib')
            assert sha(path) == locked['frozen_models'][arm]
            pc = joblib.load(path).predict(x_cal[:,index[count]])
            normal = y_cal >= .7
            sorted_normal = np.sort(pc[normal])
            scores = last[f'{count}_{seed}']
            cutoffs = {'fixed_0.70': .7}
            for ratio in (.05,.10,.20):
                cutoffs[f'calibration_{ratio:.2f}'] = float(sorted_normal[int(ratio*len(sorted_normal))])
            result[count][str(seed)] = {
                key: dict(cutoff=cut, calibration=measure(y_cal,pc,cut),
                          **{k:measure(last['y'][mask],scores[mask],cut) for k,mask in cohorts.items()})
                for key,cut in cutoffs.items()
            }
    dump(args.output,dict(calibration_rows=len(y_cal),calibration_normals=int((y_cal>=.7).sum()),
                          threshold_policy='strictly below the score at floor(target normal FPR * n); ties may reduce realized FPR',
                          models=result))


if __name__=='__main__':
    main()
