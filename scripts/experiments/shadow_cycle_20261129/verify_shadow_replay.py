"""Compare the exact processed live population with an immutable replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from common.bridge import sha256_file


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-batch-control-update', action='store_true',
                        help='Historical live run predates replay batch-control addition; core scorer hashes still required')
    args = parser.parse_args()
    live, replay = args.live / 'review', args.replay / 'review'
    left, right = [records(p / 'predictions.jsonl') for p in (live, replay)]
    assert left and [r['sample_id'] for r in left] == [r['sample_id'] for r in right]
    lc, rc = [json.loads((p / 'context.json').read_text()) for p in (live, replay)]
    for key in ['perception_identity', 'bundle_manifest_sha256', 'feature_columns',
                'window', 'budget_fraction']:
        assert lc[key] == rc[key], key
    same_worker = lc['source_sha256'] == rc['source_sha256']
    if not args.allow_batch_control_update:
        assert same_worker
        assert lc['implementation_sha256'] == rc['implementation_sha256']
    else:
        for name, digest in lc['implementation_sha256'].items():
            assert rc['implementation_sha256'][name] == digest, name
    delta = {}
    for family in left[0]['review']:
        delta[family] = {}
        for key, value in left[0]['review'][family].items():
            a = np.asarray([r['review'][family][key] for r in left])
            b = np.asarray([r['review'][family][key] for r in right])
            if a.dtype.kind in 'biu':
                assert np.array_equal(a, b), (family, key)
            else:
                change = float(np.max(np.abs(a - b)))
                assert change <= 1e-4, (family, key, change)
                delta[family][key] = change
    selections = [records(p / 'candidates.jsonl') for p in (live, replay)]
    identity = lambda r: (r['review_family'], r['window'], r['sample_id'], r['rank'],
                          r['population'], r['budget'], r['selection_reason'])
    assert list(map(identity, selections[0])) == list(map(identity, selections[1]))
    with np.load(live / 'features.npz') as a, np.load(replay / 'features.npz') as b:
        assert a.files == b.files
        feature_delta = {}
        for key in a.files:
            if a[key].dtype.kind in 'f':
                feature_delta[key] = float(np.max(np.abs(a[key] - b[key])))
            else:
                assert np.array_equal(a[key], b[key])
    report = dict(rows=len(left), same_worker_source=same_worker,
                  matched_core_scorer_and_models=True,
                  maximum_prediction_difference=delta, feature_difference=feature_delta,
                  selection_ids_order_reasons_match=True,
                  live_hardware=lc.get('runtime_hardware'),
                  replay_hardware=rc.get('runtime_hardware'),
                  source_hashes={f'{name}/{file}': sha256_file(p / file)
                                 for name, p in [('live', live), ('replay', replay)]
                                 for file in ['context.json', 'predictions.jsonl', 'candidates.jsonl']},
                  scope='Same processed RGB, windows and models. Runtime equivalence; no GT accuracy claim.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(rows=len(left), matched=True, feature_difference=feature_delta)))


if __name__ == '__main__':
    main()
