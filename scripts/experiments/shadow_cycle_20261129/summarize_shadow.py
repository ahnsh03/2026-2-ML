"""Audit immutable archive membership and receiver-to-score timing, without GT."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from common.bridge import sha256_file


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    archive, review = a.session / 'archive', a.session / 'review'
    raw = records(archive / 'rgb.jsonl')
    predicted = records(review / 'predictions.jsonl')
    skipped = records(review / 'inference_skipped.jsonl')
    selected = records(review / 'candidates.jsonl')
    rows = {r['sample_id']: r for r in raw}
    assert len(rows) == len(raw)
    seen = {r['sample_id'] for r in predicted}
    omitted = {r['sample_id'] for r in skipped}
    assert len(seen) == len(predicted) and len(omitted) == len(skipped)
    assert seen.isdisjoint(omitted) and seen | omitted == set(rows)
    for row in raw:
        assert sha256_file(archive / row['image']) == row['expected_sha256']['rgb']
    ages = []
    for row in predicted:
        original = rows[row['sample_id']]
        assert row['rgb_sha256'] == original['expected_sha256']['rgb']
        ages.append((datetime.fromisoformat(row['processed_utc']) -
                     datetime.fromisoformat(original['received_utc'])).total_seconds())
    assert ages and min(ages) >= 0
    groups = defaultdict(list)
    for row in selected:
        groups[row['review_family'], row['window']].append(row)
    windows = []
    for (family, window), group in sorted(groups.items()):
        assert len({r['sample_id'] for r in group}) == len(group) == group[0]['budget']
        assert all(r['sample_id'] in seen and not r['admission_approved'] for r in group)
        windows.append(dict(family=family, window=window, population=group[0]['population'],
                            budget=group[0]['budget'], candidates=len(group)))
    result = dict(raw_frames=len(raw), inferred=len(predicted), inference_skipped=len(skipped),
                  all_raw_bytes_and_membership_verified=True,
                  receiver_to_score_seconds={name: float(value) for name, value in
                    zip(['minimum', 'median', 'p95', 'maximum'],
                        [min(ages), np.median(ages), np.quantile(ages, .95), max(ages)])},
                  windows=windows, receiver=read(archive / 'receiver_done.json'),
                  worker=read(review / 'run.json'), context_sha256=sha256_file(review / 'context.json'),
                  source_sha256=sha256_file(Path(__file__)),
                  scope='Same Docker host receiver/worker UTC. Includes queue and inference after callback; excludes simulator exposure/network time before callback. No GT accuracy or automatic admission.')
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(raw=len(raw), inferred=len(predicted), latency=result['receiver_to_score_seconds'])))


if __name__ == '__main__':
    main()
