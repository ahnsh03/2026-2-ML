"""Preserve live inference batches during a processed-population replay."""
import argparse
import json
from pathlib import Path

from common.bridge import sha256_file


def make_batch_plan(predictions, manifest):
    ids = [r['sample_id'] for r in manifest]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('replay requires nonempty unique frame IDs')
    if ids != [r['sample_id'] for r in predictions]:
        raise ValueError('processed manifest and prediction populations differ')
    sizes, offset = [], 0
    while offset < len(predictions):
        size = predictions[offset]['batch_rows']
        if type(size) is not int or size < 1 or offset + size > len(ids):
            raise ValueError('invalid recorded inference batch size')
        batch = predictions[offset:offset + size]
        if any(r['batch_rows'] != size for r in batch):
            raise ValueError('incomplete or inconsistent recorded inference batch')
        if any(r['batch_seconds'] != batch[0]['batch_seconds'] for r in batch):
            raise ValueError('recorded inference batch timing differs')
        sizes.append(size)
        offset += size
    return dict(schema='shadow-replay-batches-1', sample_ids=ids, batch_sizes=sizes)


def validate_batch_plan(plan, rows):
    if plan.get('schema') != 'shadow-replay-batches-1':
        raise ValueError('unknown replay batch schema')
    sizes = plan['batch_sizes']
    ids = [r['sample_id'] for r in rows]
    if not sizes or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError('replay batch sizes must be positive integers')
    if sum(sizes) != len(rows) or ids != plan['sample_ids'] or len(set(ids)) != len(ids):
        raise ValueError('replay batch plan does not match the immutable manifest')
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = [args.review / n for n in ['predictions.jsonl', 'processed_rgb.jsonl']]
    records = [[json.loads(line) for line in f.read_text().splitlines()] for f in files]
    plan = make_batch_plan(*records)
    plan['source_sha256'] = {f.name: sha256_file(f) for f in files}
    with args.output.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
