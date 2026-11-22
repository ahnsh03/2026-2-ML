"""Build a GT-free review queue with priority ranking and a random remainder audit."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from common.bridge import sha256_file
from .artifacts import write_json
from .cycle_gate import completed_run
from .dataset_io import require


def choose_for_review(predictions, budget, audit_fraction=.2, seed=20261116):
    require(isinstance(budget,int) and budget > 0, "review budget must be a positive integer")
    require(math.isfinite(audit_fraction) and 0 <= audit_fraction <= 1, "invalid audit fraction")
    require(bool(predictions), "empty RGB predictions")
    ids=set();pixels={}
    for row in sorted(predictions,key=lambda r:r['sample_id']):
        sid=row['sample_id'];require(isinstance(sid,str) and sid and sid not in ids,"duplicate or invalid sample ID")
        ids.add(sid)
        require(isinstance(row['pixel_sha256'],str) and len(row['pixel_sha256'])==64,"missing decoded RGB identity")
        require(math.isfinite(row['estimated_iou']) and 0 <= row['estimated_iou'] <= 1,"invalid estimated IoU")
        pixels.setdefault(row['pixel_sha256'],row)
    pool=sorted(pixels.values(),key=lambda r:(r['estimated_iou'],r['sample_id']))
    require(budget <= len(pool),"review budget exceeds pixel-deduplicated pool")
    audit_count=min(budget,math.ceil(budget*audit_fraction))
    priority_count=budget-audit_count
    priority=pool[:priority_count]
    remainder=sorted(pool[priority_count:],key=lambda r:r['sample_id'])
    audit=random.Random(seed).sample(remainder,audit_count)
    return ([(r,'low_estimated_iou') for r in priority]+[(r,'random_remainder_audit') for r in audit],
            dict(raw_rows=len(predictions),pixel_unique_rows=len(pool),budget=budget,
                 priority_count=priority_count,audit_count=audit_count,audit_fraction=audit_fraction,
                 seed=seed,priority_order='estimated_iou ascending, then sample_id',
                 duplicate_policy='exact decoded RGB pixels; lexicographically first sample ID',
                 audit_scope='uniform without replacement from remaining unique pixels, not the whole pool'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference-run',type=Path,required=True)
    p.add_argument('--rgb-manifest',type=Path,required=True)
    p.add_argument('--budget',type=int,required=True)
    p.add_argument('--audit-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=20261116)
    p.add_argument('--purpose',choices=('candidate_pool','demonstration_only'),required=True,
                   help='Never admit held-out evaluation frames to training; use demonstration_only for them')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();require(not a.output_dir.exists(),'review queue already exists')
    run=completed_run(a.inference_run)
    require(run.get('uses_ground_truth') is False,'review input must be RGB-only inference')
    require(run['input_manifest_sha256']==sha256_file(a.rgb_manifest),'RGB manifest differs from inference')
    source=[json.loads(line) for line in a.rgb_manifest.read_text().splitlines() if line.strip()]
    predictions=[json.loads(line) for line in (a.inference_run/'predictions.jsonl').read_text().splitlines() if line.strip()]
    require([r['sample_id'] for r in source]==[r['sample_id'] for r in predictions], 'review source rows/order differ')
    # Restrict the manifest to transport references, so queue generation cannot copy labels or metadata accidentally.
    allowed={'sample_id','view','image','expected_sha256'}
    for src,row in zip(source,predictions):
        require(set(src).issubset(allowed) and src.get('view')=='front','review requires a front RGB-only manifest')
        expected=src.get('expected_sha256',{})
        require(set(expected).issubset({'rgb'}),'review manifest includes non-RGB hashes')
        require(not expected or expected['rgb']==row['rgb_sha256'],'review RGB hash differs')
    chosen,policy=choose_for_review(predictions,a.budget,a.audit_fraction,a.seed)
    by_id={r['sample_id']:r for r in source};queue=[]
    for rank,(row,reason) in enumerate(chosen,1):
        src=by_id[row['sample_id']]
        queue.append(dict(queue_index=rank,sample_id=row['sample_id'],view='front',image=src['image'],
                          expected_sha256={'rgb':row['rgb_sha256']},pixel_sha256=row['pixel_sha256'],
                          estimated_iou=row['estimated_iou'],review_reason=reason,purpose=a.purpose))
    a.output_dir.mkdir()
    with (a.output_dir/'queue.jsonl').open('w') as f:
        for row in queue:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (a.output_dir/'queue.csv').open('w') as f:
        fields=['queue_index','sample_id','image','estimated_iou','review_reason','purpose']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(queue)
    write_json(a.output_dir/'policy.json',policy)
    (a.output_dir/'source.py').write_bytes(Path(__file__).read_bytes())
    record=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),purpose=a.purpose,
                uses_ground_truth=False,training_admission_performed=False,
                source_inference_run_sha256=sha256_file(a.inference_run/'run.json'),
                source_rgb_manifest_sha256=sha256_file(a.rgb_manifest),
                model_run_sha256=run['model_run_sha256'],checkpoint_sha256=run['checkpoint_sha256'],
                perception_identity=run.get('perception_identity'),policy=policy,
                limitations=['Exact pixel deduplication only. Apply time/pose thinning upstream for moving captures.',
                             'This queue does not label, admit, train or deploy. Held-out frames are demonstration only.'],
                artifact_sha256={path.name:sha256_file(path) for path in a.output_dir.iterdir() if path.is_file()})
    write_json(a.output_dir/'run.json',record)
    (a.output_dir/'_SUCCESS').write_text(sha256_file(a.output_dir/'run.json')+'\n')
    print(json.dumps(policy))


if __name__=='__main__':
    main()
