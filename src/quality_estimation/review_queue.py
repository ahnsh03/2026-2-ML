"""Build a GT-free review queue with priority ranking and a random remainder audit."""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import random

from common.bridge import sha256_file
from .artifacts import write_json
from .cycle_gate import completed_run
from .dataset_io import require
from .environment_reference import load_reference, with_shortfall


PRIORITY_POLICIES = ("quality", "cluster_round_robin", "quality_uncertainty", "environment_relative")


def _validate_scores(row, priority_policy):
    require(isinstance(row['pixel_sha256'],str) and len(row['pixel_sha256'])==64,"missing decoded RGB identity")
    require(math.isfinite(row['estimated_iou']) and 0 <= row['estimated_iou'] <= 1,"invalid estimated IoU")
    if priority_policy == "cluster_round_robin":
        group = row.get("environment_cluster")
        require(type(group) is int and group >= 0, "expected a nonnegative environment cluster on every row")
    if priority_policy == "environment_relative":
        reference = row.get("environment_reference_iou")
        shortfall = row.get("estimated_iou_shortfall")
        require(type(reference) in (int, float) and 0 <= reference <= 1 and
                shortfall == reference - row["estimated_iou"], "expected a model-bound environment IoU shortfall")
    if priority_policy == "quality_uncertainty":
        sigma = row.get("prediction_std")
        require(type(sigma) in (int, float) and math.isfinite(sigma) and sigma >= 0,
                "expected a finite nonnegative IoU standard deviation")


def choose_for_review(predictions, budget, audit_fraction=.2, seed=20261116, *, priority_policy="quality", audit_slots=None):
    require(priority_policy in PRIORITY_POLICIES, "unknown review priority policy")
    require(isinstance(budget,int) and budget > 0, "review budget must be a positive integer")
    require(math.isfinite(audit_fraction) and 0 <= audit_fraction <= 1, "invalid audit fraction")
    require(bool(predictions), "empty RGB predictions")
    ids=set();pixels={}
    for row in sorted(predictions,key=lambda r:r['sample_id']):
        sid=row['sample_id'];require(isinstance(sid,str) and sid and sid not in ids,"duplicate or invalid sample ID")
        ids.add(sid)
        _validate_scores(row, priority_policy)
        pixels.setdefault(row['pixel_sha256'],row)
    pool=sorted(pixels.values(),key=lambda r:(r['estimated_iou'],r['sample_id']))
    require(budget <= len(pool),"review budget exceeds pixel-deduplicated pool")
    if audit_slots is None:
        audit_count = min(budget, math.ceil(budget*audit_fraction))
    else:
        require(type(audit_slots) is int and 0 <= audit_slots <= budget, "invalid explicit audit slots")
        audit_count = audit_slots
    priority_count=budget-audit_count
    priority_reason = "low_estimated_iou"
    priority_order = "estimated_iou ascending, then sample_id"
    if priority_policy == "cluster_round_robin":
        # The pool is risk-sorted. A stable sort by within-cluster rank takes
        # one available head from each cluster before taking any second row.
        seen = Counter()
        within_cluster = {}
        for row in pool:
            group = row["environment_cluster"]
            within_cluster[row["sample_id"]] = seen[group]
            seen[group] += 1
        pool.sort(key=lambda row: within_cluster[row["sample_id"]])
        priority_reason = "environment_round_robin"
        priority_order = "within-cluster rank, then estimated_iou, then sample_id"
    if priority_policy == "environment_relative":
        pool.sort(key=lambda row: (-row["estimated_iou_shortfall"], row["sample_id"]))
        priority_reason = "largest_environment_shortfall"
        priority_order = "training cluster IoU lower quintile minus estimated_iou descending, then sample_id"
    priority = [(row, priority_reason) for row in pool[:priority_count]]
    details = {}
    if priority_policy == "quality_uncertainty":
        risk_count = min(priority_count, max(1, 3 * priority_count // 4))
        exploration_count = priority_count - risk_count
        uncertain = sorted(pool[risk_count:], key=lambda row: (-row["prediction_std"], row["sample_id"]))
        priority = ([(row, "low_estimated_iou") for row in pool[:risk_count]]
                    + [(row, "high_prediction_std") for row in uncertain[:exploration_count]])
        priority_order = "low estimated_iou, then highest remaining prediction_std; ties by sample_id"
        details = {"risk_count": risk_count, "uncertainty_count": exploration_count}
    selected_ids = {row["sample_id"] for row, _ in priority}
    remainder=sorted((row for row in pool if row["sample_id"] not in selected_ids), key=lambda r:r['sample_id'])
    audit=random.Random(seed).sample(remainder,audit_count)
    return (priority+[(r,'random_remainder_audit') for r in audit],
            dict(raw_rows=len(predictions),pixel_unique_rows=len(pool),budget=budget,
                 priority_count=priority_count,audit_count=audit_count,audit_fraction=audit_fraction,
                 seed=seed,priority_order=priority_order,**details,
                 duplicate_policy='exact decoded RGB pixels; lexicographically first sample ID',
                 audit_scope='uniform without replacement from remaining unique pixels, not the whole pool'))


def review_windows(predictions, window_frames, budget_percent, audit_fraction=.2, seed=20261116, *,
                   priority_policy="quality"):
    """Yield each closed window before consuming future rows; within a session the first pixel copy wins."""
    require(type(window_frames) is int and window_frames > 0, "window size must be a positive frame count")
    require(type(budget_percent) is int and 1 <= budget_percent <= 100, "review percentage must be in 1..100")
    require(math.isfinite(audit_fraction) and 0 <= audit_fraction <= 1, "invalid audit fraction")
    require(priority_policy in PRIORITY_POLICIES, "unknown review priority policy")
    audit_ratio = Fraction(str(audit_fraction))
    ids, pixels = set(), set()
    window = []
    number = 0

    def close(rows, unique_seen, number):
        previous = (budget_percent * (unique_seen - len(rows)) + 99) // 100
        total = (budget_percent * unique_seen + 99) // 100
        slots = total - previous
        audits = math.ceil(total * audit_ratio) - math.ceil(previous * audit_ratio)
        chosen = []
        if slots:
            chosen, _ = choose_for_review(rows, slots, audit_fraction, seed + number,
                                          priority_policy=priority_policy, audit_slots=audits)
        return chosen, dict(window=number, frames=len(rows), unique_seen=unique_seen, budget=slots,
                            audit_count=audits, seed=seed + number, first_id=rows[0]["sample_id"],
                            last_id=rows[-1]["sample_id"])

    for row in predictions:
        sid = row["sample_id"]
        require(isinstance(sid, str) and sid and sid not in ids, "duplicate or invalid sample ID")
        ids.add(sid)
        _validate_scores(row, priority_policy)
        digest = row["pixel_sha256"]
        if digest in pixels:
            continue
        pixels.add(digest)
        window.append(row)
        if len(window) == window_frames:
            yield close(window, len(pixels), number)
            number += 1
            window = []
    if window:
        yield close(window, len(pixels), number)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference-run',type=Path,required=True)
    p.add_argument('--rgb-manifest',type=Path,required=True)
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument('--budget',type=int)
    size.add_argument('--budget-percent',type=int,help='Cumulative review allocation for chronological windows (1..100)')
    p.add_argument('--window-frames',type=int,help='Close a window after this many eligible unique RGB frames')
    p.add_argument('--audit-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=20261116)
    p.add_argument('--priority-policy', choices=PRIORITY_POLICIES, default='quality')
    p.add_argument('--environment-reference',type=Path,help='Frozen training reference required by environment_relative')
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
    if a.priority_policy == "cluster_round_robin":
        require(bool(run.get("environment_centers_sha256")), "environment-aware review requires frozen centers in inference")
    require((a.environment_reference is not None) == (a.priority_policy == "environment_relative"),
            "supply a frozen environment reference with environment_relative")
    if a.environment_reference is not None:
        reference = load_reference(a.environment_reference, run)
        predictions = [with_shortfall(row, reference) for row in predictions]
    if a.priority_policy == "quality_uncertainty":
        require(run.get("model") == "GaussianProcess", "uncertainty review requires a frozen GP inference run")
    require((a.window_frames is None) == (a.budget_percent is None),
            "provide --window-frames and --budget-percent together")
    window_records = []
    selected_window = {}
    if a.window_frames is None:
        chosen,policy=choose_for_review(predictions,a.budget,a.audit_fraction,a.seed,priority_policy=a.priority_policy)
    else:
        require(bool(predictions), "empty RGB predictions")
        chosen = []
        for selected, record in review_windows(predictions, a.window_frames, a.budget_percent,
                                                a.audit_fraction, a.seed, priority_policy=a.priority_policy):
            window_records.append(record)
            chosen.extend(selected)
            selected_window.update((row["sample_id"], record["window"]) for row, _ in selected)
        audit_count = sum(w["audit_count"] for w in window_records)
        policy = dict(raw_rows=len(predictions), pixel_unique_rows=window_records[-1]["unique_seen"],
                      budget=len(chosen), priority_count=len(chosen)-audit_count, audit_count=audit_count,
                      audit_fraction=a.audit_fraction, seed=a.seed, window_frames=a.window_frames,
                      budget_percent=a.budget_percent, priority_policy=a.priority_policy,
                      allocation="cumulative ceil percentages at each close; seed + zero-based window",
                      duplicate_policy="first occurrence in manifest order across the session")
    if a.priority_policy == "cluster_round_robin":
        policy["environment_centers_sha256"] = run["environment_centers_sha256"]
    if a.environment_reference is not None:
        policy["priority_policy"] = a.priority_policy
        policy["environment_reference_sha256"] = sha256_file(a.environment_reference / "run.json")
        policy["environment_centers_sha256"] = run["environment_centers_sha256"]
    by_id={r['sample_id']:r for r in source};queue=[]
    for rank,(row,reason) in enumerate(chosen,1):
        src=by_id[row['sample_id']]
        queue.append(dict(queue_index=rank,sample_id=row['sample_id'],view='front',image=src['image'],
                          expected_sha256={'rgb':row['rgb_sha256']},pixel_sha256=row['pixel_sha256'],
                          estimated_iou=row['estimated_iou'],review_reason=reason,purpose=a.purpose))
        if window_records:
            queue[-1]["window"] = selected_window[row["sample_id"]]
        if a.environment_reference is not None:
            queue[-1].update({key: row[key] for key in
                             ("environment_cluster", "environment_reference_iou", "estimated_iou_shortfall")})
        if a.priority_policy == "cluster_round_robin":
            queue[-1]["environment_cluster"] = row["environment_cluster"]
        if a.priority_policy == "quality_uncertainty":
            queue[-1]["prediction_std"] = row["prediction_std"]
    a.output_dir.mkdir()
    if window_records:
        write_json(a.output_dir / "windows.json", window_records)
    with (a.output_dir/'queue.jsonl').open('w') as f:
        for row in queue:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (a.output_dir/'queue.csv').open('w') as f:
        fields=['queue_index','sample_id','image','estimated_iou','review_reason','purpose']
        if a.priority_policy == "cluster_round_robin":
            fields.append("environment_cluster")
        if a.priority_policy == "quality_uncertainty":
            fields.append("prediction_std")
        if a.environment_reference is not None:
            fields.extend(["environment_cluster", "environment_reference_iou", "estimated_iou_shortfall"])
        if window_records:
            fields.append("window")
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
