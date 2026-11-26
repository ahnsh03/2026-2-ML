"""Create a GT-free review queue from frozen calibrated RGB replay.

Session boundaries are preserved. Far-only weakness remains a diagnostic; every
candidate requires subsequent GT/operating-policy review before admission.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .drive_review import file_sha

def read(path):return json.loads(path.read_text())
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-size',type=int,default=64);parser.add_argument('--budget-fraction',type=float,default=.1)
    args=parser.parse_args()
    if args.window_size<1 or not 0<args.budget_fraction<=1:raise ValueError('invalid review budget')
    run=read(args.replay/'run.json')
    if run['status']!='complete' or (args.replay/'_SUCCESS').read_text().strip()!=file_sha(args.replay/'run.json'):raise ValueError('incomplete RGB replay')
    for name,digest in run['artifacts_sha256'].items():
        path=(args.replay/name).resolve();path.relative_to(args.replay.resolve())
        if file_sha(path)!=digest:raise ValueError('replay artifact checksum differs')
    rows=[json.loads(line) for line in (args.replay/'input_manifest.jsonl').read_text().splitlines() if line.strip()]
    sources={r['sample_id']:r for r in rows};predictions=[json.loads(line) for line in (args.replay/'predictions.jsonl').read_text().splitlines() if line.strip()]
    if len(sources)!=len(rows) or [r['sample_id'] for r in rows]!=[r['sample_id'] for r in predictions]:raise ValueError('RGB replay row order differs')
    sessions=[sources[r['sample_id']]['session_id'] for r in predictions]
    if any(not session for session in sessions):raise ValueError('explicit session IDs required')
    candidates=[];windows=[];start=0
    while start<len(predictions):
        session=sessions[start];end=start
        while end<len(predictions) and sessions[end]==session and end-start<args.window_size:end+=1
        part=predictions[start:end];ids=np.array([r['sample_id'] for r in part]);scores=np.array([r['priority'] for r in part]);budget=int(np.ceil(len(part)*args.budget_fraction))
        chosen=np.lexsort((ids,-scores))[:budget]
        for position in chosen:
            r=part[position];spatial=r['spatial'];source=sources[r['sample_id']]
            candidates.append(dict(r,source_image=source['image'],session_id=session,predicted_iou=float(np.clip(r['raw_mean'][0],0,1)),risk_scores=dict(global_low=r['global_probability'],near_weak=spatial[1]+spatial[3],far_only=spatial[2],both_weak=spatial[3]),selection_reason='frozen_calibrated_review_priority',priority_is_probability=False,policy_sha256=run['policy_sha256'],status='awaiting_gt_and_operating_policy_review',admission_approved=False))
        windows.append(dict(session_id=session,start=start,end=end,rows=end-start,budget=budget));start=end
    args.output.mkdir(parents=True,exist_ok=False);path=args.output/'candidates.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in candidates))
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),uses_ground_truth=False,rows=len(rows),selected=len(candidates),window_size=args.window_size,budget_fraction=args.budget_fraction,actual_budget_fraction=len(candidates)/len(rows),windows=windows,replay_sha256=file_sha(args.replay/'run.json'),candidates_sha256=file_sha(path),scope='Offline candidate logging only; no live ROS or automatic retraining admission.')
    result['bound_artifacts']={key:run[key] for key in ['checkpoint_sha256','model_sha256','auxiliary_model_sha256','calibration_sha256','policy_sha256','environment_profile_sha256']}
    (args.output/'run.json').write_text(json.dumps(result,indent=2)+'\n');(args.output/'_SUCCESS').write_text(file_sha(args.output/'run.json')+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
