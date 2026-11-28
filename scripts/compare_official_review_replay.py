#!/usr/bin/env python3
"""Compare a completed RGB replay with an immutable official reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_completed(root):
    run = json.loads((root/"run.json").read_text())
    if run["status"] != "complete" or (root/"_SUCCESS").read_text().strip() != sha(root/"run.json"):
        raise ValueError("incomplete or changed run: "+str(root))
    for name, digest in run["artifact_sha256"].items():
        path = (root/name).resolve()
        path.relative_to(root.resolve())
        if sha(path) != digest:
            raise ValueError("run artifact checksum differs: "+name)
    records = lambda file: [json.loads(s) for s in (root/file).read_text().splitlines() if s.strip()]
    return run, records("predictions.jsonl"), records("candidates.jsonl")


def compare(actual, reference, tolerance=1e-4):
    left, predictions, candidates = read_completed(actual)
    right, expected, expected_candidates = read_completed(reference)
    failures, numeric = [], {}
    for key in ("rows", "budget", "perception_identity", "profile", "regions_sha256", "bundle_manifest_sha256", "manifest_sha256", "source_sha256"):
        if left[key] != right[key]:
            failures.append("run."+key)
    if [r["sample_id"] for r in predictions] != [r["sample_id"] for r in expected]:
        failures.append("prediction ID/order mismatch")
    elif len(predictions) != left["rows"]:
        failures.append("prediction count mismatch")
    else:
        for a,b in zip(predictions,expected):
            if set(a) != set(b):
                failures.append("prediction field schema mismatch")
                break
            for key in a:
                x,y = a[key],b[key]
                if isinstance(x,(float,int,list)) and not isinstance(x,bool):
                    aa,bb=np.asarray(x),np.asarray(y)
                    if aa.dtype.kind in "fiu" and bb.dtype.kind in "fiu" and aa.shape==bb.shape:
                        delta=float(np.max(abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.
                        numeric[key]=max(numeric.get(key,0.),delta)
                        if not np.isfinite(delta) or delta>tolerance:
                            failures.append("prediction."+key)
                        continue
                if x!=y:
                    failures.append("prediction."+key)
    selection_fields=("sample_id","selection_rank","selection_component","review_population","budget")
    select=lambda rows:[tuple(row.get(k) for k in selection_fields) for row in rows]
    selection_equal=select(candidates)==select(expected_candidates)
    if not selection_equal:
        failures.append("candidate ID/order/reason mismatch")
    feature_difference={}
    with np.load(actual/"features.npz",allow_pickle=False) as a,np.load(reference/"features.npz",allow_pickle=False) as b:
        if a.files!=b.files:
            failures.append("feature schema mismatch")
        else:
            for name in a.files:
                if a[name].dtype.kind=="f":
                    delta=float(np.max(abs(a[name]-b[name])))
                    feature_difference[name]=delta
                    if delta>tolerance:
                        failures.append("features."+name)
                elif not np.array_equal(a[name],b[name]):
                    failures.append("features."+name)
    return dict(status="matched" if not failures else "differences_detected",rows=left["rows"],budget=left["budget"],
                selection_ids_order_reasons_equal=selection_equal,
                selection_overlap=len({r["sample_id"]for r in candidates}&{r["sample_id"]for r in expected_candidates}),
                max_numeric_differences=numeric,feature_differences=feature_difference,
                tolerance=tolerance,failures=sorted(set(failures)),
                actual_run_sha256=sha(actual/"run.json"),reference_run_sha256=sha(reference/"run.json"),
                uses_ground_truth=False,actual_path=str(actual),reference_path=str(reference))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actual",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tolerance",type=float,default=1e-4,
                   help="Same numerical comparison threshold as the existing shadow replay verifier")
    a=p.parse_args()
    result=compare(a.actual,a.reference,a.tolerance)
    with a.output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:result[k]for k in ("status","rows","budget","selection_overlap","failures")}))
    if result["failures"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
