"""Assess one checked f0/g0 -> f1/g1 update without changing runtime defaults."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from common.bridge import sha256_file
from .artifacts import write_json
from .dataset_io import require
from .model_io import load_model_run
from .perception_identity import identity_from_adaptation


def assess_update(incumbent, candidate, policy):
    keys = ("new_lane_iou", "retention_lane_iou", "quality_mae")
    for values in (incumbent, candidate):
        require(set(values) == set(keys), "missing or unknown cycle metric")
        require(all(isinstance(values[k], (int,float)) and math.isfinite(values[k])
                    and 0 <= values[k] <= 1 for k in keys), "invalid cycle metric")
    require(policy["schema_version"] == "ml-cycle-acceptance-1", "unknown cycle policy")
    for k in ("minimum_new_lane_gain", "maximum_retention_lane_loss", "maximum_quality_mae_increase"):
        require(isinstance(policy[k], (int,float)) and math.isfinite(policy[k]) and 0 <= policy[k] <= 1,
                "invalid cycle bound")
    delta = {k:candidate[k]-incumbent[k] for k in keys}
    checks = dict(new_lane_improvement=delta["new_lane_iou"] >= policy["minimum_new_lane_gain"],
                  historical_retention=delta["retention_lane_iou"] >= -policy["maximum_retention_lane_loss"],
                  matched_quality_accuracy=delta["quality_mae"] <= policy["maximum_quality_mae_increase"])
    return dict(promotion_eligible=all(checks.values()), checks=checks, delta=delta,
                failed_checks=[k for k,v in checks.items() if not v], deployment_performed=False)


def completed_run(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory/"run.json").read_text())
    require(record.get("status") == "complete", "cycle evidence incomplete")
    require((directory/"_SUCCESS").read_text().strip() == sha256_file(directory/"run.json"),
            "cycle evidence completion changed")
    require(bool(record.get("artifact_sha256")), "cycle evidence has no artifact inventory")
    for relative,digest in record["artifact_sha256"].items():
        path = (directory/relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("cycle artifact escapes evidence run") from exc
        require(sha256_file(path) == digest, "cycle evidence artifact changed")
    return record


def perception_rows(directories, checkpoint):
    result = {}
    for directory in directories:
        directory = Path(directory);run = completed_run(directory)
        require(run["checkpoint_sha256"] == checkpoint, "cycle perception checkpoint differs")
        for line in (directory/"per_frame.jsonl").read_text().splitlines():
            row = json.loads(line);sid = row["sample_id"]
            require(sid not in result, "duplicate cycle evaluation sample")
            for m in row["metrics"].values():
                require(all(isinstance(m[k],int) and m[k] >= 0 for k in ("tp","fp","fn")),
                        "invalid perception counts")
                union = m["tp"]+m["fp"]+m["fn"]
                require(m["iou"] is None if not union else m["iou"] is not None
                        and abs(m["iou"]-m["tp"]/union)<1e-12, "perception count/IoU mismatch")
            result[sid] = row
    require(bool(result), "empty cycle evaluation")
    return result


def check_paired(incumbent, candidate):
    require(list(incumbent) == list(candidate), "cycle evaluation rows/order differ")
    for sid,left in incumbent.items():
        right = candidate[sid]
        require(left["input_sha256"] == right["input_sha256"], "cycle RGB/GT differs")
        require(left.get("condition") == right.get("condition"), "cycle condition metadata differs")
        require(set(left["metrics"]) == set(right["metrics"]), "cycle target heads differ")
        for head,m in left["metrics"].items():
            other=right["metrics"][head]
            require(m["tp"]+m["fn"] == other["tp"]+other["fn"], "cycle GT support differs")
        require((left["metrics"]["lane"]["iou"] is None) == (right["metrics"]["lane"]["iou"] is None),
                "cycle lane denominators differ")


def quality_mae(directories, model_run, model, perception):
    predictions = {}
    for directory in directories:
        directory=Path(directory);run=completed_run(directory)
        require(run["model_run_sha256"] == sha256_file(Path(model_run)/"run.json"),
                "cycle quality model differs")
        with (directory/"predictions.csv").open() as f:
            for row in csv.DictReader(f):
                sid=row["sample_id"];require(sid not in predictions,"duplicate cycle quality row")
                predictions[sid]=row
    require(list(predictions) == list(perception), "quality/perception evaluation rows differ")
    errors=[]
    for sid,row in predictions.items():
        target=perception[sid]["metrics"]["lane"]["iou"]
        require(bool(row["actual_iou"]) == (target is not None), "quality target defined mask differs")
        if target is not None:
            require(abs(float(row["actual_iou"])-target)<=3e-8, "quality target differs from perception")
            prediction=float(row[model+"_clipped"])
            require(math.isfinite(prediction) and 0 <= prediction <= 1,"invalid quality estimate")
            errors.append(abs(prediction-float(row["actual_iou"])))
    require(bool(errors),"no defined quality targets")
    return math.fsum(errors)/len(errors)


def lane_mean(records):
    values=[r["metrics"]["lane"]["iou"] for r in records.values() if r["metrics"]["lane"]["iou"] is not None]
    require(bool(values),"no defined perception targets")
    return math.fsum(values)/len(values)


def condition_deltas(incumbent, candidate):
    """Expose subgroup regressions even when the pooled acceptance metric improves."""
    groups={}
    for sid,row in incumbent.items():
        key=json.dumps(row.get('condition'),sort_keys=True)
        groups.setdefault(key,[]).append(sid)
    result={}
    for key,ids in groups.items():
        defined=[sid for sid in ids if incumbent[sid]['metrics']['lane']['iou'] is not None]
        left=lane_mean({sid:incumbent[sid] for sid in defined}) if defined else None
        right=lane_mean({sid:candidate[sid] for sid in defined}) if defined else None
        result[key]=dict(rows=len(ids),defined_rows=len(defined),incumbent_lane_iou=left,candidate_lane_iou=right,
                         delta=right-left if defined else None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("incumbent-new-perception","candidate-new-perception","incumbent-retention","candidate-retention",
                 "incumbent-quality-evaluation","candidate-quality-evaluation"):
        p.add_argument("--"+name,action="append",type=Path,required=True)
    p.add_argument("--incumbent-model-run",type=Path,required=True)
    p.add_argument("--candidate-model-run",type=Path,required=True)
    p.add_argument("--candidate-adaptation-run",type=Path,required=True)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();require(not a.output_dir.exists(),"cycle assessment already exists")
    identity=identity_from_adaptation(a.candidate_adaptation_run)
    base,_=load_model_run(a.incumbent_model_run)
    updated,_=load_model_run(a.candidate_model_run,perception_identity=identity)
    left=perception_rows(a.incumbent_new_perception,base["checkpoint_sha256"])
    right=perception_rows(a.candidate_new_perception,updated["checkpoint_sha256"])
    oldleft=perception_rows(a.incumbent_retention,base["checkpoint_sha256"])
    oldright=perception_rows(a.candidate_retention,updated["checkpoint_sha256"])
    check_paired(left,right);check_paired(oldleft,oldright)
    require(not set(left).intersection(oldleft), "new/retention evaluation IDs overlap")
    require(not {r["pixel_sha256"] for r in left.values()}.intersection(
                r["pixel_sha256"] for r in oldleft.values()), "new/retention RGB pixels overlap")
    incumbent=dict(new_lane_iou=lane_mean(left),retention_lane_iou=lane_mean(oldleft),
                   quality_mae=quality_mae(a.incumbent_quality_evaluation,a.incumbent_model_run,base["model"],left))
    candidate=dict(new_lane_iou=lane_mean(right),retention_lane_iou=lane_mean(oldright),
                   quality_mae=quality_mae(a.candidate_quality_evaluation,a.candidate_model_run,updated["model"],right))
    policy=json.loads(a.policy.read_text());decision=assess_update(incumbent,candidate,policy)
    evidence={}
    for values in vars(a).values():
        for path in values if isinstance(values,list) else [values]:
            if isinstance(path,Path) and (path/"run.json").exists():evidence[str(path)]=sha256_file(path/"run.json")
    a.output_dir.mkdir(parents=True)
    (a.output_dir/"source.py").write_bytes(Path(__file__).read_bytes())
    (a.output_dir/"policy.json").write_bytes(a.policy.read_bytes())
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status="complete",decision=decision,
                incumbent=incumbent,candidate=candidate,new_rows=len(left),retention_rows=len(oldleft),
                new_condition_deltas=condition_deltas(left,right),
                retention_condition_deltas=condition_deltas(oldleft,oldright),
                candidate_perception_identity=identity,evidence_run_sha256=evidence,paired_inputs_and_gt_verified=True,
                limitations=["Acceptance bounds are research policy, not statistical significance or safety certification.",
                             "The predeclared policy constrains pooled means; reported subgroup deltas are not individually bounded.",
                             "Scene/environment independence must be assessed separately from matching row hashes."],
                artifact_sha256={p.name:sha256_file(p) for p in a.output_dir.iterdir() if p.is_file()})
    write_json(a.output_dir/"report.json",report)
    (a.output_dir/"_SUCCESS").write_text(sha256_file(a.output_dir/"report.json")+"\n")
    print(json.dumps(decision))


if __name__=="__main__":
    main()
