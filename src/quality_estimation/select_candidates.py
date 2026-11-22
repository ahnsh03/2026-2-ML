"""Write review/admission manifests from RGB-only inference, without reading GT."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from itertools import islice, zip_longest
import math
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .artifacts import write_json
from .dataset_io import require
from .features import FEATURE_NAMES, feature_schema
from .splits import eligible_rows


def risk_pool_selections(order, ids, risk, X, count, budget, seeds, factor):
    """Compare diversity and uniform sampling inside exactly the same high-risk subset."""
    shortlist = order[:count * factor]
    ordered_ids = ids[shortlist].tolist()
    evidence = dict(risk_pool_factor=factor, risk_pool_rows=len(shortlist), risk_pool_sha256=hashlib.sha256(
        json.dumps(ordered_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
    features = X[shortlist].astype(np.float64)
    spread = features.std(axis=0)
    features = (features - features.mean(axis=0)) / np.where(spread > 0, spread, 1.)
    minimum = np.full(len(shortlist), np.inf)
    diverse = []
    current = 0
    for _ in range(count):
        diverse.append(current)
        offsets = features - features[current]
        minimum = np.minimum(minimum, np.einsum("ij,ij->i", offsets, offsets))
        minimum[current] = -1.
        current = int(np.argmax(minimum))
    groups = [("RiskPoolDiverse", None, shortlist[diverse])]
    for seed in seeds:
        groups.append(("RiskPoolRandom", seed, np.random.default_rng(seed).permutation(shortlist)[:count]))
    return [dict(method=method, seed=seed, budget=budget, selected_count=count, indices=indices.tolist(),
                 scores=risk[indices].tolist(), **evidence) for method, seed, indices in groups]


def area_stratified_selections(order, ids, risk, X, count, budget, seeds):
    """Allocate equal opportunities to small, medium and large predicted lane masks."""
    area = X[:, FEATURE_NAMES.index("global.lane_area_fraction")]
    require(((area[order] >= 0) & (area[order] <= 1)).all(), "invalid predicted lane fraction")
    edges = np.asarray([.001, .01], dtype=area.dtype)
    bands = np.searchsorted(edges, area, side="right")
    groups = [order[bands[order] == k] for k in range(3)]
    members = [[str(ids[i]) for i in group] for group in groups]
    digest = hashlib.sha256(json.dumps(members, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    selections = []
    for seed in [None, *seeds]:
        queues = groups
        if seed is not None:
            rng = np.random.default_rng(seed)
            queues = [rng.permutation(g) for g in groups]
        interleaved = (i for cycle in zip_longest(*queues) for i in cycle if i is not None)
        ids_in_order = [int(i) for i in islice(interleaved, count)]
        selected_bands = bands[ids_in_order].tolist()
        selections.append(dict(method="AreaStratifiedRisk" if seed is None else "AreaStratifiedRandom",
                               seed=seed, budget=budget, selected_count=count, indices=ids_in_order,
                               scores=risk[ids_in_order].tolist(), area_pool_sha256=digest,
                               area_pool_counts=[len(g) for g in groups], area_band_edges=[.001, .01],
                               selected_area_bands=selected_bands))
    return selections


def plan_selection(rows, predictions, X, budgets, seeds, risk_pool_factor=None, area_stratified=False):
    require(bool(rows) and [r["sample_id"] for r in rows]==[r["sample_id"] for r in predictions],"selection row order differs")
    require(len({r["sample_id"] for r in rows})==len(rows),"duplicate selection IDs")
    require(all(r.get("preregistered_quality_role")=="candidate_pool" for r in rows),"only reserved candidate rows may be admitted")
    require(all(0<b<=1 for b in budgets) and len(set(budgets))==len(budgets),"invalid/duplicate budgets")
    require(bool(seeds) and len(set(seeds))==len(seeds),"empty/duplicate random seeds")
    require(risk_pool_factor is None or (type(risk_pool_factor) is int and risk_pool_factor >= 1),
            "risk pool factor must be a positive integer")
    require(X.shape==(len(rows),56) and np.isfinite(X).all(),"invalid selection features")
    augmented=[]
    for row,pred in zip(rows,predictions):
        require(bool(pred.get("pixel_sha256")),"RGB inference must include decoded-pixel hashes")
        augmented.append(dict(row,pixel_sha256=pred["pixel_sha256"],input_sha256={"rgb":pred["rgb_sha256"]}))
    eligible,reasons=eligible_rows(augmented)
    pool=np.flatnonzero(eligible);require(bool(len(pool)),"no eligible candidate frames")
    ids=np.array([r["sample_id"] for r in rows])
    risk=np.array([r["risk_score"] for r in predictions],dtype=np.float64)
    require(np.isfinite(risk).all() and ((risk>=0)&(risk<=1)).all(),"invalid risk scores")
    scores={"FrozenQualityRank":risk,"Confidence":-X[:,FEATURE_NAMES.index("global.lane_confidence")],
            "Entropy":X[:,FEATURE_NAMES.index("global.marking_entropy")]}
    rankings={name:pool[np.lexsort((ids[pool],-values[pool]))] for name,values in scores.items()}
    selections=[]
    for budget in budgets:
        count=math.ceil(budget*len(pool))
        for method,order in rankings.items():
            selections.append(dict(method=method,budget=budget,selected_count=count,seed=None,
                                   indices=order[:count].tolist(),scores=[float(scores[method][i]) for i in order[:count]]))
        for seed in seeds:
            order=np.random.default_rng(seed).permutation(pool)
            selections.append(dict(method="Random",budget=budget,selected_count=count,seed=seed,indices=order[:count].tolist(),scores=[None]*count))
        if risk_pool_factor is not None:
            selections.extend(risk_pool_selections(rankings["FrozenQualityRank"], ids, risk, X, count,
                                                   budget, seeds, risk_pool_factor))
        if area_stratified:
            selections.extend(area_stratified_selections(rankings["FrozenQualityRank"], ids, risk, X,
                                                         count, budget, seeds))
    return dict(pool_rows=len(pool),eligible=eligible,reasons=reasons,selections=selections,
                thinning=dict(minimum_elapsed_seconds=1.,minimum_displacement_m=5.,heading_change_degrees=10.,uses_GT=False),
                budget_rounding="ceil",score_ties="sample_id ascending",uses_GT=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True,help="Same RGB/lineage manifest used for inference; annotations are not opened")
    p.add_argument("--budget",type=float,action="append")
    p.add_argument("--random-seed",type=int,action="append")
    p.add_argument("--risk-pool-factor", type=int, help="also compare uniform and farthest-first sampling within this multiple of the risk budget")
    p.add_argument("--area-stratified", action="store_true", help="also rank or sample within three predicted lane-area bands")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();require(not a.output_dir.exists(),"selection output already exists")
    root=a.inference_dir.resolve();metadata=json.loads((root/"run.json").read_text())
    require(metadata.get("status")=="complete" and metadata.get("uses_ground_truth") is False,"completed RGB-only inference required")
    require((root/"_SUCCESS").read_text().strip()==sha256_file(root/"run.json"),"inference completion hash differs")
    for relative,digest in metadata["artifact_sha256"].items():
        path=(root/relative).resolve()
        try:path.relative_to(root)
        except ValueError as exc:raise ValueError("inference artifact escapes run") from exc
        require(sha256_file(path)==digest,"inference artifact changed: "+relative)
    require(metadata["input_manifest_sha256"]==sha256_file(a.manifest),"selection/inference manifest differs")
    require(json.loads((root/"schema.json").read_text())==feature_schema(),"selection feature schema differs")
    require(json.loads((root/"feature_names.json").read_text())==list(FEATURE_NAMES),"selection feature names differ")
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    predictions=[json.loads(s) for s in (root/"predictions.jsonl").read_text().splitlines() if s.strip()]
    require(json.loads((root/"sample_ids.json").read_text())==[r["sample_id"] for r in rows],"selection IDs differ")
    result=plan_selection(rows,predictions,np.load(root/"X.npy",allow_pickle=False),a.budget or [.05,.1,.2],
                          a.random_seed or [20261116], a.risk_pool_factor, a.area_stratified)
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    write_json(out/"pool.json",[dict(sample_id=row["sample_id"],eligible=keep,reason=reason)
               for row,keep,reason in zip(rows,result["eligible"],result["reasons"])])
    files=[]
    for selection in result["selections"]:
        fraction=format(selection["budget"],".12g").replace('.','p')
        suffix="_seed"+str(selection["seed"]) if selection["seed"] is not None else ""
        name=selection["method"]+"_budget"+fraction+suffix+".jsonl"
        pool_evidence = {k:v for k,v in selection.items() if k.startswith(("risk_pool_", "area_"))}
        with (out/name).open("x") as stream:
            for rank,(index,score) in enumerate(zip(selection["indices"],selection["scores"]),start=1):
                row=dict(rows[index],selection=dict(method=selection["method"],budget=selection["budget"],rank=rank,score=score,seed=selection["seed"],
                         inference_run_sha256=sha256_file(root/"run.json"),model_run_sha256=metadata["model_run_sha256"], **pool_evidence))
                if "selected_area_bands" in selection:
                    row["selection"]["predicted_area_band"] = selection["selected_area_bands"][rank - 1]
                stream.write(json.dumps(row,allow_nan=False)+"\n")
        files.append(dict(file=name,method=selection["method"],budget=selection["budget"],seed=selection["seed"],selected_rows=selection["selected_count"], **pool_evidence))
    report=dict(status="complete",schema_version="ml-quality-selection-1",source_rows=len(rows),pool_rows=result["pool_rows"],uses_GT=False,
                inference_run_sha256=sha256_file(root/"run.json"),input_manifest_sha256=sha256_file(a.manifest),
                model_run_sha256=metadata["model_run_sha256"],thinning=result["thinning"],budget_rounding="ceil",score_ties=result["score_ties"],
                exclusion_counts=dict(Counter(reason for keep,reason in zip(result["eligible"],result["reasons"]) if not keep)),outputs=files)
    write_json(out/"selection.json",report);(out/"selector_source.py").write_bytes(Path(__file__).read_bytes())
    hashes={f.name:sha256_file(f) for f in out.iterdir() if f.is_file()}
    write_json(out/"run.json",dict(status="complete",artifact_sha256=hashes));(out/"_SUCCESS").write_text(sha256_file(out/"run.json")+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
