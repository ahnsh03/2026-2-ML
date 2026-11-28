#!/usr/bin/env python3
"""Freeze GT-free review choices, then evaluate matched sigma/mean policies."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shlex
import shutil
import sys

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from quality_estimation.roi9_handoff import load_roi9_handoff, sha256
from quality_estimation.roi9_review import score_rows, replay_policy, low_quality_mask
from run_gp_search import save
from run_roi9_gp_search import predict_batches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("config/roi9_comparisons_v1.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.protocol.read_text())
    spec = config["review"]
    pack = load_roi9_handoff(args.handoff, config["recipe_sha256"])
    data, metadata = pack.arrays, pack.rows
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out/"source").mkdir()
    sources = [Path(__file__), Path("src/quality_estimation/roi9_review.py"),
               Path("src/quality_estimation/review_queue.py"), Path("src/quality_estimation/roi9_handoff.py")]
    for source in sources:
        shutil.copy2(source, out/"source"/source.name)
    shutil.copy2(args.protocol, out/"protocol.json")
    save(out/"input_receipt.json", dict(hashes=pack.hashes, source_sha256={s.name:sha256(s) for s in sources},
        spatial_cell_m=50.0, failure_comparison="float32 IoU < float32(0.7); equality is normal"))
    (out/"command.txt").write_text("PYTHONPATH=src "+shlex.join([sys.executable]+sys.argv)+"\n")
    with threadpool_limits(config["threads"]):
        rf = joblib.load(args.handoff/"models/quality_rf.joblib").set_params(n_jobs=1).predict(data["X"])
        gp = joblib.load(args.handoff/"models/quality_gp.joblib")
        mean, sigma = predict_batches(gp, data["X"], 512)
        with np.load(args.handoff/"models/environment.npz", allow_pickle=False) as env:
            scaled = data["RGB54"][:, env["selected"]].copy()
            scaled -= env["mean"]
            scaled /= env["scale"]
            labels = cdist(scaled, env["centers"], "sqeuclidean").argmin(axis=1)
            for role, key in (("train", "train_labels"), ("development", "dev_labels")):
                np.testing.assert_array_equal(labels[data["split"]==role], env[key])
    np.savez_compressed(out/"scores.npz", ids=data["ids"], rf=rf, gp_mean=mean, gp_sigma=sigma, cluster=labels)
    index = {sid:i for i,sid in enumerate(data["ids"])}
    arms = []
    # Selection is completed and hashed before any label-based policy evaluation.
    with (out/"selections.jsonl").open("w") as selections:
        for mode, cohorts in (("static",spec["static_cohorts"]),("windows",spec["window_cohorts"])):
            for cohort in cohorts:
                mask = (np.isin(data["source_cohort"], ["fresh", "fresh_clock_supplement"])
                        if cohort == "fresh_all" else data["source_cohort"]==cohort)
                rows = [metadata[i] for i in np.flatnonzero(mask)]
                for percent in spec["budgets_percent"]:
                    for seed in config["seeds"]:
                        for policy in spec["policies"]:
                            scored, audit, priority = score_rows(rows, rf[mask], mean[mask], sigma[mask], labels[mask], policy)
                            choices, population, windows = replay_policy(scored, mode, percent, seed, audit, priority, spec["window_frames"])
                            arm = dict(mode=mode, cohort=cohort, percent=percent, seed=seed, policy=policy,
                                       selected_ids=[r["sample_id"] for r,_ in choices],
                                       population_ids=[r["sample_id"] for r in population], windows=windows,
                                       reasons=dict(Counter(reason for _,reason in choices)))
                            arms.append(arm)
                            for row,reason in choices:
                                record = {k:arm[k] for k in ("mode","cohort","percent","seed","policy")}
                                selections.write(json.dumps(dict(record, sample_id=row["sample_id"], reason=reason))+"\n")
                print("Frozen selections:",mode,cohort,flush=True)
    save(out/"frozen_selection_receipt.json",dict(selections_sha256=sha256(out/"selections.jsonl"),
        scores_sha256=sha256(out/"scores.npz"), arms=len(arms), parameters_chosen_using_GT=False))
    save(out/"window_allocations.json", [{k:v for k,v in arm.items() if k not in ("selected_ids","population_ids")} for arm in arms])
    results, comparisons = [], []
    is_failure = low_quality_mask(data["y"], spec["failure_threshold"])
    for arm in arms:
        selected = np.array([index[sid] for sid in arm["selected_ids"]],dtype=int)
        population = np.array([index[sid] for sid in arm["population_ids"]],dtype=int)
        failure = is_failure[population]
        chosen_failure = is_failure[selected]
        found = int(chosen_failure.sum())
        failing = selected[chosen_failure]
        def cells(indices):
            return {(int(np.floor(float(metadata[i]["x"])/50)),int(np.floor(float(metadata[i]["y"])/50))) for i in indices}
        def cohort_record(indices):
            return len({metadata[i]["site"] for i in indices})
        record = {k:arm[k] for k in ("mode","cohort","percent","seed","policy")}
        record.update(population=len(population), actual_budget=len(selected), actual_budget_fraction=len(selected)/len(population),
            total_failures=int(failure.sum()), found=found,
            precision=found/len(selected), recall=found/int(failure.sum()) if failure.any() else None,
            unique_failure_site_records=cohort_record(failing), unique_failure_50m_cells=len(cells(failing)),
            selected_mean_iou=float(data["y"][selected].mean()))
        results.append(record)
    by_key = {(a["mode"],a["cohort"],a["percent"],a["seed"],a["policy"]):a for a in arms}
    result_key = {(r["mode"],r["cohort"],r["percent"],r["seed"],r["policy"]):r for r in results}
    for arm in arms:
        if arm["policy"] != "gp_sigma_audit":
            continue
        key = (arm["mode"],arm["cohort"],arm["percent"],arm["seed"])
        base = by_key[(*key,"gp_mean_audit")]
        extra, removed = set(arm["selected_ids"])-set(base["selected_ids"]),set(base["selected_ids"])-set(arm["selected_ids"])
        comparisons.append(dict(mode=key[0],cohort=key[1],percent=key[2],seed=key[3],
            added_ids=sorted(extra), removed_ids=sorted(removed),
            added_failures=sum(bool(is_failure[index[sid]]) for sid in extra),
            removed_failures=sum(bool(is_failure[index[sid]]) for sid in removed),
            found_delta=result_key[(*key,"gp_sigma_audit")]["found"]-result_key[(*key,"gp_mean_audit")]["found"]))
    # Assert every compared policy had the identical population, budget and chronological allocations.
    groups = defaultdict(list)
    for arm in arms:
        groups[arm["mode"],arm["cohort"],arm["percent"],arm["seed"]].append(arm)
    for group in groups.values():
        reference = group[0]
        for arm in group[1:]:
            assert arm["population_ids"] == reference["population_ids"]
            assert len(arm["selected_ids"]) == len(reference["selected_ids"])
            assert [(w["run_id"],w["window"],w["budget"]) for w in arm["windows"]] == [(w["run_id"],w["window"],w["budget"]) for w in reference["windows"]]
    save(out/"summary.json", dict(results=results, primary_sigma_comparisons=comparisons,
        diagnostic_only=True, policies_tuned_using_labels=False,
        source_site_records_may_include_run_prefixes=True, physical_diversity_uses_fixed_50m_cells=True))
    save(out/"sigma_changes.json", comparisons)
    with (out/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    save(out/"artifacts.json", {str(f.relative_to(out)):sha256(f) for f in out.rglob("*") if f.is_file()})
    (out/"_SUCCESS").write_text(sha256(out/"artifacts.json")+"\n")
    print("Review comparison complete:",len(results),"arms",flush=True)


if __name__ == "__main__":
    main()
