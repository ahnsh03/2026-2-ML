#!/usr/bin/env python3
"""Reproduce the frozen ROI9 v2 GP and run paired train-only CV searches.

Use the handoff Python 3.12 environment and an output directory that does not exist.
No calibration or previously untouched final evaluation is available in this pack.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from quality_estimation.gaussian_process import fit_quality_gp
from quality_estimation.gp_search import evaluate_cv, metrics, next_point, search_space, shared_start
from quality_estimation.roi9_handoff import load_roi9_handoff, sha256
from run_gp_search import export_predictions, plot_uncertainty, save


def predict_batches(model, x, batch_rows):
    if batch_rows < 1 or not len(x):
        raise ValueError("positive batch size and nonempty input required")
    pieces = [model.predict(x[i:i+batch_rows], return_std=True) for i in range(0, len(x), batch_rows)]
    return tuple(np.concatenate([p[k] for p in pieces]) for k in (0, 1))


def cohort_masks(data):
    result = {str(c): data["source_cohort"] == c for c in np.unique(data["source_cohort"])}
    result["fresh_all"] = np.isin(data["source_cohort"], ["fresh", "fresh_clock_supplement"])
    return {key: mask for key, mask in result.items() if mask.any()}


def breakdown(rows, y, mean, std):
    result = {}
    for field in ("group", "road_link", "weather", "hour"):
        values = np.array([r[field] or "unavailable" for r in rows])
        result[field] = {str(value): metrics(y[mask], mean[mask], std[mask])
                         for value in np.unique(values) if (mask := values == value).any()}
    return result


def run(handoff, protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    pack = load_roi9_handoff(handoff, protocol["recipe_sha256"])
    data, rows, recipe = pack.arrays, pack.rows, pack.recipe
    if recipe["tasks"]["quality_gp"]["parameters"] != protocol["baseline"]:
        raise ValueError("baseline must reproduce the handoff recipe")
    points, coordinates, base_index = search_space(protocol)
    if not 1 <= protocol["shared_initial"] <= protocol["budget"] <= len(points):
        raise ValueError("invalid search budget")
    if len(protocol["seeds"]) != len(set(protocol["seeds"])):
        raise ValueError("duplicate search seeds")
    train = data["split"] == "train"
    # Preserve float32 labels: the archived model normalized targets in float32.
    x, y = data["X"][train], data["y"][train]
    folds, fold_audit = pack.training_folds()
    xy = np.array([[float(r[k]) for k in ("x", "y")] for r in rows if r["split"] == "train"])
    dxy = np.array([[float(r[k]) for k in ("x", "y")] for r in rows if r["split"] == "development"])
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("models", "predictions", "figures", "source"):
        (output/directory).mkdir()
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), root/"scripts/run_gp_search.py",
                    root/"src/quality_estimation/gaussian_process.py", root/"src/quality_estimation/gp_search.py",
                    root/"src/quality_estimation/roi9_handoff.py"]
    for path in source_files:
        shutil.copy2(path, output/"source"/path.name)
    shutil.copy2(protocol_path, output/"protocol.json")
    shutil.copy2(handoff/"code/config/region_feature_recipe_v2.json", output/"recipe.json")
    shutil.copy2(handoff/"frames.csv", output/"membership.csv")
    (output/"command.txt").write_text("PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
                                     +shlex.join([sys.executable]+sys.argv)+"\n")
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib", "threadpoolctl")}
    audit = dict(handoff_path=str(handoff.resolve()), checksums_verified=len(pack.hashes),
                 input_hashes=pack.hashes, protocol_sha256=sha256(protocol_path),
                 recipe_sha256=sha256(output/"recipe.json"),
                 feature_count=x.shape[1], ordered_features=data["quality_columns"].tolist(),
                 n_train=int(train.sum()), n_development=int((data["split"] == "development").sum()),
                 n_diagnostic=int((data["split"] == "diagnostic").sum()),
                 n_new_calibration=0, n_untouched_final=0,
                 train_development_pixel_overlap=0, nearest_train_development_m=float(cdist(xy, dxy).min()),
                 folds_sha256=None, versions=versions, source_sha256={p.name: sha256(p) for p in source_files},
                 git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                 limitation=protocol["limitation"], evaluation_use=protocol["evaluation_use"])
    save(output/"folds.json", fold_audit)
    audit["folds_sha256"] = sha256(output/"folds.json")
    save(output/"input_audit.json", audit)
    save(output/"status.json", dict(status="running", stage="baseline_reproduction"))
    started = time.perf_counter()
    with threadpool_limits(protocol["threads"]):
        baseline = fit_quality_gp(x, y, **protocol["baseline"])
        archive = joblib.load(handoff/"models/quality_gp.joblib")
        bm, bs = predict_batches(baseline, x, protocol["prediction_batch_rows"])
        am, ast = predict_batches(archive, x, protocol["prediction_batch_rows"])
        np.testing.assert_allclose(bm, am, rtol=0, atol=2e-8)
        np.testing.assert_allclose(bs, ast, rtol=0, atol=2e-8)
        save(output/"baseline_reproduction.json", dict(
            archived_model_sha256=sha256(handoff/"models/quality_gp.joblib"),
            max_training_mean_difference=float(np.max(abs(bm-am))),
            max_training_std_difference=float(np.max(abs(bs-ast)))))
        print("Baseline training predictions reproduce archived ROI9 GP", flush=True)
        baseline_cv, om, os = evaluate_cv(x, y, folds, protocol["baseline"])
        export_predictions(output/"predictions/baseline_oof.csv", data["ids"][train], y, om, os)
        plot_uncertainty(output/"figures/baseline_oof.png", y, om, os, "ROI9 v2: fixed-feature training OOF")
        summary = [dict(method="baseline", seed=None, parameters=protocol["baseline"],
                        oof=metrics(y, om, os), **baseline_cv)]
        histories = []
        save(output/"status.json", dict(status="running", stage="train_only_search"))
        for seed in protocol["seeds"]:
            initial = shared_start(len(points), base_index, seed, protocol["shared_initial"])
            for method in ("random", "bo"):
                history, begin = [], time.perf_counter()
                for step in range(protocol["budget"]):
                    begin_proposal = time.perf_counter()
                    index = next_point(coordinates, history, initial, method, seed, protocol["bo"])
                    proposal_seconds = time.perf_counter()-begin_proposal
                    settings = dict(zip(("length_multiplier", "noise_variance"), points[index].tolist()))
                    score, _, _ = evaluate_cv(x, y, folds, settings)
                    history.append(dict(step=step+1, index=index, parameters=settings,
                                        proposal_seconds=proposal_seconds, **score))
                    save(output/"current_search.json", dict(seed=seed, method=method, trials=history))
                    print(method, seed, step+1, round(score["cv_mae"], 6), flush=True)
                best = min(history, key=lambda r: (r["cv_mae"], r["step"]))
                summary.append(dict(method=method, seed=seed, parameters=best["parameters"],
                                    cv_mae=best["cv_mae"], seconds=time.perf_counter()-begin))
                histories.append(dict(method=method, seed=seed, initial_indices=initial, trials=history))
                save(output/"histories.json", histories)
        # This file is written before any non-training predictions or score comparisons.
        save(output/"frozen_choices.json", summary)
        print("Choices frozen; beginning non-training diagnostics", flush=True)
        save(output/"status.json", dict(status="running", stage="diagnostics_choices_frozen"))
        masks = cohort_masks(data)
        reference_metrics = json.loads((handoff/"evidence/quality_v2_reference.json").read_text())
        for result in summary:
            name = result["method"]+("" if result["seed"] is None else "_"+str(result["seed"]))
            begin = time.perf_counter()
            model = fit_quality_gp(x, y, **result["parameters"])
            result["full_fit_seconds"] = time.perf_counter()-begin
            file = output/"models"/(name+".joblib")
            joblib.dump(model, file, compress=2)
            begin = time.perf_counter()
            mean, std = predict_batches(model, data["X"], protocol["prediction_batch_rows"])
            result["all_rows_predict_seconds"] = time.perf_counter()-begin
            rm, rs = predict_batches(joblib.load(file), data["X"], protocol["prediction_batch_rows"])
            np.testing.assert_allclose(mean, rm, rtol=0, atol=1e-12)
            np.testing.assert_allclose(std, rs, rtol=0, atol=1e-12)
            result.update(model_sha256=sha256(file), reload_max_mean_difference=float(np.max(abs(mean-rm))),
                          reload_max_std_difference=float(np.max(abs(std-rs))))
            export_predictions(output/"predictions"/(name+".csv"), data["ids"], data["y"], mean, std)
            result["cohorts"] = {cohort: metrics(data["y"][mask], mean[mask], std[mask])
                                 for cohort, mask in masks.items()}
            # Training scores above are in-sample and are never substituted for OOF CV scores.
            result["cohorts"]["train"]["use"] = "in_sample_diagnostic_only"
            dev = data["split"] == "development"
            save(output/(name+"_development_breakdown.json"),
                 breakdown([r for r in rows if r["split"] == "development"], data["y"][dev], mean[dev], std[dev]))
            if result["method"] == "baseline":
                archived_mean, archived_std = predict_batches(archive, data["X"], protocol["prediction_batch_rows"])
                np.testing.assert_allclose(mean, archived_mean, rtol=0, atol=2e-8)
                np.testing.assert_allclose(std, archived_std, rtol=0, atol=2e-8)
                result["archived_all_rows_max_mean_difference"] = float(np.max(abs(mean-archived_mean)))
                result["archived_all_rows_max_std_difference"] = float(np.max(abs(std-archived_std)))
                for cohort, reference in reference_metrics.items():
                    mask = masks[cohort]
                    actual = result["cohorts"][cohort]
                    np.testing.assert_allclose(actual["clipped_mae"], reference["GP"]["clipped_mae"], atol=1e-8, rtol=0)
                    handoff_coverage = float(np.mean(abs(mean[mask]-data["y"][mask]) <= protocol["handoff_reference_quantile"]*std[mask]))
                    np.testing.assert_allclose(handoff_coverage, reference["GP"]["coverage_90_unadjusted"], atol=1e-12, rtol=0)
                    if actual["n"] != reference["GP"]["n"]:
                        raise ValueError("cohort count differs from handoff")
                for cohort in ("development", "fresh_all", "fresh_uncertainty"):
                    mask = masks[cohort]
                    plot_uncertainty(output/"figures"/("baseline_"+cohort+".png"),
                                     data["y"][mask], mean[mask], std[mask], "ROI9 v2: "+cohort+" (diagnostic)")
            print("Diagnostics saved:", name, flush=True)
    save(output/"summary.json", dict(results=summary, elapsed_seconds=time.perf_counter()-started,
                                    diagnostics_only=True, calibration_fitted=False, untouched_final_rows=0))
    fields = ["method", "seed", "cohort", "n", "cv_mae", "clipped_mae", "bias", "coverage90_raw", "width90_raw",
              "sigma_absolute_error_spearman", "length_multiplier", "noise_variance"]
    with (output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in summary:
            for cohort, scores in r["cohorts"].items():
                writer.writerow(dict(method=r["method"], seed=r["seed"], cohort=cohort,
                                     cv_mae=r["cv_mae"], **r["parameters"], **scores))
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, color in (("random", "tab:orange"), ("bo", "tab:blue")):
        curves = np.array([np.minimum.accumulate([t["cv_mae"] for t in h["trials"]])
                           for h in histories if h["method"] == method])
        steps = np.arange(1, protocol["budget"]+1)
        ax.plot(steps, curves.mean(axis=0), color=color, label=method)
        ax.fill_between(steps, curves.min(axis=0), curves.max(axis=0), color=color, alpha=.15)
    ax.axhline(baseline_cv["cv_mae"], color="black", linestyle="--", label="frozen GP v2")
    ax.set(xlabel="Evaluated settings (shared initialization included)", ylabel="Best fixed-feature CV MAE",
           title="ROI9 v2 GP: optimizer mean and range over 3 seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"figures/search_curves.png", dpi=150)
    plt.close(fig)
    report = ["ROI9 v2 GP reproduction and matched BO/random search", "",
              f"Training {len(x)} rows; features {x.shape[1]}; provided CV folds {len(folds)}.",
              "Existing recipe reproduced before search; X and y retain handoff float32 dtype.",
              "Every arm uses only training-fold MAE, identical first four points, 20 trials, 3 seeds.",
              "All choices frozen before non-training scores; no cross-arm evaluation cache.",
              "", "Method | mean CV MAE | development MAE | fresh480 MAE | extra144 MAE"]
    for method in ("baseline", "random", "bo"):
        rr = [r for r in summary if r["method"] == method]
        values = [np.mean([r["cv_mae"] for r in rr])]+[
            np.mean([r["cohorts"][c]["clipped_mae"] for r in rr])
            for c in ("development", "fresh_all", "fresh_uncertainty")]
        report.append(method+" | "+" | ".join(f"{v:.6f}" for v in values))
    report += ["", "Limits:", protocol["limitation"], protocol["evaluation_use"],
               "Nominal 90% intervals use raw means and normal quantile 1.6448536269514722.",
               "Reference reproduction separately checks the handoff's 1.645 quantile.",
               "MAE selection does not calibrate uncertainty; compare coverage and interval width.",
               "Do not select a new setting using the diagnostic cohorts. No deployment decision is made.",
               "Model files are research pipelines with recipe/protocol sidecars, not stream deployment bundles.",
               "", "Read summary.csv, figures/, input_audit.json, folds.json and frozen_choices.json.",
               "Per-frame predictions include training in-sample scores; use baseline_oof.csv for OOF training errors.",
               "membership.csv maps prediction IDs to roles, original cohorts and metadata.",
               "Reproduction: command.txt from repository root, with a NEW output directory."]
    (output/"REPORT.txt").write_text("\n".join(report)+"\n")
    save(output/"status.json", dict(status="complete", stage="diagnostics_complete"))
    save(output/"artifacts.json", {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    (output/"_SUCCESS").write_text(sha256(output/"artifacts.json")+"\n")
    print("Complete:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("config/gp_search_roi9_v2.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.handoff, args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
