#!/usr/bin/env python3
"""Audit historical X56 input, reproduce GP, and compare matched random/BO searches.

Example (input paths are read-only; output must not exist):
  PYTHONPATH=src python3 scripts/run_gp_search.py --dataset-dir DATASET \
    --split SPLIT.json --reference-run OLD_GP_RUN --output-dir NEW_RUN
Latest roi9 data needs an explicit schema adapter, not an X56 relabeling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.spatial.distance import cdist
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from quality_estimation.dataset_io import load_dataset
from quality_estimation.splits import read_split
from quality_estimation.gaussian_process import fit_quality_gp
from quality_estimation.gp_search import (purged_folds, metrics, evaluate_cv,
                                         search_space, shared_start, next_point)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def export_predictions(path, ids, y, mean, std):
    radius = norm.ppf(.95)*std
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_iou", "raw_mean", "predicted_iou", "observation_std", "lower90_raw", "upper90_raw"])
        writer.writerows(zip(ids, y, mean, np.clip(mean, 0, 1), std, mean-radius, mean+radius))


def stratified_metrics(samples, y, mean, std):
    result = {}
    for label in ("parent_session_id", "weather", "hour"):
        values = [str(s.get("condition", {}).get(label, "unknown")) if label != "parent_session_id"
                  else s[label] for s in samples]
        result[label] = {}
        for value in sorted(set(values)):
            mask = np.array(values) == value
            result[label][value] = metrics(y[mask], mean[mask], std[mask])
    return result


def plot_uncertainty(path, y, mean, std, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].scatter(y, np.clip(mean, 0, 1), s=9, alpha=.5)
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="Actual IoU", ylabel="Predicted IoU", title="Point prediction")
    axes[1].scatter(std, abs(mean-y), s=9, alpha=.5)
    axes[1].set(xlabel="Observation standard deviation", ylabel="Absolute raw error", title="Uncertainty vs error")
    levels = np.array([.5, .7, .8, .9, .95])
    cover = [np.mean(abs(mean-y) <= norm.ppf((1+c)/2)*std) for c in levels]
    axes[2].plot(levels, cover, "o-")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set(xlabel="Nominal coverage", ylabel="Observed coverage", title="Uncalibrated intervals", ylim=(0, 1))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("config/gp_search_v1.json"))
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    points, coordinates, baseline_index = search_space(protocol)
    if not 1 <= protocol["shared_initial"] <= protocol["budget"] <= len(points):
        raise ValueError("invalid search budget")
    data = load_dataset(args.dataset_dir)
    split = read_split(args.split, data)
    indices = {role: np.array([i for i, a in enumerate(split["assignments"])
                              if a["role"] == role and a["eligible"] and data.defined[i]], dtype=int)
               for role in ("meta_train", "meta_dev")}
    ti, di = indices["meta_train"], indices["meta_dev"]
    if len(ti) < 2 or not len(di):
        raise ValueError("missing supervised train/development rows")
    # Match train_models' float64 target normalization as well as its input dtype.
    x, y = data.X[ti].astype(np.float64), data.y[ti].astype(np.float64)
    dx, dy = data.X[di].astype(np.float64), data.y[di].astype(np.float64)
    rows = [data.samples[i] for i in ti]
    drows = [data.samples[i] for i in di]
    xy = np.array([[r["ego"]["position"][k] for k in ("x", "y")] for r in rows])
    dxy = np.array([[r["ego"]["position"][k] for k in ("x", "y")] for r in drows])
    groups = np.array([r["parent_session_id"] for r in rows])
    station = np.array([r["route_station_m"] for r in rows])
    folds = purged_folds(groups, xy, station, protocol["spatial_bins"], protocol["purge_distance_m"])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out/"models").mkdir()
    (out/"predictions").mkdir()
    (out/"figures").mkdir()
    (out/"source").mkdir()
    sources = [Path(__file__).resolve(), Path("src/quality_estimation/gp_search.py"),
               Path("src/quality_estimation/gaussian_process.py"), Path("src/quality_estimation/dataset_io.py"),
               Path("src/quality_estimation/splits.py")]
    for source in sources:
        shutil.copy2(source, out/"source"/source.name)
    save(out/"protocol.json", protocol)
    (out/"command.txt").write_text("PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
                                  +shlex.join([sys.executable]+sys.argv)+"\n")
    train_pixels = {r["pixel_sha256"] for r in rows}
    dev_pixels = {r["pixel_sha256"] for r in drows}
    if train_pixels & dev_pixels:
        raise ValueError("source pixels overlap")
    near = cdist(dxy, xy).min(axis=1)
    audit = dict(dataset_metadata_sha256=digest(args.dataset_dir/"dataset.json"),
                 split_sha256=digest(args.split), protocol_sha256=digest(args.protocol),
                 dataset_path=str(args.dataset_dir.resolve()), split_path=str(args.split.resolve()),
                 n_train=len(ti), n_development_diagnostic=len(di), feature_count=x.shape[1],
                 ordered_features=data.names, undefined_excluded={role: sum(
                     a["role"] == role and a["eligible"] and not data.defined[i]
                     for i, a in enumerate(split["assignments"])) for role in indices},
                 train_parent_groups=sorted(set(groups)),
                 development_parent_groups=sorted({r["parent_session_id"] for r in drows}),
                 pixel_overlap=0, nearest_train_dev_m=float(near.min()),
                 development_within_20m_of_training=int((near < 20).sum()),
                 limitation="Historical same-map, position-overlapping development only. No independent road/generalization claim. No final-test or calibration-role rows used.",
                 source_sha256={s.name: digest(s) for s in sources},
                 git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 sklearn_version=sklearn.__version__)
    save(out/"input_audit.json", audit)
    save(out/"membership.json", {role: [dict(sample_id=data.ids[i], pixel_sha256=data.samples[i]["pixel_sha256"],
               parent_session_id=data.samples[i]["parent_session_id"]) for i in idx] for role, idx in indices.items()})
    save(out/"folds.json", [dict(name=n, train=t.tolist(), valid=v.tolist(),
                              minimum_distance_m=float(cdist(xy[t], xy[v]).min())) for n,t,v in folds])

    # All choices are frozen using training-only CV before development predictions.
    with threadpool_limits(limits=protocol["threads"]):
        baseline_cv, bm, bs = evaluate_cv(x, y, folds, protocol["baseline"])
        export_predictions(out/"predictions/baseline_oof.csv", np.array(data.ids)[ti], y, bm, bs)
        summary = [dict(method="baseline", seed=None, parameters=protocol["baseline"], **baseline_cv)]
        histories = []
        started = time.perf_counter()
        for seed in protocol["seeds"]:
            initial = shared_start(len(points), baseline_index, seed, protocol["shared_initial"])
            for method in ("random", "bo"):
                history = []
                method_start = time.perf_counter()
                for step in range(protocol["budget"]):
                    propose_start = time.perf_counter()
                    index = next_point(coordinates, history, initial, method, seed, protocol["bo"])
                    proposal_seconds = time.perf_counter()-propose_start
                    settings = dict(zip(("length_multiplier", "noise_variance"), points[index].tolist()))
                    scores, _, _ = evaluate_cv(x, y, folds, settings)
                    history.append(dict(step=step+1, index=index, parameters=settings,
                                        proposal_seconds=proposal_seconds, **scores))
                    save(out/"current_search.json", dict(method=method, seed=seed, history=history))
                    print(method, seed, step+1, round(scores["cv_mae"], 6), flush=True)
                best = min(history, key=lambda r: (r["cv_mae"], r["step"]))
                summary.append(dict(method=method, seed=seed, parameters=best["parameters"], cv_mae=best["cv_mae"],
                                    seconds=time.perf_counter()-method_start))
                histories.append(dict(method=method, seed=seed, initial_indices=initial, trials=history))
                save(out/"histories.json", histories)
        save(out/"frozen_choices.json", summary)
        print("Choices frozen; evaluating historical development diagnostics", flush=True)
        for result in summary:
            name = result["method"]+("" if result["seed"] is None else "_"+str(result["seed"]))
            start = time.perf_counter()
            model = fit_quality_gp(x, y, **result["parameters"])
            result["full_fit_seconds"] = time.perf_counter()-start
            path = out/"models"/(name+".joblib")
            joblib.dump(model, path, compress=2)
            start = time.perf_counter()
            mean, std = model.predict(dx, return_std=True)
            result["batch_prediction_seconds"] = time.perf_counter()-start
            reloaded = joblib.load(path)
            rm, rs = reloaded.predict(dx, return_std=True)
            np.testing.assert_allclose(mean, rm, rtol=0, atol=1e-12)
            np.testing.assert_allclose(std, rs, rtol=0, atol=1e-12)
            result.update(reload_max_mean_difference=float(np.max(abs(mean-rm))),
                          reload_max_std_difference=float(np.max(abs(std-rs))),
                          model_sha256=digest(path), development_diagnostic=metrics(dy, mean, std))
            export_predictions(out/"predictions"/(name+".csv"), np.array(data.ids)[di], dy, mean, std)
            save(out/(name+"_by_condition.json"), stratified_metrics(drows, dy, mean, std))
            if result["method"] == "baseline":
                result["oof"] = metrics(y, bm, bs)
                save(out/"baseline_oof_by_condition.json", stratified_metrics(rows, y, bm, bs))
                plot_uncertainty(out/"figures/baseline_development.png", dy, mean, std, "Historical X56 development (position overlap)")
                plot_uncertainty(out/"figures/baseline_spatial_oof.png", y, bm, bs, "Historical X56: spatial and parent purged OOF")
                if args.reference_run:
                    reference = args.reference_run/"models/GaussianProcess.joblib"
                    recorded = json.loads((args.reference_run/"run.json").read_text())
                    if digest(reference) != recorded["artifact_sha256"]["models/GaussianProcess.joblib"]:
                        raise ValueError("archived reference hash mismatch")
                    comparison = json.loads((args.reference_run/"comparison.json").read_text())
                    if (comparison["dataset_metadata_sha256"] != audit["dataset_metadata_sha256"]
                            or comparison["split_sha256"] != audit["split_sha256"]
                            or comparison["ordered_features"] != data.names):
                        raise ValueError("archived reference input/split/schema mismatch")
                    bundle = joblib.load(reference)
                    old_mean, old_std = bundle["estimator"].predict(dx[:, bundle["column_indices"]], return_std=True)
                    result["archived_reference"] = dict(model_sha256=digest(reference),
                        max_mean_difference=float(np.max(abs(mean-old_mean))),
                        max_std_difference=float(np.max(abs(std-old_std))))
                    np.testing.assert_allclose(mean, old_mean, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(std, old_std, rtol=0, atol=1e-10)
        save(out/"summary.json", dict(results=summary, elapsed_search_and_diagnostics_seconds=time.perf_counter()-started,
                                    development_is_diagnostic=True, calibration_fitted=False, final_test_accessed=False))
    columns = ["method", "seed", "cv_mae", "development_mae", "coverage90", "width90", "seconds", "length_multiplier", "noise_variance"]
    with (out/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in summary:
            writer.writerow(dict(method=r["method"], seed=r["seed"], cv_mae=r["cv_mae"],
                                 development_mae=r["development_diagnostic"]["clipped_mae"],
                                 coverage90=r["development_diagnostic"]["coverage90_raw"],
                                 width90=r["development_diagnostic"]["width90_raw"], seconds=r["seconds"], **r["parameters"]))
    report = ["GP reproduction and paired search: historical X56", "",
              "Training: 824 eligible rows, 3 parents. Historical development: 576 rows, 2 parents.",
              "The input audit contains actual counts; 27-column roi9 performance is NOT measured here.",
              "Development is previously inspected and position-overlapping; not an independent final test.",
              "CV holds out parent x route bin and purges training neighbors within 20m.",
              "BO and random search each evaluate 20 settings including identical first 4, over 3 seeds.",
              "The acquisition is expected improvement on a finite log grid, with a separate fixed-kernel GP surrogate.",
              "No trial cache: measured search time includes actual repeated fits and proposal overhead.",
              "", "Method | mean selected spatial CV MAE | mean historical development MAE | search seconds/seed"]
    for method in ("baseline", "random", "bo"):
        selected = [r for r in summary if r["method"] == method]
        report.append("{} | {:.6f} | {:.6f} | {:.3f}".format(method,
            np.mean([r["cv_mae"] for r in selected]),
            np.mean([r["development_diagnostic"]["clipped_mae"] for r in selected]),
            np.mean([r["seconds"] for r in selected])))
    report += ["", "Interpretation:",
               "MAE optimization does not optimize interval calibration; inspect coverage and width separately.",
               "No interval calibration was fitted. Fixed-0.7 alarm counts are diagnostics, not calibrated policies.",
               "Three search seeds measure optimizer variability on ONE dataset, not independent road repetitions.",
               "No model is approved for deployment by this run.",
               "", "Files: summary.csv/json, input_audit.json, membership.json, folds.json, histories.json,",
               "frozen_choices.json, predictions/, models/, figures/, source/, command.txt, artifacts.json.",
               "Reproduction: run command.txt from repository root with a NEW --output-dir."]
    (out/"REPORT.txt").write_text("\n".join(report)+"\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, color in (("random", "tab:orange"), ("bo", "tab:blue")):
        curves = np.array([np.minimum.accumulate([t["cv_mae"] for t in h["trials"]]) for h in histories if h["method"] == method])
        steps = np.arange(1, protocol["budget"]+1)
        ax.plot(steps, curves.mean(axis=0), label=method, color=color)
        ax.fill_between(steps, curves.min(axis=0), curves.max(axis=0), color=color, alpha=.15)
    ax.axhline(baseline_cv["cv_mae"], color="black", linestyle="--", label="existing setting")
    ax.set(xlabel="Evaluated settings (shared initial 4 included)", ylabel="Best spatial CV MAE", title="Historical X56 GP: mean and range over search seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"figures/search_curves.png", dpi=150)
    plt.close(fig)
    files = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
    save(out/"artifacts.json", files)
    (out/"_SUCCESS").write_text(digest(out/"artifacts.json")+"\n")
    print(json.dumps(dict(output=str(out), baseline=summary[0]["development_diagnostic"], completed=True)), flush=True)


if __name__ == "__main__":
    main()
