#!/usr/bin/env python3
"""Paired RF baseline/random/BO comparison using the frozen ROI9 handoff folds."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from quality_estimation.gp_search import next_point, shared_start
from quality_estimation.rf_search import rf_space, fit_rf, evaluate_rf
from quality_estimation.roi9_handoff import load_roi9_handoff, sha256
from run_gp_search import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("config/roi9_comparisons_v1.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.protocol.read_text())
    spec = config["rf"]
    pack = load_roi9_handoff(args.handoff, config["recipe_sha256"])
    if spec["baseline"] != pack.recipe["tasks"]["quality_rf"]["parameters"]:
        raise ValueError("RF baseline differs from handoff recipe")
    data = pack.arrays
    train = data["split"] == "train"
    x, y = data["X"][train], data["y"][train]
    folds, fold_audit = pack.training_folds()
    choices, coords, base = rf_space(spec)
    if not 1 <= spec["shared_initial"] <= spec["budget"] <= len(choices):
        raise ValueError("invalid RF budget")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    for d in ("models", "source", "predictions"):
        (out/d).mkdir()
    sources = [Path(__file__), Path("src/quality_estimation/rf_search.py"),
               Path("src/quality_estimation/gp_search.py"), Path("src/quality_estimation/roi9_handoff.py")]
    for source in sources:
        shutil.copy2(source, out/"source"/source.name)
    shutil.copy2(args.protocol, out/"protocol.json")
    save(out/"folds.json", fold_audit)
    save(out/"input_receipt.json", dict(hashes=pack.hashes, source_sha256={s.name:sha256(s) for s in sources},
        protocol_sha256=sha256(args.protocol), versions={n:importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "scipy")}))
    (out/"command.txt").write_text("PYTHONPATH=src "+shlex.join([sys.executable]+sys.argv)+"\n")
    start = time.perf_counter()
    with threadpool_limits(config["threads"]):
        baseline = fit_rf(x, y, spec["baseline"], spec["model_seed"])
        archived = joblib.load(args.handoff/"models/quality_rf.joblib").set_params(n_jobs=1)
        actual = baseline.predict(data["X"])
        expected = archived.predict(data["X"])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        save(out/"baseline_reproduction.json", dict(max_prediction_difference=float(np.max(abs(actual-expected)))))
        score, oof = evaluate_rf(x, y, folds, spec["baseline"], spec["model_seed"])
        np.savez_compressed(out/"predictions/baseline_oof.npz", ids=data["ids"][train], y=y, predicted=oof)
        results = [dict(method="baseline", seed=None, parameters=spec["baseline"], **score)]
        histories = []
        for seed in config["seeds"]:
            initial = shared_start(len(choices), base, seed, spec["shared_initial"])
            for method in ("random", "bo"):
                history, started = [], time.perf_counter()
                for step in range(spec["budget"]):
                    begin = time.perf_counter()
                    index = next_point(coords, history, initial, method, seed, spec["bo"])
                    proposal_seconds = time.perf_counter()-begin
                    score, _ = evaluate_rf(x, y, folds, choices[index], spec["model_seed"])
                    history.append(dict(step=step+1, index=index, parameters=choices[index], proposal_seconds=proposal_seconds, **score))
                    save(out/"current_search.json", dict(method=method, seed=seed, trials=history))
                    print(method, seed, step+1, round(score["cv_mae"], 6), flush=True)
                best = min(history, key=lambda r:(r["cv_mae"], r["step"]))
                results.append(dict(method=method, seed=seed, parameters=best["parameters"], cv_mae=best["cv_mae"],
                                    seconds=time.perf_counter()-started))
                histories.append(dict(method=method, seed=seed, initial_indices=initial, trials=history))
                save(out/"histories.json", histories)
        save(out/"frozen_choices.json", results)
        for r in results:
            name = r["method"]+("" if r["seed"] is None else "_"+str(r["seed"]))
            model = fit_rf(x, y, r["parameters"], spec["model_seed"])
            file = out/"models"/(name+".joblib")
            joblib.dump(model, file, compress=2)
            predicted = np.clip(model.predict(data["X"]), 0, 1)
            np.testing.assert_allclose(predicted, joblib.load(file).predict(data["X"]), rtol=0, atol=1e-12)
            r["model_sha256"] = sha256(file)
            np.savez_compressed(out/"predictions"/(name+".npz"), ids=data["ids"], predicted=predicted)
            masks = {str(c): data["source_cohort"]==c for c in np.unique(data["source_cohort"])}
            masks["fresh_all"] = np.isin(data["source_cohort"], ["fresh", "fresh_clock_supplement"])
            r["cohorts"] = {name:dict(n=int(mask.sum()), mae=float(np.mean(abs(predicted[mask]-data["y"][mask]))),
                bias=float(np.mean(predicted[mask]-data["y"][mask]))) for name,mask in masks.items()}
    save(out/"summary.json", dict(results=results, elapsed_seconds=time.perf_counter()-start,
        feature_selection_repeated=False, diagnostic_only=True, calibration_fitted=False))
    save(out/"artifacts.json", {str(f.relative_to(out)):sha256(f) for f in out.rglob("*") if f.is_file()})
    (out/"_SUCCESS").write_text(sha256(out/"artifacts.json")+"\n")
    print("RF comparison complete", flush=True)


if __name__ == "__main__":
    main()
