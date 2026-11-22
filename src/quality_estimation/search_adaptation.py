"""Suggest the next development-only perception adaptation trial from measured runs."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from .cycle_gate import completed_run
from .dataset_io import require


def search_grid(design):
    rates, replay, weights = (design[key] for key in ("lr", "replay", "lam"))
    require(all(len(v) > 0 and len(set(v)) == len(v) for v in (rates, replay, weights)), "empty or duplicate axis")
    require(all(np.isfinite(v) and v > 0 for v in rates), "invalid learning rate")
    require(all(isinstance(v, int) and 0 < v < design["batch"] for v in replay), "invalid replay slots")
    require(all(np.isfinite(v) and v >= 0 for v in weights), "invalid distillation weight")
    grid = list(itertools.product(rates, replay, weights))
    initial = [grid.index(tuple(v)) for v in design["initial"]]
    require(len(set(initial)) == len(initial) and 1 <= len(initial) <= design["budget"] <= len(grid),
            "invalid initial points or search budget")
    require(0 <= design["max_loss"] <= 1 and design["steps"] > 0, "invalid loss limit or step budget")
    coordinates = np.array([[np.log(lr), n, np.log1p(weight)] for lr, n, weight in grid])
    width = np.ptp(coordinates, axis=0)
    return grid, initial, (coordinates - coordinates.min(axis=0)) / np.where(width > 0, width, 1)


def next_index(coordinates, initial, history, design, method):
    """All policies see only their own earlier trials. Indices break ties in grid order."""
    seen = [r["index"] for r in history]
    if len(seen) < len(initial):
        return initial[len(seen)]
    remaining = np.array([i for i in range(len(coordinates)) if i not in seen], dtype=int)
    if method == "random":
        order = np.random.default_rng(design["random_search_seed"]).permutation(len(coordinates))
        return next(int(i) for i in order if i not in seen)
    require(method == "bo", "unknown search method")

    def predict(key):
        kernel = ConstantKernel(1., "fixed") * Matern(length_scale=[.35, .6, .5],
                                                      length_scale_bounds="fixed", nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=.005, normalize_y=True, optimizer=None)
        return gp.fit(coordinates[seen], [r[key] for r in history]).predict(coordinates[remaining], return_std=True)

    loss, loss_sd = predict("retention_loss")
    utility = norm.cdf((design["max_loss"] - loss) / np.maximum(loss_sd, 1e-12))
    feasible = [r["new_gain"] for r in history if r["retention_loss"] <= design["max_loss"]]
    if feasible:
        gain, gain_sd = predict("new_gain")
        delta = gain - max(feasible)
        z = delta / np.maximum(gain_sd, 1e-12)
        utility *= delta * norm.cdf(z) + gain_sd * norm.pdf(z)
    return int(remaining[np.argmax(utility)])


def development_score(directory, *, checkpoint, adaptation, role):
    run = completed_run(directory)
    require({"evaluation.json", "per_frame.jsonl"} <= run["artifact_sha256"].keys(), "evaluation missing from run inventory")
    require(run["role"] == role and not run.get("freeze_sha256"), "search requires development evaluation")
    require(run["checkpoint_sha256"] == checkpoint and run["adaptation_run_sha256"] == adaptation,
            "evaluation checkpoint or adaptation differs")
    summary = json.loads((directory / "evaluation.json").read_text())
    require(all(run[k] == summary[k] for k in ("role", "checkpoint_sha256", "manifest_sha256", "target_policy_sha256")),
            "evaluation summary differs from run")
    value = summary["metrics"]["lane"]["mean_frame_iou"]
    require(value is not None and 0 <= value <= 1, "undefined development IoU")
    return value, (run["manifest_sha256"], run["target_policy_sha256"])


def identities(rows):
    ids, pixels = set(), set()
    for row in rows:
        ids.add(row["sample_id"])
        pixels.add(row["pixel_sha256"] if row.get("pixel_sha256") else row["input_sha256"]["rgb"])
    require(bool(ids) and bool(pixels), "missing run membership")
    return ids, pixels


def search(runs, design, method, seed):
    grid, initial, scaled = search_grid(design)
    scopes = (("development", "meta_dev"), ("retention", "diagnostic_only"))
    baseline = [development_score(runs / ("f0_" + key), checkpoint=DEFAULT_CHECKPOINT_SHA256,
                                  adaptation=None, role=role) for key, role in scopes]
    require(baseline[0][1][0] != baseline[1][1][0], "development and retention manifests must differ")
    evaluation_sets = []
    for key, _ in scopes:
        with (runs / ("f0_" + key) / "per_frame.jsonl").open() as stream:
            evaluation_sets.append(identities(json.loads(line) for line in stream))
    require(all(left.isdisjoint(right) for left, right in zip(*evaluation_sets)), "evaluation IDs or pixels overlap")
    held = [left | right for left, right in zip(*evaluation_sets)]
    history, fixed_inputs = [], None
    while len(history) < design["budget"]:
        index = next_index(scaled, initial, history, design, method)
        name = f"t{index:03d}_s{seed}"
        paths = [runs / name, *(runs / (name + "_" + key) for key, _ in scopes)]
        if not all((path / "_SUCCESS").is_file() for path in paths):
            return dict(status="next_trial", name=name, index=index, settings=grid[index],
                        required_runs=[path.name for path in paths if not (path / "_SUCCESS").is_file()], history=history)
        trial = completed_run(paths[0])
        hashes = trial["artifact_sha256"]
        for artifact in ("protocol.json", "checkpoint.pt", "new_manifest.jsonl", "replay_manifest.jsonl", "training_membership.json"):
            require(artifact in hashes, "missing search artifact: " + artifact)
        require(trial["seed"] == seed and trial["steps"] == design["steps"] and not trial["final_test_accessed"]
                and trial["initial_checkpoint_sha256"] == DEFAULT_CHECKPOINT_SHA256, "trial seed, exposure or budget differs")
        config = json.loads((paths[0] / "protocol.json").read_text())
        weight = config.get("replay_distillation", {}).get("weight", 0.)
        require((config["learning_rate"], config["replay_per_batch"], weight) == grid[index] and
                config["new_per_batch"] + config["replay_per_batch"] == config["batch_size"] == design["batch"] and
                config["steps"] == design["steps"] and trial["protocol_sha256"] == hashes["protocol.json"],
                "trial settings differ from grid")
        fixed = (hashes["new_manifest.jsonl"], hashes["replay_manifest.jsonl"], trial["arm"], trial["target_policy_sha256"],
                 {k: config[k] for k in ("optimizer", "augmentation", "gradient_clip_norm", "weight_decay")})
        if fixed_inputs is None:
            fixed_inputs = fixed
            training = identities(json.loads((paths[0] / "training_membership.json").read_text()))
            require(all(a.isdisjoint(b) for a, b in zip(training, held)), "training and evaluation IDs or pixels overlap")
            require(all(trial["target_policy_sha256"] == src[1][1] for src in baseline), "training and evaluation targets differ")
        require(fixed == fixed_inputs, "training inputs, arm, optimizer or target changed across trials")
        identity = sha256_file(paths[0] / "run.json")
        observations = [development_score(path, checkpoint=hashes["checkpoint.pt"], adaptation=identity, role=role)
                        for path, (_, role) in zip(paths[1:], scopes)]
        require([v[1] for v in observations] == [v[1] for v in baseline], "evaluation manifest or target differs from f0")
        history.append(dict(index=index, settings=grid[index], new_gain=observations[0][0] - baseline[0][0],
                            retention_loss=baseline[1][0] - observations[1][0], run_sha256=identity))
    feasible = [r for r in history if r["retention_loss"] <= design["max_loss"]]
    best = max(feasible, key=lambda r: r["new_gain"]) if feasible else None
    return dict(status="complete", feasible_count=len(feasible), selected=best if best and best["new_gain"] > 0 else None,
                history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Completed baseline and trial evaluation directories")
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--method", choices=("bo", "random"), required=True)
    parser.add_argument("--seed", type=int, required=True, help="Training seed for this search trace")
    args = parser.parse_args()
    result = search(args.runs, json.loads(args.design.read_text()), args.method, args.seed)
    print(json.dumps(dict(result, method=args.method, seed=args.seed, design_sha256=sha256_file(args.design)),
                     allow_nan=False))


if __name__ == "__main__":
    main()
