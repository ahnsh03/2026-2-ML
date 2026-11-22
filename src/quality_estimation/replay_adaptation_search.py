"""Compare search policies on a complete, previously measured development response surface."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .dataset_io import require
from .search_adaptation import next_index, search_grid


def replay(surface, grid_name):
    config = surface["search"]
    domain = surface["grids"][grid_name]
    points = list(itertools.product(*(domain[k] for k in ("lr", "replay", "lam"))))
    observed = {}
    for row in surface["observations"]:
        key = tuple(row["settings"])
        require(key not in observed, "duplicate trial settings")
        require(all(np.isfinite(row[k]) and -1 <= row[k] <= 1 for k in ("new_gain", "retention_loss")),
                "invalid measured IoU difference")
        observed[key] = row
    require(set(points) <= observed.keys(), "response surface is incomplete")
    observations = [observed[key] for key in points]
    budgets = sorted(config["budgets"])
    require(budgets and len(set(budgets)) == len(budgets) and
            1 <= config["shared_initial"] <= budgets[0] <= budgets[-1] <= len(points), "invalid replay budgets")
    require(config["initializations"] > 0, "at least one paired initialization required")
    result = {(method, b): [] for method in ("bo", "random") for b in budgets}
    feasible = [r["new_gain"] for r in observations if r["retention_loss"] <= config["max_loss"]]
    optimum = max([0.] + feasible)
    for seed in range(config["first_seed"], config["first_seed"] + config["initializations"]):
        starts = np.random.default_rng(seed).choice(len(points), config["shared_initial"], replace=False)
        design = dict(domain, batch=config["batch"], steps=config["steps"], budget=budgets[-1],
                      max_loss=config["max_loss"], random_search_seed=seed + config["random_seed_offset"],
                      initial=[points[i] for i in starts])
        _, initial, coordinates = search_grid(design)
        for method in ("bo", "random"):
            history, gains = [], []
            for b in range(1, budgets[-1]+1):
                # The policy sees only its own earlier observations.
                i = next_index(coordinates, initial, history, design, method)
                row = observations[i]
                history.append(dict(index=i, new_gain=row["new_gain"], retention_loss=row["retention_loss"]))
                if row["retention_loss"] <= config["max_loss"]:
                    gains.append(row["new_gain"])
                if b in budgets:
                    result[method, b].append((len(gains), max([0.] + gains)))
    summaries = []
    for (method, b), pairs in result.items():
        count, best = np.asarray(pairs).T
        summaries.append(dict(method=method, budget=b, initializations=len(pairs),
                              starts_with_feasible=int(np.sum(count > 0)), median_feasible=float(np.median(count)),
                              median_best_gain=float(np.median(best)), mean_best_gain=float(np.mean(best)),
                              median_regret=float(np.median(optimum - best)), exact_best_gain=optimum,
                              starts_reaching_exact_best=int(np.sum(optimum-best < 1e-12))))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--grid", required=True, help="Named complete grid in the input surface")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(replay(json.loads(args.surface.read_text()), args.grid), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
