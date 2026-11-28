#!/usr/bin/env python3
"""Compare RGB54 e-means and Lloyd with identical starts and five-pass budgets."""
import argparse
import importlib.metadata
import itertools
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import faiss
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

from quality_estimation.cluster_comparison import fit_centers, assign_centers, environment_pairs
from quality_estimation.roi9_handoff import load_roi9_handoff, sha256
from run_gp_search import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("config/roi9_comparisons_v1.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.protocol.read_text())
    spec = config["clusters"]
    pack = load_roi9_handoff(args.handoff, config["recipe_sha256"])
    d = pack.arrays
    train = d["split"] == "train"
    selector = VarianceThreshold().fit(d["RGB54"][train])
    scaler = StandardScaler().fit(selector.transform(d["RGB54"][train]))
    x = np.ascontiguousarray(scaler.transform(selector.transform(d["RGB54"])), dtype=np.float32)
    sites = np.array([r["site"] for r in pack.rows])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out/"models").mkdir()
    (out/"source").mkdir()
    sources = [Path(__file__), Path("src/quality_estimation/cluster_comparison.py"), Path("src/quality_estimation/roi9_handoff.py")]
    for source in sources:
        shutil.copy2(source, out/"source"/source.name)
    shutil.copy2(args.protocol, out/"protocol.json")
    save(out/"input_receipt.json", dict(hashes=pack.hashes, source_sha256={s.name:sha256(s) for s in sources},
         active_dimensions=x.shape[1], versions={n:importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "faiss-cpu")}))
    (out/"command.txt").write_text("PYTHONPATH=src "+shlex.join([sys.executable]+sys.argv)+"\n")
    masks = {str(c):d["source_cohort"]==c for c in np.unique(d["source_cohort"])}
    masks["fresh_all"] = np.isin(d["source_cohort"], ["fresh", "fresh_clock_supplement"])
    results, assignments = [], {}
    faiss.omp_set_num_threads(config["threads"])
    with threadpool_limits(config["threads"]):
        for k, seed in itertools.product(spec["counts"], spec["seeds"]):
            initial, _ = kmeans_plusplus(x[train], n_clusters=k, random_state=seed)
            initial_digest = __import__("hashlib").sha256(initial.tobytes()).hexdigest()
            for method in ("kmeans", "emeans"):
                start = time.perf_counter()
                centers, details = fit_centers(method, x[train], initial, seed, spec["passes"])
                elapsed = time.perf_counter()-start
                labels, _ = assign_centers(x, centers)
                name = f"{method}_k{k}_s{seed}"
                assignments[method,k,seed] = labels
                file = out/"models"/(name+".npz")
                np.savez_compressed(file, centers=centers, initial_centers=initial, selected=selector.get_support(indices=True),
                    mean=scaler.mean_, scale=scaler.scale_, ordered_features=d["rgb54_columns"], ids=d["ids"], labels=labels)
                row = dict(method=method, k=k, seed=seed, fit_seconds=elapsed,
                           initial_centers_sha256=initial_digest, artifact_sha256=sha256(file), details=details, cohorts={})
                for cohort, mask in masks.items():
                    _, mse = assign_centers(x[mask], centers)
                    condition = d["weather"][mask]*100+d["hour"][mask]
                    row["cohorts"][cohort] = dict(n=int(mask.sum()), mean_squared_distance=mse,
                        weather_ami=float(adjusted_mutual_info_score(d["weather"][mask], labels[mask])),
                        hour_ami=float(adjusted_mutual_info_score(d["hour"][mask], labels[mask])),
                        site_ami=float(adjusted_mutual_info_score(sites[mask], labels[mask])),
                        occupied_clusters=int(np.unique(labels[mask]).size),
                        largest_cluster_fraction=float(np.bincount(labels[mask], minlength=k).max()/mask.sum()),
                        **environment_pairs(labels[mask], condition, sites[mask]))
                results.append(row)
                save(out/"results.json", results)
                print(name, "seconds", round(elapsed, 4), "dev mse", round(row["cohorts"]["development"]["mean_squared_distance"], 4), flush=True)
    stability = []
    dev = d["split"] == "development"
    for method,k in itertools.product(("kmeans", "emeans"), spec["counts"]):
        stability.append(dict(method=method,k=k,development_mean_seed_ari=float(np.mean([
            adjusted_rand_score(assignments[method,k,a][dev], assignments[method,k,b][dev])
            for a,b in itertools.combinations(spec["seeds"], 2)]))))
    save(out/"stability.json", stability)
    save(out/"summary.json", dict(results=results, stability=stability, diagnostic_only=True,
          final_loss_uses_nearest_final_centroids=True, matched_initializations=True, same_pass_count_not_equal_runtime=True))
    save(out/"artifacts.json", {str(f.relative_to(out)):sha256(f) for f in out.rglob("*") if f.is_file()})
    (out/"_SUCCESS").write_text(sha256(out/"artifacts.json")+"\n")
    print("Cluster comparison complete", flush=True)


if __name__ == "__main__":
    main()
