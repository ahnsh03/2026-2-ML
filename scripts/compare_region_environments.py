#!/usr/bin/env python3
"""Compare RGB descriptor clustering on held-out locations without using IoU labels."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

from faiss.contrib.e_means import NumpyEmeans
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from compare_region_candidates import load, save
from quality_estimation.region_feature_views import environment_views, feature_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    schema = json.loads((args.packs/"schema.json").read_text())
    train = load(args.packs, protocol["training_keys"])
    dev = load(args.packs, protocol["development_keys"])
    if set(train["pixel_sha256"]) & set(dev["pixel_sha256"]):
        raise ValueError("training and development overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = environment_views(schema)
    results, stability = [], []
    for view in protocol["cluster_views"]:
        columns = views[view]
        x = feature_matrix(train, schema, columns)
        dx = feature_matrix(dev, schema, columns)
        selector = VarianceThreshold().fit(x)
        scaler = StandardScaler().fit(selector.transform(x))
        scaled = scaler.transform(selector.transform(x)).astype(np.float32)
        dscaled = scaler.transform(selector.transform(dx)).astype(np.float32)
        for method, k in itertools.product(("kmeans", "emeans"), protocol["cluster_counts"]):
            assignments = []
            for seed in protocol["cluster_seeds"]:
                start = time.perf_counter()
                if method == "kmeans":
                    centers = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled).cluster_centers_
                else:
                    centers, _ = NumpyEmeans(scaled.shape[1], k, seed=seed).train(scaled)
                elapsed = time.perf_counter()-start
                train_labels = cdist(scaled, centers, "sqeuclidean").argmin(1)
                dev_labels = cdist(dscaled, centers, "sqeuclidean").argmin(1)
                assignments.append(dev_labels)
                row = dict(view=view, method=method, k=k, seed=seed, dimensions=len(columns),
                           active_dimensions=scaled.shape[1], seconds=elapsed,
                           training_inertia=float(cdist(scaled, centers, "sqeuclidean").min(1).sum()),
                           dev_weather_ami=float(adjusted_mutual_info_score(dev["weather"], dev_labels)),
                           dev_hour_ami=float(adjusted_mutual_info_score(dev["hour"], dev_labels)),
                           dev_site_ami=float(adjusted_mutual_info_score(dev["site"], dev_labels)),
                           occupied_train_clusters=int(np.unique(train_labels).size),
                           occupied_dev_clusters=int(np.unique(dev_labels).size))
                # Majority tag names are learned from training settings, then evaluated elsewhere.
                for target in ("weather", "hour"):
                    labels = np.unique(train[target])
                    default = labels[np.argmax([(train[target] == v).sum() for v in labels])]
                    tags = []
                    for cluster in range(k):
                        mask = train_labels == cluster
                        values, counts = np.unique(train[target][mask], return_counts=True)
                        tags.append(values[counts.argmax()] if len(values) else default)
                    prediction = np.asarray(tags)[dev_labels]
                    row["dev_"+target+"_balanced_accuracy"] = float(balanced_accuracy_score(dev[target], prediction))
                name = view+"_"+method+"_k"+str(k)+"_s"+str(seed)
                path = args.output_dir/(name+".npz")
                np.savez_compressed(path, centers=centers, selected=selector.get_support(indices=True),
                                    mean=scaler.mean_, scale=scaler.scale_, ordered_features=columns,
                                    train_labels=train_labels, dev_labels=dev_labels)
                row["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                results.append(row)
                save(args.output_dir/"summary.json", results)
                print(name, round(row["dev_weather_ami"], 4), round(row["dev_hour_ami"], 4), flush=True)
            pairs = [adjusted_rand_score(assignments[i], assignments[j])
                     for i, j in itertools.combinations(range(len(assignments)), 2)]
            stability.append(dict(view=view, method=method, k=k,
                                  mean_pairwise_dev_ari=float(np.mean(pairs))))
            save(args.output_dir/"stability.json", stability)
    save(args.output_dir/"receipt.json", dict(
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        use="descriptor validation for environment context; cluster IDs do not define acceptable IoU",
    ))


if __name__ == "__main__":
    main()
