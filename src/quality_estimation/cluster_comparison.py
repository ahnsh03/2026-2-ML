"""Matched-start e-means/Lloyd training and pair-count environment diagnostics."""
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def fit_centers(method, x, initial, seed, passes):
    x, initial = np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(initial, dtype=np.float32)
    if method == "emeans":
        from faiss.contrib.e_means import NumpyEmeans
        model = NumpyEmeans(x.shape[1], len(initial), nepochs=passes, seed=seed)
        centers, _ = model.train(x, init_centroids=initial.copy())
        return centers, dict(recipe=model.recipe, lloyd_iterations=None)
    if method == "kmeans":
        model = KMeans(n_clusters=len(initial), init=initial.copy(), n_init=1,
                       max_iter=passes, tol=0, random_state=seed, algorithm="lloyd").fit(x)
        return model.cluster_centers_, dict(lloyd_iterations=int(model.n_iter_))
    raise ValueError("unknown clustering method")


def assign_centers(x, centers):
    squared = cdist(x, centers, metric="sqeuclidean")
    return squared.argmin(axis=1), float(squared.min(axis=1).mean())


def environment_pairs(labels, environment, sites):
    """Exact image-pair proportions via contingency counts; no quadratic pair arrays."""
    if not (len(labels) == len(environment) == len(sites)):
        raise ValueError("unaligned environment diagnostic rows")

    def pairs(*columns):
        counts = Counter(zip(*columns))
        return sum(n*(n-1)//2 for n in counts.values())

    same_env_total = pairs(environment)-pairs(environment, sites)
    same_env_match = pairs(environment, labels)-pairs(environment, sites, labels)
    changed_env_total = pairs(sites)-pairs(sites, environment)
    changed_env_match = pairs(sites, labels)-pairs(sites, environment, labels)
    return dict(same_environment_different_site_pairs=same_env_total,
                same_environment_cluster_agreement=(same_env_match/same_env_total if same_env_total else None),
                same_site_changed_environment_pairs=changed_env_total,
                changed_environment_cluster_separation=(1-changed_env_match/changed_env_total if changed_env_total else None))
