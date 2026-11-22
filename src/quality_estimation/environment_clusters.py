"""Train environment centroids offline and assign RGB descriptors without Faiss at inference."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .environment_features import NAMES, PROFILE


def _descriptors(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError("expected finite rows in environment descriptor schema order")
    return x


def fit_environment_clusters(values, k, *, method="kmeans", seed=0):
    x = _descriptors(values)
    if not isinstance(k, int) or not 2 <= k <= len(x):
        raise ValueError("cluster count must be between two and training sample count")
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    if method == "kmeans":
        centers = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled).cluster_centers_
    elif method == "emeans":
        from faiss.contrib.e_means import NumpyEmeans

        centers, _ = NumpyEmeans(scaled.shape[1], k, seed=seed).train(np.ascontiguousarray(scaled, dtype=np.float32))
    else:
        raise ValueError("unsupported environment clustering method")
    return dict(profile=PROFILE, features=list(NAMES), method=method, k=k, seed=seed,
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), centers=centers.tolist())


def _squared_distances(spec, values):
    x = _descriptors(values)
    if spec["profile"] != PROFILE or spec["features"] != list(NAMES):
        raise ValueError("environment feature profile or column order differs")
    centers = np.asarray(spec["centers"], dtype=np.float64)
    mean = np.asarray(spec["mean"], dtype=np.float64)
    scale = np.asarray(spec["scale"], dtype=np.float64)
    if (centers.shape != (spec["k"], len(NAMES)) or centers.shape[0] < 2
            or mean.shape != (len(NAMES),) or scale.shape != mean.shape or np.any(scale <= 0)
            or not all(np.isfinite(param).all() for param in (mean, scale, centers))):
        raise ValueError("invalid centroid or scaling parameters")
    return cdist((x-mean)/scale, centers, metric="sqeuclidean")


def assign_environment_clusters(spec, values):
    return np.argmin(_squared_distances(spec, values), axis=1)


def environment_assignment_distances(spec, values):
    """Return cluster ID, standardized Euclidean distance, and runner-up minus nearest distance."""
    squared = _squared_distances(spec, values)
    labels = np.argmin(squared, axis=1)
    closest = np.sqrt(np.partition(squared, 1, axis=1)[:, :2])
    return labels, closest[:, 0], closest[:, 1] - closest[:, 0]
