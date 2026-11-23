"""Train environment centroids offline and assign RGB descriptors without Faiss at inference."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .environment_features import NAMES, PROFILE, extract_environment
from .roi10 import FrontProfile, PROFILE_DIR, REGIONS, extract_region_nodes
from common.bridge import sha256_file


ROI10_HAZE = "front-1280x720-roi10-haze-v1"
ROI10_NAMES = tuple(f"{region}.{feature}" for region in REGIONS
                    for feature in ("chroma", "dark_channel", "local_contrast"))


def column_names(profile):
    if profile == PROFILE:
        return NAMES
    if profile == ROI10_HAZE:
        return ROI10_NAMES
    raise ValueError("unsupported environment feature profile")


def rgb_extractor(profile):
    """Load the calibrated profile once when starting an environment-aware run."""
    if profile == PROFILE:
        return extract_environment
    if profile == ROI10_HAZE:
        camera = FrontProfile.load()
        return lambda rgb: extract_region_nodes(rgb, camera)[:, [1, 3, 4]].ravel()
    raise ValueError("unsupported environment feature profile")


def _descriptors(values, names):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all():
        raise ValueError("expected finite rows in environment descriptor schema order")
    return x


def fit_environment_clusters(values, k, *, method="kmeans", seed=0, profile=PROFILE):
    names = column_names(profile)
    x = _descriptors(values, names)
    if not isinstance(k, int) or not 2 <= k <= len(x):
        raise ValueError("cluster count must be between two and training sample count")
    scaler = StandardScaler().fit(x)
    # The original 7-patch artifacts use float64 scaling. The ten-region study used float32 rows.
    scaled = scaler.transform(x if profile == PROFILE else np.asarray(values, dtype=np.float32))
    if method == "kmeans":
        centers = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled).cluster_centers_
    elif method == "emeans":
        from faiss.contrib.e_means import NumpyEmeans

        centers, _ = NumpyEmeans(scaled.shape[1], k, seed=seed).train(np.ascontiguousarray(scaled, dtype=np.float32))
    else:
        raise ValueError("unsupported environment clustering method")
    spec = dict(profile=profile, features=list(names), method=method, k=k, seed=seed,
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), centers=centers.tolist())
    if profile == ROI10_HAZE:
        spec["camera_profile_sha256"] = sha256_file(PROFILE_DIR / "profile.json")
    return spec


def _squared_distances(spec, values):
    names = column_names(spec["profile"])
    x = _descriptors(values, names)
    if spec["profile"] == ROI10_HAZE and spec.get("camera_profile_sha256") != sha256_file(PROFILE_DIR / "profile.json"):
        raise ValueError("environment camera profile differs")
    if spec["features"] != list(names):
        raise ValueError("environment feature profile or column order differs")
    centers = np.asarray(spec["centers"], dtype=np.float64)
    mean = np.asarray(spec["mean"], dtype=np.float64)
    scale = np.asarray(spec["scale"], dtype=np.float64)
    if (centers.shape != (spec["k"], len(names)) or centers.shape[0] < 2
            or mean.shape != (len(names),) or scale.shape != mean.shape or np.any(scale <= 0)
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
