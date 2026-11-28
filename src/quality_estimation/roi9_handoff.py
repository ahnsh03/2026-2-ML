"""Validate the exported ROI9 v2 table without relabeling it as historical X56."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root):
    root = Path(root).resolve()
    files = {}
    for line in (root/"SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = (root/relative).resolve()
        if root not in path.parents or relative in files:
            raise ValueError("unsafe or duplicate checksum path")
        if sha256(path) != digest:
            raise ValueError("handoff checksum mismatch: "+relative)
        files[relative] = digest
    required = {"v2_frames.npz", "frames.csv", "code/config/region_feature_recipe_v2.json"}
    if not required <= files.keys():
        raise ValueError("handoff manifest omits required inputs")
    return files


@dataclass
class ROI9Handoff:
    root: Path
    arrays: dict
    rows: list
    recipe: dict
    hashes: dict

    def training_folds(self):
        rows = [r for r in self.rows if r["split"] == "train"]
        try:
            labels = np.array([int(r["train_cv_fold"]) for r in rows])
        except (KeyError, ValueError) as exc:
            raise ValueError("missing training CV membership") from exc
        if set(labels) != {0, 1, 2}:
            raise ValueError("expected handoff folds 0, 1 and 2")
        folds, audit = [], []
        xy = np.array([[float(r[k]) for k in ("x", "y")] for r in rows])
        if not np.isfinite(xy).all():
            raise ValueError("invalid training coordinates")
        for k in range(3):
            train, valid = np.flatnonzero(labels != k), np.flatnonzero(labels == k)
            for field in ("group", "site", "pixel_sha256"):
                left, right = ({rows[i][field] for i in index} for index in (train, valid))
                if "" in left or "" in right or left & right:
                    raise ValueError(field+" missing or crossing CV folds")
            minimum = float(cdist(xy[train], xy[valid]).min())
            if minimum <= 0:
                raise ValueError("same coordinates cross CV folds")
            folds.append(("handoff_fold_"+str(k), train, valid))
            audit.append(dict(name=folds[-1][0], train=train.tolist(), valid=valid.tolist(),
                              minimum_distance_m=minimum))
        return folds, audit


def load_roi9_handoff(root, expected_recipe_sha256=None):
    root = Path(root).resolve()
    hashes = verify_manifest(root)
    recipe_relative = "code/config/region_feature_recipe_v2.json"
    if expected_recipe_sha256 and hashes[recipe_relative] != expected_recipe_sha256:
        raise ValueError("recipe differs from frozen search protocol")
    recipe = json.loads((root/recipe_relative).read_text())
    if recipe["profile"] != "roi9-front-v1":
        raise ValueError("not the roi9 profile")
    with np.load(root/"v2_frames.npz", allow_pickle=False) as pack:
        data = {key: pack[key] for key in pack.files}
    with (root/"frames.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    n = len(rows)
    if not n or len(np.unique(data["ids"])) != n:
        raise ValueError("empty or duplicate frame IDs")
    for key, field in (("ids", "sample_id"), ("split", "split"), ("source_cohort", "source_cohort")):
        if data[key].shape != (n,) or not np.array_equal(data[key], [r[field] for r in rows]):
            raise ValueError("CSV/NPZ row alignment differs: "+key)
    if [r["row"] for r in rows] != [str(i) for i in range(n)]:
        raise ValueError("CSV row numbers differ")
    for key, count in (("X", 27), ("RGB54", 54)):
        if data[key].shape != (n, count) or data[key].dtype != np.float32 or not np.isfinite(data[key]).all():
            raise ValueError("invalid "+key)
    if (data["y"].shape != (n,) or data["y"].dtype != np.float32
            or not np.isfinite(data["y"]).all() or np.any((data["y"] < 0) | (data["y"] > 1))):
        raise ValueError("invalid IoU targets")
    for key in ("weather", "hour"):
        if data[key].shape != (n,) or not np.array_equal(data[key], [int(r[key]) for r in rows]):
            raise ValueError("CSV/NPZ condition mismatch")
    tasks = recipe["tasks"]
    for task, key in (("quality_gp", "quality_columns"), ("quality_rf", "quality_columns"), ("environment", "rgb54_columns")):
        if data[key].tolist() != tasks[task]["ordered_features"] or len(set(data[key])) != len(data[key]):
            raise ValueError("ordered feature schema differs: "+task)
        relative = "models/"+tasks[task]["artifact_filename"]
        if hashes.get(relative) != tasks[task]["model_sha256"]:
            raise ValueError("model recipe/manifest hash mismatch: "+task)
    roles = set(data["split"])
    if not {"train", "development"} <= roles or not roles <= {"train", "development", "diagnostic"}:
        raise ValueError("unexpected data roles")
    for field in ("pixel_sha256", "site", "group"):
        sets = {role: {r[field] for r in rows if r["split"] == role} for role in ("train", "development")}
        if "" in sets["train"] or "" in sets["development"] or sets["train"] & sets["development"]:
            raise ValueError(field+" missing or overlapping train/development")
    result = ROI9Handoff(root, data, rows, recipe, hashes)
    result.training_folds()
    return result
