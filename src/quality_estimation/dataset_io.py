"""Read completed, hash-checked X/y artifacts before training or evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from .features import FEATURE_NAMES, feature_schema
from .perception_identity import require_matching_identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class QualityDataset:
    path: Path
    X: np.ndarray
    y: np.ndarray
    defined: np.ndarray
    ids: list
    samples: list
    metadata: dict
    names: list


def load_dataset(path, verify_hashes=True, perception_identity=None):
    path = Path(path).resolve()
    metadata = json.loads((path / "dataset.json").read_text())
    require(metadata.get("status") == "complete", "dataset is not complete")
    require((path / "_SUCCESS").read_text().strip() == sha256_file(path / "dataset.json"), "completion hash mismatch")
    if verify_hashes:
        for name, digest in metadata["artifact_sha256"].items():
            artifact = (path / name).resolve()
            try:
                artifact.relative_to(path)
            except ValueError as exc:
                raise ValueError("artifact path escapes dataset") from exc
            require(sha256_file(artifact) == digest, "artifact hash mismatch: " + name)
    schema = json.loads((path / "schema.json").read_text())
    require(schema == feature_schema(), "feature/target schema differs from runtime")
    require_matching_identity(metadata["checkpoint_sha256"], metadata.get("perception_identity"), perception_identity)
    names = json.loads((path / "feature_names.json").read_text())
    require(names == list(FEATURE_NAMES), "feature column order mismatch")
    ids = json.loads((path / "sample_ids.json").read_text())
    require(bool(ids) and len(ids) == len(set(ids)), "empty or duplicate sample IDs")
    X = np.load(path / "X.npy", allow_pickle=False)
    y = np.load(path / "y.npy", allow_pickle=False)
    defined = np.load(path / "label_defined.npy", allow_pickle=False)
    require(X.shape == (len(ids), len(names)) and X.dtype == np.float32 and np.isfinite(X).all(), "invalid X")
    require(y.shape == defined.shape == (len(ids),) and y.dtype == np.float32 and defined.dtype == bool, "invalid y/mask")
    require(np.isfinite(y[defined]).all() and np.isnan(y[~defined]).all(), "invalid defined/undefined targets")
    require(((y[defined] >= 0) & (y[defined] <= 1)).all(), "IoU outside [0,1]")
    samples = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
    require([r["sample_id"] for r in samples] == ids, "metadata/array row mismatch")
    require([r["row_index"] for r in samples] == list(range(len(ids))), "metadata row indexes differ")
    require(metadata["sample_count"] == len(ids), "sample count mismatch")
    return QualityDataset(path, X, y, defined, ids, samples, metadata, names)
