"""Immutable, aligned feature/label artifacts shared by input adapters."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common.bridge import sha256_file
from .features import FEATURE_NAMES, feature_schema


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class FeatureDatasetWriter:
    """A failed run is retained; only a fully finalized run gets _SUCCESS."""

    def __init__(self, out, metadata, audit_per_run=1):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.metadata = dict(metadata, schema_version=feature_schema()["schema_version"],
                             dataset_format_version="ml-quality-dataset-1.1.0", status="running",
                             started_utc=datetime.now(timezone.utc).isoformat(),
                             feature_count=len(FEATURE_NAMES), regression_training_performed=False)
        self.audit_per_run = audit_per_run
        self.audited = Counter()
        self.ids, self.X, self.y, self.defined = [], [], [], []
        self.seen = set()
        self.runs = {}
        self.streams = []
        self.finished = False
        (self.out / "audit").mkdir()
        write_json(self.out / "schema.json", feature_schema())
        write_json(self.out / "feature_names.json", list(FEATURE_NAMES))
        self.metadata["feature_schema_sha256"] = sha256_file(self.out / "schema.json")
        write_json(self.out / "dataset.json", self.metadata)
        write_json(self.out / "run.json", {"status": "running"})

    def __enter__(self):
        for name in ("features.csv", "labels.csv", "samples.jsonl"):
            self.streams.append((self.out / name).open("w", newline="", encoding="utf-8"))
        self.wx = csv.writer(self.streams[0])
        self.wx.writerow(["sample_id", *FEATURE_NAMES])
        fields = ["sample_id", "lane_iou", "label_defined", "undefined_reason", "tp", "fp", "fn",
                  "union", "gt_pixels", "pred_pixels", "valid_pixels"]
        self.wy = csv.DictWriter(self.streams[1], fieldnames=fields)
        self.wy.writeheader()
        return self

    def add(self, x, label, sample, fixture=None):
        sid = sample["sample_id"]
        if sid in self.seen:
            raise ValueError("duplicate sample_id: " + sid)
        if x.shape != (len(FEATURE_NAMES),) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("invalid feature row")
        defined = label["label_defined"]
        value = label["lane_iou"]
        if (defined and (value is None or not np.isfinite(value) or not 0 <= value <= 1)) or (not defined and value is not None):
            raise ValueError("inconsistent label validity")
        index = len(self.ids)
        self.seen.add(sid)
        self.ids.append(sid)
        self.X.append(x.copy())
        self.y.append(value if defined else np.nan)
        self.defined.append(defined)
        self.wx.writerow([sid, *x.tolist()])
        self.wy.writerow({"sample_id": sid, **label})
        self.streams[2].write(json.dumps(dict(sample, row_index=index), ensure_ascii=False, allow_nan=False) + "\n")
        stat = self.runs.setdefault(sample["run_id"], {"n": 0, "defined": 0, "iou_sum": 0.0})
        stat["n"] += 1
        stat["defined"] += int(defined)
        stat["iou_sum"] += value if defined else 0.0
        if fixture is not None and self.audited[sample["run_id"]] < self.audit_per_run:
            np.savez_compressed(self.out / "audit" / f"{index:06d}.npz", **fixture)
            self.audited[sample["run_id"]] += 1

    def finalize(self, extra=None):
        if self.finished or not self.ids:
            raise ValueError("empty or already finalized dataset")
        for stream in self.streams:
            stream.flush()
            stream.close()
        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y, dtype=np.float32)
        defined = np.asarray(self.defined, dtype=bool)
        for name, array in (("X", X), ("y", y), ("label_defined", defined)):
            np.save(self.out / (name + ".npy"), array, allow_pickle=False)
        write_json(self.out / "sample_ids.json", self.ids)
        for stat in self.runs.values():
            total = stat.pop("iou_sum")
            stat["mean_iou"] = total / stat["defined"] if stat["defined"] else None
        write_json(self.out / "summary.json", dict(X_shape=list(X.shape), y_shape=list(y.shape),
                   defined_labels=int(defined.sum()), undefined_labels=int((~defined).sum()),
                   finite_features=True, runs=self.runs, elapsed_seconds=time.perf_counter()-self.started))
        self.metadata.update(status="complete", sample_count=len(self.ids),
                             finished_utc=datetime.now(timezone.utc).isoformat(), **(extra or {}))
        # Include probability fixtures as well as all top-level artifacts.
        self.metadata["artifact_sha256"] = {str(p.relative_to(self.out)): sha256_file(p)
            for p in sorted(self.out.rglob("*")) if p.is_file() and p.name not in ("dataset.json", "run.json", "_SUCCESS")}
        write_json(self.out / "dataset.json", self.metadata)
        digest = sha256_file(self.out / "dataset.json")
        write_json(self.out / "run.json", {"status": "complete", "dataset_metadata_sha256": digest})
        (self.out / "_SUCCESS").write_text(digest + "\n")
        self.finished = True
        return {"X_shape": list(X.shape), "y_shape": list(y.shape), "defined": int(defined.sum())}

    def __exit__(self, exc_type, exc, tb):
        for stream in self.streams:
            if not stream.closed:
                stream.close()
        if not self.finished:
            self.metadata.update(status="failed", error=str(exc) if exc else "finalize was not called")
            write_json(self.out / "dataset.json", self.metadata)
            write_json(self.out / "run.json", {"status": "failed", "error": self.metadata["error"]})
        return False
