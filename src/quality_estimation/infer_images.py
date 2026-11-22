"""Estimate perception quality from RGB only, without opening any GT files."""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from camera_semantic_perception.geometry import letterbox_bgr
from common.bridge import load_target_model, sha256_file
from .artifacts import write_json
from .build_dataset import collate, worker_init
from .image_io import color_channels, decoded_pixel_sha256, read_image
from .dataset_io import require
from .environment_clusters import assign_environment_clusters
from .environment_features import NAMES as ENVIRONMENT_NAMES, extract_environment
from .features import FEATURE_NAMES, extract_features, feature_schema
from .model_io import load_model_run
from .models import QUALITY_MODELS, predict_array
from .perception_identity import resolve_quality_perception
from .alarm import load_alarm_calibration
from .calibration import group_alarm_thresholds


def load_rgb_row(root, row, index):
    require(row.get("view") == "front", "explicit front view is required")
    image, digest = read_image(root, row["image"], cv2.IMREAD_UNCHANGED)
    image = color_channels(image, "RGB")
    expected = row.get("expected_sha256", {}).get("rgb")
    require(expected is None or digest == expected, "RGB hash mismatch")
    tensor, geometry = letterbox_bgr(image, (384, 640))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return dict(index=index, sample_id=row["sample_id"], image=tensor, rgb=rgb,
                pixel_sha256=decoded_pixel_sha256(rgb), geometry=geometry, sha256=digest)


class RGBSamples(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return load_rgb_row(self.root, self.rows[index], index)


class RGBQualityScorer:
    """A fixed perception model, quality model and optional alarm shared by batch and live input."""

    def __init__(self, model_run, model, device, *, checkpoint=None, adaptation_run=None,
                 alarm_calibration=None, environment_centers=None):
        checkpoint, identity, _ = resolve_quality_perception(adaptation_run, checkpoint)
        self.bundle, _ = load_model_run(model_run, model, perception_identity=identity)
        require(sha256_file(checkpoint) == self.bundle["checkpoint_sha256"],
                "inference checkpoint differs from model training")
        self.alarm = load_alarm_calibration(alarm_calibration, model_run, self.bundle) if alarm_calibration else None
        self.grouped_alarm = self.alarm.get("group_policy") if self.alarm else None
        if self.grouped_alarm and environment_centers is None:
            environment_centers = alarm_calibration / "environment_centers.json"
        self.centers = None
        self.centers_bytes = None
        self.metadata = dict(model_run_sha256=sha256_file(model_run / "run.json"), model=self.bundle["model"],
                             inference_n_jobs=self.bundle["runtime_inference_jobs"], uses_ground_truth=False,
                             checkpoint_sha256=self.bundle["checkpoint_sha256"], perception_identity=identity,
                             alarm_calibration_run_sha256=sha256_file(alarm_calibration / "run.json") if self.alarm else None,
                             precision="fp32; tf32 off; deterministic cudnn")
        if environment_centers is not None:
            self.centers_bytes = environment_centers.read_bytes()
            digest = hashlib.sha256(self.centers_bytes).hexdigest()
            if self.grouped_alarm:
                require(digest == self.alarm["centers_sha256"], "environment centers differ from alarm calibration")
            self.centers = json.loads(self.centers_bytes)
            assign_environment_clusters(self.centers, np.empty((0, len(ENVIRONMENT_NAMES))))
            self.metadata.update(environment_centers_sha256=digest, environment_profile=self.centers["profile"])
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        torch.manual_seed(20261116)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.perception = load_target_model(checkpoint, self.device)

    @torch.inference_mode()
    def score(self, images, items):
        probabilities = self.perception.model(images.to(self.device))[1].float().softmax(1).cpu().numpy()
        features = np.stack([extract_features(item["rgb"], prob, item["geometry"])
                             for item, prob in zip(items, probabilities)])
        predictions = predict_array(self.bundle, features)
        descriptors = None
        if self.centers is not None:
            descriptors = np.stack([extract_environment(item["rgb"]) for item in items])
            assignments = assign_environment_clusters(self.centers, descriptors)
            if self.grouped_alarm:
                cutoffs = group_alarm_thresholds(self.grouped_alarm, assignments)
        results = []
        for index, item in enumerate(items):
            result = dict(sample_id=item["sample_id"], rgb_sha256=item["sha256"], pixel_sha256=item["pixel_sha256"],
                          geometry=asdict(item["geometry"]),
                          **{name: float(values[index]) for name, values in predictions.items()})
            if self.centers is not None:
                result["environment_cluster"] = int(assignments[index])
            if self.alarm:
                cutoff = float(cutoffs[index]) if self.grouped_alarm else self.alarm["policy"]["alarm_threshold"]
                if self.grouped_alarm:
                    group = str(assignments[index])
                    result["alarm_cutoff_source"] = ("environment_group" if group in self.grouped_alarm["group_thresholds"]
                                                     else "global_fallback")
                result.update(low_quality_alarm=result["estimated_iou"] < cutoff, alarm_threshold=cutoff,
                              quality_target_iou=self.alarm["policy"]["target_iou"])
            results.append(result)
        return results, features, descriptors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="JSONL: sample_id, view=front, image relative to data root")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--model", default="selected", choices=("selected", *QUALITY_MODELS))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--adaptation-run", type=Path)
    p.add_argument("--alarm-calibration", type=Path, help="Completed model-bound development calibration directory")
    p.add_argument("--environment-centers", type=Path, help="Optional frozen front-camera environment centroid JSON")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    require(a.batch_size > 0 and a.num_workers >= 0, "invalid batch/worker settings")
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    ids = [r["sample_id"] for r in rows]
    require(bool(ids) and len(ids) == len(set(ids)), "empty or duplicate sample IDs")
    scorer = RGBQualityScorer(a.model_run, a.model, a.device, checkpoint=a.checkpoint, adaptation_run=a.adaptation_run,
                              alarm_calibration=a.alarm_calibration, environment_centers=a.environment_centers)
    loader = DataLoader(RGBSamples(a.data_root, rows), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, collate_fn=collate, worker_init_fn=worker_init)
    out = a.output_dir; out.mkdir(parents=True, exist_ok=False)
    metadata = dict(scorer.metadata, status="running", input_manifest_sha256=sha256_file(a.manifest), sample_count=len(ids))
    if scorer.centers_bytes is not None:
        (out / "environment_centers.json").write_bytes(scorer.centers_bytes)
    write_json(out / "run.json", metadata)
    environment_rows = []
    started = time.perf_counter(); features = []
    try:
        with (out / "predictions.jsonl").open("w") as stream, torch.inference_mode():
            for images, items in loader:
                results, batch_features, descriptors = scorer.score(images, items)
                features.append(batch_features)
                if descriptors is not None:
                    environment_rows.append(descriptors)
                for result in results:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
        if scorer.centers is not None:
            np.save(out / "environment_features.npy", np.concatenate(environment_rows))
        np.save(out / "X.npy", np.concatenate(features))
        write_json(out / "sample_ids.json", ids)
        write_json(out / "schema.json", feature_schema())
        write_json(out / "feature_names.json", list(FEATURE_NAMES))
        hashes = {str(path.relative_to(out)): sha256_file(path) for path in out.iterdir()
                  if path.is_file() and path.name != "run.json"}
        metadata.update(status="complete", elapsed_seconds=time.perf_counter()-started, artifact_sha256=hashes)
        write_json(out / "run.json", metadata)
        (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")
        print(json.dumps({k:v for k,v in metadata.items() if k != "artifact_sha256"}), flush=True)
    except BaseException as exc:
        write_json(out / "run.json", dict(metadata, status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
