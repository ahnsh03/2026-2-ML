"""RGB-only nine-region GP replay. Does not open Semantic/GT or use weather/time labels."""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
from dataclasses import asdict
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from common.bridge import load_target_model, sha256_file
from .infer_images import RGBSamples
from .build_dataset import collate, worker_init
from .spatial_features import SpatialProfile
from .selected_region_features import extract_selected_features
from .roi10 import reduce_heads
from .spatial_predictor import FrozenSpatialPredictor


class SpatialRGBScorer:
    def __init__(
        self,
        recipe,
        model,
        checkpoint,
        device,
        *,
        calibration=None,
        simulator_release=None,
    ):
        self.profile = SpatialProfile.load()
        self.predictor = FrozenSpatialPredictor(
            recipe,
            model,
            self.profile,
            calibration_path=calibration,
            simulator_release=simulator_release,
        )
        if sha256_file(checkpoint) != self.predictor.specification["checkpoint_sha256"]:
            raise ValueError("perception checkpoint differs from feature/model recipe")
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device(device)
        self.perception = load_target_model(checkpoint, self.device, weights="ema")

    @torch.inference_mode()
    def score(self, images, items):
        start = time.perf_counter()
        drive, mark = self.perception.model(images.to(self.device))
        drives = drive.argmax(1).cpu().numpy()
        probabilities = mark.float().softmax(1).cpu().numpy()
        perception_seconds = time.perf_counter() - start
        vectors = []
        results = []
        maps = []
        feature_seconds = 0.0
        gp_seconds = 0.0
        for item, d, p in zip(items, drives, probabilities):
            started = time.perf_counter()
            head = reduce_heads(d, p, item["geometry"])
            vector = extract_selected_features(
                item["rgb"],
                head,
                self.profile,
                self.predictor.columns,
                marking_probability=p,
            )
            feature_seconds += time.perf_counter() - started
            started = time.perf_counter()
            result = self.predictor.predict_vector(vector)
            gp_seconds += time.perf_counter() - started
            result.update(
                sample_id=item["sample_id"],
                rgb_sha256=item["sha256"],
                pixel_sha256=item["pixel_sha256"],
                geometry=asdict(item["geometry"]),
                uses_ground_truth=False,
                checkpoint_sha256=self.perception.checkpoint_sha256,
                profile=self.profile.version,
                regions_sha256=self.profile.regions_sha256,
            )
            results.append(result)
            vectors.append(vector)
            maps.append(head)
        return (
            results,
            np.stack(vectors),
            np.stack(maps),
            probabilities,
            dict(
                perception_seconds=perception_seconds,
                feature_seconds=feature_seconds,
                gp_seconds=gp_seconds,
            ),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "data-root",
        "recipe",
        "model",
        "checkpoint",
        "output-dir",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--simulator-release",
        help="Declared capture provenance; must match a release-specific calibration",
    )
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--save-heads",
        action="store_true",
        help="Save optional maps/softmax on the experiment output volume",
    )
    a = ap.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if a.batch_size <= 0 or a.num_workers < 0:
        raise ValueError("invalid batch/workers")
    rows = [
        json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()
    ]
    ids = [r["sample_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate RGB manifest")
    scorer = SpatialRGBScorer(
        a.recipe,
        a.model,
        a.checkpoint,
        a.device,
        calibration=a.calibration,
        simulator_release=a.simulator_release,
    )
    loader = DataLoader(
        RGBSamples(a.data_root, rows),
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        collate_fn=collate,
        worker_init_fn=worker_init,
        shuffle=False,
    )
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "input_manifest.jsonl").write_bytes(a.manifest.read_bytes())
    metadata = dict(
        status="running",
        uses_ground_truth=False,
        recipe_sha256=sha256_file(a.recipe),
        model_sha256=sha256_file(a.model),
        checkpoint_sha256=sha256_file(a.checkpoint),
        calibration_sha256=sha256_file(a.calibration) if a.calibration else None,
        input_manifest_sha256=sha256_file(a.manifest),
        rows=len(rows),
        torch=torch.__version__,
        numpy=np.__version__,
        opencv=cv2.__version__,
        device=str(scorer.device),
        batch_size=a.batch_size,
        declared_simulator_release=a.simulator_release,
        profile=scorer.profile.version,
        regions_sha256=scorer.profile.regions_sha256,
        source_sha256=sha256_file(Path(__file__)),
        ordered_features=scorer.predictor.columns,
    )

    def write_run():
        (out / "run.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )

    write_run()
    features = []
    timings = []
    started = time.perf_counter()
    offset = 0
    try:
        with (out / "predictions.jsonl").open("x") as stream:
            for images, items in loader:
                results, x, heads, probability, timing = scorer.score(images, items)
                features.append(x)
                timings.append(dict(rows=len(items), **timing))
                for j, result in enumerate(results):
                    if a.save_heads:
                        name = f"{offset + j:06d}.npz"
                        np.savez_compressed(
                            out / name,
                            head_maps=heads[j],
                            marking_probability=probability[j],
                        )
                        result.update(
                            heads_file=name, heads_sha256=sha256_file(out / name)
                        )
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                offset += len(items)
        np.save(out / "X.npy", np.concatenate(features))
        (out / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        metadata.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            artifact_sha256={
                f.name: sha256_file(f)
                for f in out.iterdir()
                if f.is_file() and f.name != "run.json"
            },
        )
        write_run()
        (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")
        print(
            json.dumps(
                {
                    key: metadata[key]
                    for key in ["status", "rows", "device", "elapsed_seconds"]
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc))
        write_run()
        raise


if __name__ == "__main__":
    main()
