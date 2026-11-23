"""Fit front-camera environment centroids from RGBs in a frozen training membership."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from common.bridge import sha256_file
from .artifacts import write_json
from .dataset_io import require
from .environment_clusters import ROI10_HAZE, fit_environment_clusters, rgb_extractor
from .environment_features import PROFILE, extract_environment
from .image_io import color_channels, decoded_pixel_sha256, read_image


def training_descriptors(model_run, manifest, root, *, extractor=extract_environment):
    """Read only training images, in frozen membership order, one row per unique RGB."""
    run_path = model_run / "run.json"
    membership_path = model_run / "train_membership.json"
    run = json.loads(run_path.read_text())
    require(run.get("status") == "complete" and (model_run / "_SUCCESS").read_text().strip() == sha256_file(run_path),
            "quality model run is incomplete or changed")
    digest = sha256_file(membership_path)
    require(run["artifact_sha256"].get(membership_path.name) == digest, "training membership differs")
    members = json.loads(membership_path.read_text())
    ids = [r["sample_id"] for r in members]
    require(bool(ids) and len(set(ids)) == len(ids), "invalid training IDs")
    available = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            sid = row["sample_id"]
            require(sid not in available, "duplicate RGB manifest ID")
            available[sid] = row
    require(set(ids) <= available.keys(), "RGB manifest is missing training IDs")
    features, kept, seen = [], [], set()
    for member in members:
        sid = member["sample_id"]
        row = available[sid]
        require(row.get("view") == "front", "expected front camera")
        image, file_digest = read_image(root, row["image"], cv2.IMREAD_UNCHANGED)
        require(file_digest == member["rgb_sha256"], "training RGB file differs")
        expected = row.get("expected_sha256", {}).get("rgb")
        require(expected is None or expected == file_digest, "RGB manifest file hash differs")
        rgb = cv2.cvtColor(color_channels(image, "RGB"), cv2.COLOR_BGR2RGB)
        pixels = decoded_pixel_sha256(rgb)
        require(member.get("pixel_sha256") in (None, pixels), "training RGB pixels differ")
        if pixels not in seen:
            features.append(extractor(rgb))
            kept.append(dict(sample_id=sid, rgb_sha256=file_digest, pixel_sha256=pixels))
            seen.add(pixels)
    evidence = dict(fit_role="meta_train", model_run_sha256=sha256_file(run_path),
                    membership_sha256=digest, input_manifest_sha256=sha256_file(manifest),
                    membership_rows=len(ids), unique_training_pixels=len(kept))
    return np.stack(features), kept, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True, help="Completed run defining the training IDs and hashes")
    parser.add_argument("--manifest", type=Path, required=True, help="RGB JSONL containing every training ID")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", choices=("kmeans", "emeans"), required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=(PROFILE, ROI10_HAZE), default=PROFILE)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    cv2.setNumThreads(1)
    features, members, source = training_descriptors(args.model_run, args.manifest, args.data_root,
                                                   extractor=rgb_extractor(args.profile))
    with threadpool_limits(limits=1):
        centers = fit_environment_clusters(features, args.clusters, method=args.method, seed=args.seed, profile=args.profile)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "environment_features.npy", features, allow_pickle=False)
    write_json(out / "training_rows.json", members)
    write_json(out / "environment_centers.json", centers)
    hashes = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "run.json", dict(source, status="complete", artifact_sha256=hashes,
                                      method=args.method, clusters=args.clusters, seed=args.seed))
    (out / "_SUCCESS").write_text(sha256_file(out / "run.json") + "\n")
    print(json.dumps(source))


if __name__ == "__main__":
    main()
