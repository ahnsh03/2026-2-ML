#!/usr/bin/env python3
"""Build reusable nine-region candidate packs from hash-verified recorded inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from quality_estimation.image_io import decoded_pixel_sha256
from quality_estimation.region_candidates import candidate_schema, extract_candidates
from quality_estimation.spatial_features import SpatialProfile, UNCERTAINTY_FEATURES


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--reference-packs", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--uncertainty-packs", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = SpatialProfile.load()
    schema = candidate_schema(profile)
    if args.uncertainty_packs:
        schema["version"] = "region-candidates-v2-raw-uncertainty"
        schema["uncertainty"] = list(UNCERTAINTY_FEATURES)
    schema_path = args.output_dir / "schema.json"
    encoded_schema = json.dumps(schema, indent=2)+"\n"
    if schema_path.exists() and schema_path.read_text() != encoded_schema:
        raise ValueError("existing candidate schema differs")
    schema_path.write_text(encoded_schema)
    keys = args.keys or sorted(p.stem for p in args.reference_packs.glob("*.npz")
                              if (args.source_dir / "protocols" / (p.stem+".jsonl")).exists())
    code_path = Path(__file__).resolve().parents[1]/"src/quality_estimation/region_candidates.py"
    for key in keys:
        destination = args.output_dir / (key+".npz")
        if destination.exists():
            receipt = json.loads(destination.with_suffix(".json").read_text())
            if (sha256(destination) != receipt["sha256"]
                    or sha256(code_path) != receipt["extractor_sha256"]):
                raise ValueError("existing candidate pack or extractor differs")
            print("verified", key, flush=True)
            continue
        start = time.perf_counter()
        reference = args.reference_packs / (key+".npz")
        receipt = json.loads(reference.with_suffix(".json").read_text())
        if sha256(reference) != receipt["sha256"]:
            raise ValueError("reference pack checksum differs")
        manifest = args.source_dir / "protocols" / (key+".jsonl")
        if sha256(manifest) != receipt["manifest_sha256"]:
            raise ValueError("source manifest differs")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        with np.load(reference, allow_pickle=False) as archive:
            metadata = {name: archive[name] for name in
                        ("y", "ids", "pixel_sha256", "group", "site", "cohort", "weather", "hour", "xy")}
        ids, maps, sources = [], [], []
        path = args.source_dir / "maps" / (key+".npz")
        paths = [path]
        if key == "stream_new" and not path.exists():
            paths = [path.with_name("nov23_roi10_run_"+suffix+".npz") for suffix in ("a", "b", "c")]
        for path in paths:
            digest = sha256(path)
            expected = receipt["maps_sha256"]
            if digest != (expected[path.name] if isinstance(expected, dict) else expected):
                raise ValueError("cached map checksum differs")
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["checkpoint_sha"]) != "a26311b3af4e92880f632509eff3c81defc1b95135f5b463c7a08e5b6e23d24b":
                    raise ValueError("unexpected perception checkpoint")
                ids.extend(archive["ids"].tolist())
                maps.append(archive["maps"])
            sources.append({"file": path.name, "sha256": digest})
        if ids != metadata["ids"].tolist() or ids != [r["sample_id"] for r in rows]:
            raise ValueError("sample IDs do not align")
        maps = np.concatenate(maps)
        def extract(item):
            index, row = item
            raw = (args.media_root / row["image"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["expected_sha256"]["rgb"]:
                raise ValueError("RGB checksum differs")
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("could not decode RGB")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if decoded_pixel_sha256(rgb) != row["pixel_sha256"]:
                raise ValueError("decoded RGB checksum differs")
            return extract_candidates(rgb, maps[index], profile)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            extracted = list(pool.map(extract, enumerate(rows)))
        arrays = {name: np.stack([record[name] for record in extracted]) for name in extracted[0]}
        # Scene metadata and labels are attached after RGB/map-only extraction.
        arrays.update(metadata)
        uncertainty_sha256 = None
        if args.uncertainty_packs:
            extra_path = args.uncertainty_packs/(key+".npz")
            extra_receipt = json.loads(extra_path.with_suffix(".json").read_text())
            uncertainty_sha256 = sha256(extra_path)
            if uncertainty_sha256 != extra_receipt["sha256"]:
                raise ValueError("raw uncertainty checksum differs")
            with np.load(extra_path, allow_pickle=False) as extra:
                if (not np.array_equal(extra["ids"], metadata["ids"])
                        or extra["ordered_features"].tolist() != list(UNCERTAINTY_FEATURES)
                        or str(extra["checkpoint_sha"]) != "a26311b3af4e92880f632509eff3c81defc1b95135f5b463c7a08e5b6e23d24b"):
                    raise ValueError("raw uncertainty sample, column or checkpoint identity differs")
                values = extra["values"]
                if (values.dtype != np.float32 or values.shape != (len(rows), 3)
                        or not np.isfinite(values).all()):
                    raise ValueError("invalid raw uncertainty summary values")
                arrays["uncertainty"] = values
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        result = dict(key=key, cohort=str(metadata["cohort"][0]), n=len(rows),
                      sha256=sha256(destination), reference_sha256=sha256(reference),
                      manifest_sha256=sha256(manifest), maps=sources,
                      schema_sha256=sha256(schema_path), extractor_sha256=sha256(code_path),
                      raw_uncertainty_sha256=uncertainty_sha256,
                      elapsed_seconds=time.perf_counter()-start)
        destination.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
        print(key, len(rows), round(result["elapsed_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
