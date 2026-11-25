"""Verify a self-contained inference release using only the Python standard library."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "release_manifest.json").read_text())
    if manifest["version"] != "roi9-inference-release-1":
        raise ValueError("unknown inference release")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("release file escapes root") from exc
        if digest(path) != expected:
            raise ValueError("release file changed: " + relative)
    recipe = json.loads((root / "models/quality/feature_recipe.json").read_text())
    if recipe["checkpoint_sha256"] != digest(root / "models/perception/best_mean.pt"):
        raise ValueError("perception checkpoint differs from frozen quality recipe")
    rows = [
        json.loads(line)
        for line in (root / "examples/manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(set(row["sample_id"] for row in rows)) != len(rows):
        raise ValueError("duplicate example IDs")
    for row in rows:
        if set(row) != {"sample_id", "image", "view", "expected_sha256"} or set(
            row["expected_sha256"]
        ) != {"rgb"}:
            raise ValueError("examples must contain only RGB metadata")
        if (
            digest(root / "examples/rgb" / row["image"])
            != row["expected_sha256"]["rgb"]
        ):
            raise ValueError("example RGB changed")
    return dict(
        status="verified",
        files=len(manifest["files"]),
        rgb_examples=len(rows),
        uses_ground_truth=False,
        source_git=manifest["source_git"],
        scope="File/context verification; not final model/policy adoption or live ROS acceptance",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
