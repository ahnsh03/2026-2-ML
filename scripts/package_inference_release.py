"""Create a relocatable inference-only handoff, including a real pinned upstream checkout."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from common.bridge import DEFAULT_CHECKPOINT_SHA256
from verify_inference_release import digest, verify


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "repo",
        "quality-bundle",
        "portable-bundle",
        "checkpoint",
        "rgb-root",
        "rgb-manifest",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output.resolve()
    if git(repo, "status", "--porcelain"):
        raise ValueError("package a committed clean source tree")
    if args.examples < 1:
        raise ValueError("positive example count required")
    if digest(args.checkpoint) != DEFAULT_CHECKPOINT_SHA256:
        raise ValueError("this release requires the fixed f0 checkpoint")
    commit = git(repo, "rev-parse", "HEAD")
    upstream = repo / "third_party/TwinLiteNetPlus"
    upstream_commit = git(upstream, "rev-parse", "HEAD")
    if git(upstream, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("upstream tracked source is modified")
    recipe = json.loads((args.quality_bundle / "feature_recipe.json").read_text())
    portable = json.loads((args.portable_bundle / "feature_recipe.json").read_text())
    if (
        recipe["checkpoint_sha256"] != DEFAULT_CHECKPOINT_SHA256
        or portable["checkpoint_sha256"] != DEFAULT_CHECKPOINT_SHA256
    ):
        raise ValueError("model recipe perception version differs")
    out.mkdir(parents=True, exist_ok=False)
    code = out / "code"
    code.mkdir()
    archive = out / "source_git_archive.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            "--output",
            str(archive),
            commit,
            "src",
            "config",
            "docker",
            "scripts",
            ".gitmodules",
            "ros/src/data_collection/src/data_collection",
        ],
        check=True,
    )
    with tarfile.open(archive) as source:
        source.extractall(code, filter="data")
    if not (
        code / "ros/src/data_collection/src/data_collection/mask_baker.py"
    ).is_file():
        raise ValueError("shared pure-Python capture palette helper is missing")
    target = code / "third_party/TwinLiteNetPlus"
    target.parent.mkdir(exist_ok=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.logAllRefUpdates=false",
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            str(upstream),
            str(target),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "remote",
            "set-url",
            "origin",
            git(
                repo,
                "config",
                "--file",
                ".gitmodules",
                "--get",
                "submodule.third_party/TwinLiteNetPlus.url",
            ),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "-c",
            "core.logAllRefUpdates=false",
            "checkout",
            "--quiet",
            "--detach",
            upstream_commit,
        ],
        check=True,
    )
    if git(target, "rev-parse", "HEAD") != upstream_commit:
        raise ValueError("packaged upstream differs")
    if (target / ".git/objects/info/alternates").exists():
        raise ValueError("upstream clone depends on external object storage")
    # Reflogs are local packaging metadata, not upstream history or required source.
    if (target / ".git/logs").exists():
        shutil.rmtree(target / ".git/logs")
    artifacts = {}

    def copy(source, destination):
        destination = out / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(source) != digest(destination):
            raise ValueError("copy checksum differs")
        artifacts[str(destination.relative_to(out))] = digest(destination)

    copy(args.checkpoint, "models/perception/best_mean.pt")
    copy(
        args.quality_bundle / "feature_recipe.json",
        "models/quality/feature_recipe.json",
    )
    for task in recipe["tasks"].values():
        path = args.quality_bundle / task["artifact_filename"]
        if digest(path) != task["model_sha256"]:
            raise ValueError("quality model changed")
        copy(path, "models/quality/" + path.name)
        if "calibration_filename" in task:
            path = args.quality_bundle / task["calibration_filename"]
            if digest(path) != task["calibration_sha256"]:
                raise ValueError("quality calibration changed")
            copy(path, "models/quality/" + path.name)
    copy(
        args.portable_bundle / "feature_recipe.json",
        "models/portable/feature_recipe.json",
    )
    for task in portable["tasks"].values():
        path = args.portable_bundle / task["artifact_filename"]
        if digest(path) != task["model_sha256"]:
            raise ValueError("portable model changed")
        copy(path, "models/portable/" + path.name)
    rows = [
        json.loads(line)
        for line in args.rgb_manifest.read_text().splitlines()
        if line.strip()
    ][: args.examples]
    if len(rows) != args.examples:
        raise ValueError("not enough source RGB examples")
    examples = []
    for i, row in enumerate(rows):
        image = (args.rgb_root / row["image"]).resolve()
        image.relative_to(args.rgb_root.resolve())
        if digest(image) != row["expected_sha256"]["rgb"]:
            raise ValueError("source example changed")
        name = f"{i:04d}" + image.suffix
        copy(image, "examples/rgb/" + name)
        examples.append(
            dict(
                sample_id=row["sample_id"],
                image=name,
                view="front",
                expected_sha256=dict(rgb=digest(image)),
            )
        )
    (out / "examples/manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in examples)
    )
    (out / "README.md").write_text("""# Nine-region inference handoff

This is an inference-only snapshot, not final policy/perception-update/live-ROS adoption. It contains fixed f0/quality28 GP, the separate portable RF/environment tasks and RGB-only examples. The source commit and all checksums are in release_manifest.json. Full private training membership, raw GT and diagnostic Far31/f1 candidates are not included. Use the full research bundle for training provenance.

Verify with `python3 code/scripts/verify_inference_release.py --root .`. Build the matching40/50 image from code/docker if needed. Load models once when reusing them. Example paths in this package are relative; neither D/E storage nor an original workspace path is required.

Run the example in a read-only mount with an empty external output directory:

```bash
docker run --rm --user "$(id -u):$(id -g)" \\
  -v "$RELEASE_ROOT":/bundle:ro -v "$NEW_RESULTS":/results \\
  -e PYTHONPATH=/bundle/code/src -e PYTHONDONTWRITEBYTECODE=1 \\
  -e MPLCONFIGDIR=/tmp/ml-mpl -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \\
  -w /bundle/code ml-perception:rtx40-cu118 python3 -m quality_estimation.infer_spatial_images \\
  --manifest /bundle/examples/manifest.jsonl --data-root /bundle/examples/rgb \\
  --recipe /bundle/models/quality/feature_recipe.json --model /bundle/models/quality/quality_spatial_gp.npz \\
  --calibration /bundle/models/quality/calibration_26r1.json --checkpoint /bundle/models/perception/best_mean.pt \\
  --simulator-release 26.R1 --batch-size 8 --num-workers 0 --device cpu --output-dir /results/replay
```

For a50 GPU use `--gpus all`, the rtx50-cu128 image and `--device cuda`. The40 GPU hardware must be tested on the teammate's PC. Forest-only microbenchmarks are not end-to-end FPS. Candidate queues and deferred GT review remain separate; this package does not approve training admission.
""")
    files = {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        version="roi9-inference-release-1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_git=commit,
        upstream_git=upstream_commit,
        checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
        rgb_examples=len(examples),
        contains_raw_ground_truth=False,
        contains_training_membership=False,
        files=files,
    )
    (out / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(verify(out), indent=2))


if __name__ == "__main__":
    main()
