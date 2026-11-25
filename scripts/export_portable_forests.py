"""Export frozen quality/environment RFs without changing fitted trees or active GP."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import joblib
from quality_estimation.portable_forest import PortableForest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "quality-recipe",
        "quality-model-root",
        "environment-recipe",
        "environment-model-root",
        "output",
    ]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    quality = json.loads(args.quality_recipe.read_text())
    environment = json.loads(args.environment_recipe.read_text())
    for key in ["profile", "regions_sha256", "checkpoint_sha256"]:
        if quality[key] != environment[key]:
            parser.error("quality and environment parent contexts differ")
    args.output.mkdir(exist_ok=False)
    result = {
        key: quality[key] for key in ["profile", "regions_sha256", "checkpoint_sha256"]
    }
    result.update(
        version="roi9-portable-forest-1",
        tasks={},
        quality_parent_recipe_sha256=sha(args.quality_recipe),
        environment_parent_recipe_sha256=sha(args.environment_recipe),
        status="Inference-only export of frozen fitted trees. No retraining, policy adoption or GP replacement.",
    )
    for name in ["quality_rf", "weather", "hour", "environment"]:
        parent = quality if name == "quality_rf" else environment
        root = (
            args.quality_model_root
            if name == "quality_rf"
            else args.environment_model_root
        )
        spec = copy.deepcopy(parent["tasks"][name])
        path = root / spec["artifact_filename"]
        if sha(path) != spec["model_sha256"]:
            raise ValueError("parent model checksum differs: " + name)
        spec["parent_model_sha256"] = spec["model_sha256"]
        destination = args.output / (name + ".npz")
        if name == "environment":
            shutil.copyfile(path, destination)
        else:
            model = joblib.load(path)
            context = dict(
                profile=parent["profile"],
                regions_sha256=parent["regions_sha256"],
                checkpoint_sha256=parent["checkpoint_sha256"]
                if name == "quality_rf"
                else None,
            )
            portable = PortableForest.from_sklearn(
                model, spec["ordered_features"], context
            )
            portable.save(destination)
            spec.update(
                format="portable-random-forest-1",
                runtime="NumPy-only finite-input inference; no sklearn/joblib loading",
            )
        spec.update(artifact_filename=destination.name, model_sha256=sha(destination))
        result["tasks"][name] = spec
    (args.output / "feature_recipe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print("Exported quality RF, weather RF, hour RF and unchanged environment centers")


if __name__ == "__main__":
    main()
