"""Attach actual pool GT only after acquisition freeze, then freeze Bayesian GP updates."""

from __future__ import annotations
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quality_estimation.conditioned_spatial_gp import ConditionedSpatialGP
from quality_estimation.region_feature_views import feature_matrix
from study import load, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "inputs", "selection", "bundle", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    selection = read(args.selection / "selection.json")
    if (args.selection / "_SUCCESS").read_text().strip() != sha(
        args.selection / "selection.json"
    ) or selection["uses_ground_truth"]:
        raise ValueError("input-only selection must be complete first")
    if sha(args.inputs) != selection["inputs_sha256"]:
        raise ValueError("acquisition input changed")
    recipe = read(args.bundle / "feature_recipe.json")
    task = recipe["tasks"]["quality_spatial_gp"]
    parent = args.bundle / task["artifact_filename"]
    if sha(parent) != selection["parent_model_sha256"]:
        raise ValueError("parent model differs")
    data, schema = load(args.study / "packs", ["release26_heldroads"])
    if (
        read(args.study / "packs/release26_heldroads.json")["checkpoint_sha256"]
        != recipe["checkpoint_sha256"]
    ):
        raise ValueError("pool labels belong to another perception checkpoint")
    with np.load(args.inputs, allow_pickle=False) as z:
        X = z["X"]
        ids = z["ids"]
        if ids.tolist() != data["ids"].tolist() or not np.array_equal(
            X, feature_matrix(data, schema, task["ordered_features"])
        ):
            raise ValueError("label/input row or column order differs")
    args.output.mkdir(exist_ok=False)
    models = []
    for entry in selection["selections"]:
        indices = np.asarray(entry["indices"], int)
        if ids[indices].tolist() != entry["ids"] or len(indices) != entry["budget"]:
            raise ValueError("selected IDs differ")
        updated = ConditionedSpatialGP.fit(parent, X[indices], data["Y"][indices])
        filename = f"{entry['method']}_b{entry['budget']:02d}.npz"
        path = args.output / filename
        updated.save(path)
        models.append(
            dict(
                entry,
                filename=filename,
                sha256=sha(path),
                observed_values=len(updated.observed),
            )
        )
    for name in [
        "src/quality_estimation/conditioned_spatial_gp.py",
        "src/quality_estimation/spatial_gp.py",
        "scripts/experiments/roi9_gp_20261125/acquisition_update.py",
    ]:
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    save(
        args.output / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            selection_sha256=sha(args.selection / "selection.json"),
            parent_model_sha256=sha(parent),
            checkpoint_sha256=recipe["checkpoint_sha256"],
            models=models,
            source_sha256={
                name: sha(Path(name))
                for name in [
                    "src/quality_estimation/conditioned_spatial_gp.py",
                    "src/quality_estimation/spatial_gp.py",
                    "scripts/experiments/roi9_gp_20261125/acquisition_update.py",
                ]
            },
            scope="Quality-GP posterior updates only. Fixed perception, kernels, task coupling, scaling and noise. No perception retraining or policy adoption.",
        ),
    )
    (args.output / "_SUCCESS").write_text(sha(args.output / "frozen.json") + "\n")
    print("Frozen", len(models), "quality-GP updates using actual post-selection GT")


if __name__ == "__main__":
    main()
