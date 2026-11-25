"""Compare frozen RF exports on recorded inputs, including an older NumPy runtime."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from quality_estimation.portable_forest import PortableForest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args, recipe):
    import joblib
    from study import load
    from quality_estimation.region_feature_views import feature_matrix

    output = {}
    data, schema = load(args.study / "packs", ["release26_followup_test"])
    environment = np.load(
        args.study / "environment_transfer26_cpu/per_frame.npz", allow_pickle=False
    )
    for name in ["quality_rf", "weather", "hour"]:
        spec = recipe["tasks"][name]
        if name == "quality_rf":
            X = feature_matrix(data, schema, spec["ordered_features"])
            parent = args.quality_model_root / "quality_rf.joblib"
        else:
            columns = environment["ordered_features"].tolist()
            X = environment["X"][
                :, [columns.index(n) for n in spec["ordered_features"]]
            ]
            parent = args.environment_model_root / (name + ".joblib")
        if sha(parent) != spec["parent_model_sha256"]:
            raise ValueError("reference parent changed: " + name)
        model = joblib.load(parent)
        model.set_params(n_jobs=1)
        output[name + "_X"] = X
        output[name + "_prediction"] = model.predict(X)
        if name != "quality_rf":
            output[name + "_probability"] = model.predict_proba(X)
    spec = recipe["tasks"]["environment"]
    columns = environment["ordered_features"].tolist()
    X = environment["X"][:, [columns.index(n) for n in spec["ordered_features"]]]
    with np.load(args.bundle / spec["artifact_filename"], allow_pickle=False) as z:
        scaled = (X[:, z["selected"]] - z["mean"]) / z["scale"]
        squared = ((scaled[:, None] - z["centers"][None]) ** 2).sum(2)
    output.update(
        environment_X=X,
        environment_prediction=squared.argmin(1),
        environment_distance=np.sqrt(squared.min(1)),
    )
    np.savez_compressed(args.inputs, **output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    for name in ["bundle", "inputs", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ["study", "quality-model-root", "environment-model-root"]:
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    recipe = json.loads((args.bundle / "feature_recipe.json").read_text())
    if args.prepare:
        if args.inputs.exists():
            parser.error("refusing to overwrite reference inputs")
        if any(
            getattr(args, name) is None
            for name in ["study", "quality_model_root", "environment_model_root"]
        ):
            parser.error("preparation requires study and both parent model roots")
        prepare(args, recipe)
    reference = np.load(args.inputs, allow_pickle=False)
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        recipe_sha256=sha(args.bundle / "feature_recipe.json"),
        inputs_sha256=sha(args.inputs),
        tasks={},
    )
    for name, spec in recipe["tasks"].items():
        path = args.bundle / spec["artifact_filename"]
        if sha(path) != spec["model_sha256"]:
            raise ValueError("portable artifact changed")
        X = reference[name + "_X"]
        started = time.perf_counter()
        if name == "environment":
            with np.load(path, allow_pickle=False) as z:
                if z["ordered_features"].tolist() != spec["ordered_features"]:
                    raise ValueError("environment input order differs")
                scaled = (X[:, z["selected"]] - z["mean"]) / z["scale"]
                squared = ((scaled[:, None] - z["centers"][None]) ** 2).sum(2)
            prediction = squared.argmin(1)
            delta = float(
                abs(np.sqrt(squared.min(1)) - reference["environment_distance"]).max()
            )
        else:
            model = PortableForest.load(path)
            model.validate_context(
                spec["ordered_features"],
                recipe["profile"],
                recipe["regions_sha256"],
                recipe["checkpoint_sha256"] if name == "quality_rf" else None,
            )
            prediction = model.predict(X)
            delta = float(abs(prediction - reference[name + "_prediction"]).max())
        result = dict(
            rows=len(X),
            dimensions=X.shape[1],
            maximum_prediction_difference=delta,
            inference_batch_seconds=time.perf_counter() - started,
            artifact_sha256=sha(path),
        )
        if name != "quality_rf":
            changed = int((prediction != reference[name + "_prediction"]).sum())
            result["changed_class_ids"] = changed
            if changed:
                raise AssertionError("class IDs changed")
        if name in ["weather", "hour"]:
            probability_difference = float(
                abs(model.predict_proba(X) - reference[name + "_probability"]).max()
            )
            result["maximum_probability_difference"] = probability_difference
            if probability_difference > 1e-12:
                raise AssertionError("probability export differs")
        if delta > 1e-12:
            raise AssertionError("inference export differs")
        report["tasks"][name] = result
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
