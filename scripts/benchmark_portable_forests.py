"""Measure warmed NumPy-forest inference only; this is not a camera/ROS FPS test."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "inputs", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1:
        parser.error("at least3timed and1warmup calls required")
    recipe = json.loads((args.bundle / "feature_recipe.json").read_text())
    records = []
    with np.load(args.inputs, allow_pickle=False) as z:
        for name in ["quality_rf", "weather", "hour"]:
            spec = recipe["tasks"][name]
            path = args.bundle / spec["artifact_filename"]
            if sha(path) != spec["model_sha256"]:
                raise ValueError("model checksum differs")
            model = PortableForest.load(path)
            for batch in [1, 8, 64]:
                X = z[name + "_X"][:batch]
                expected = z[name + "_prediction"][:batch]
                for _ in range(args.warmup):
                    model.predict(X)
                seconds = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    prediction = model.predict(X)
                    seconds.append(time.perf_counter() - started)
                if not np.allclose(prediction, expected, atol=1e-12, rtol=0):
                    raise AssertionError(
                        "benchmark prediction differs from original RF"
                    )
                records.append(
                    dict(
                        task=name,
                        batch=len(X),
                        median_ms=1000 * float(np.median(seconds)),
                        p95_ms=1000 * float(np.quantile(seconds, 0.95)),
                        max_prediction_difference=float(
                            abs(prediction - expected).max()
                        ),
                    )
                )
    result = dict(
        scope="Warmed forest.predict only: no feature extraction, image inference, IO or ROS. No GPU-speed claim.",
        python=platform.python_version(),
        numpy=np.__version__,
        repeats=args.repeats,
        warmup=args.warmup,
        recipe_sha256=sha(args.bundle / "feature_recipe.json"),
        input_sha256=sha(args.inputs),
        forest_source_sha256=sha(
            Path(__file__).resolve().parents[1]
            / "src/quality_estimation/portable_forest.py"
        ),
        benchmark_source_sha256=sha(Path(__file__)),
        records=records,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
