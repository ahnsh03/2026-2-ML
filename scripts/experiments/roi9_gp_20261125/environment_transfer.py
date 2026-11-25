"""RGB-only transfer check of frozen v2 weather/hour/e-means artifacts; no quality GT."""

from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np, joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.selected_region_features import extract_selected_features


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["index", "data-root", "recipe", "models", "output"]:
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    spec = json.loads(a.recipe.read_text())
    profile = SpatialProfile.load()
    assert (
        spec["profile"] == profile.version
        and spec["regions_sha256"] == profile.regions_sha256
    )
    tasks = {name: spec["tasks"][name] for name in ["weather", "hour", "environment"]}
    columns = list(
        dict.fromkeys(n for t in tasks.values() for n in t["ordered_features"])
    )
    locations = {name: i for i, name in enumerate(columns)}
    index = json.loads(a.index.read_text())
    rows = [
        dict(row, source_key=key)
        for key, value in index.items()
        for row in value["rows"]
    ]
    assert len(set(row["sample_id"] for row in rows)) == len(rows)
    X = []
    seconds = []
    for row in rows:
        image = a.data_root / row["image"]
        assert sha(image) == row["expected_sha256"]["rgb"]
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        X.append(extract_selected_features(rgb, None, profile, columns))
        seconds.append(time.perf_counter() - started)
    X = np.stack(X)
    truth = {key: np.asarray([row[key] for row in rows]) for key in ["weather", "hour"]}
    output = {}
    aggregate = {}
    for name in ["weather", "hour"]:
        task = tasks[name]
        path = a.models / task["artifact_filename"]
        assert sha(path) == task["model_sha256"]
        model = joblib.load(path)
        model.set_params(n_jobs=1)
        inputs = X[:, [locations[n] for n in task["ordered_features"]]]
        assert model.n_features_in_ == inputs.shape[1]
        prediction = model.predict(inputs)
        output[name] = prediction
        labels = np.unique(np.r_[truth[name], prediction])
        aggregate[name] = dict(
            rows=len(rows),
            accuracy=float(accuracy_score(truth[name], prediction)),
            balanced_accuracy=float(balanced_accuracy_score(truth[name], prediction)),
            macro_f1=float(
                f1_score(truth[name], prediction, average="macro", zero_division=0)
            ),
            labels=labels.tolist(),
            confusion=confusion_matrix(truth[name], prediction, labels=labels).tolist(),
            model_sha256=sha(path),
        )
    task = tasks["environment"]
    path = a.models / task["artifact_filename"]
    assert sha(path) == task["model_sha256"]
    with np.load(path, allow_pickle=False) as z:
        assert z["ordered_features"].tolist() == task["ordered_features"]
        raw = X[:, [locations[n] for n in task["ordered_features"]]]
        scaled = (raw[:, z["selected"]] - z["mean"]) / z["scale"]
        distances = np.sqrt(cdist(scaled, z["centers"], "sqeuclidean"))
    cluster = distances.argmin(1)
    output["cluster"] = cluster
    output["center_distance"] = distances.min(1)
    count = np.bincount(cluster, minlength=distances.shape[1])
    aggregate["environment"] = dict(
        method="Frozen official e-means centers, NumPy nearest-center assignment; no center refit",
        clusters=distances.shape[1],
        occupied=int((count > 0).sum()),
        counts=count.tolist(),
        largest_fraction=float(count.max() / len(rows)),
        weather_nmi=float(normalized_mutual_info_score(truth["weather"], cluster)),
        hour_nmi=float(normalized_mutual_info_score(truth["hour"], cluster)),
        center_distance_quantiles=np.quantile(
            output["center_distance"], [0.05, 0.5, 0.95, 1]
        ).tolist(),
        artifact_sha256=sha(path),
    )
    aggregate["by_source"] = {}
    keys = np.asarray([row["source_key"] for row in rows])
    for key in np.unique(keys):
        mask = keys == key
        aggregate["by_source"][str(key)] = dict(
            rows=int(mask.sum()),
            weather_accuracy=float(
                (output["weather"][mask] == truth["weather"][mask]).mean()
            ),
            hour_accuracy=float((output["hour"][mask] == truth["hour"][mask]).mean()),
            weather_nmi=float(
                normalized_mutual_info_score(truth["weather"][mask], cluster[mask])
            ),
            occupied_clusters=len(set(cluster[mask])),
        )
    aggregate.update(
        uses_semantic_or_iou=False,
        refits_models=False,
        recipe_sha256=sha(a.recipe),
        index_sha256=sha(a.index),
        source_sha256=sha(Path(__file__)),
        rgb_feature_columns=len(columns),
        cpu_extraction_p50_p95=np.quantile(seconds, [0.5, 0.95]).tolist(),
        scope="Frozen v2 environmental expression transfer to26R1. Weather/time are evaluation labels only. Accuracy/NMI do not establish improved quality estimation or an operational failure policy.",
    )
    np.savez_compressed(
        a.output / "per_frame.npz",
        X=X,
        ids=np.asarray([row["sample_id"] for row in rows]),
        ordered_features=np.asarray(columns),
        **output,
    )
    (a.output / "summary.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False) + "\n"
    )
    print(
        "frozen environmental transfer",
        len(rows),
        aggregate["weather"]["accuracy"],
        aggregate["hour"]["accuracy"],
        aggregate["environment"]["occupied"],
        flush=True,
    )


if __name__ == "__main__":
    main()
