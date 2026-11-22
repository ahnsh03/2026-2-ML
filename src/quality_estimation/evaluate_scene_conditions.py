"""Train RGB-only weather and capture-hour baselines, then score held-out collections."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from common.bridge import sha256_file
from .artifacts import write_json
from .environment_features import NAMES, PROFILE, extract_environment
from .image_io import color_channels, decoded_pixel_sha256, read_image

WEATHER = ("SUNNY", "CLOUDY", "FOGGY", "STORM", "RAINY", "SNOWY")
TARGETS = ("weather", "hour")


def condition_label(value, target):
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "unknown"}):
        return None
    if target == "weather" and isinstance(value, str):
        value = value.upper().replace("WEATHER_TYPE_", "", 1)
        if value in WEATHER:
            return WEATHER.index(value)
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    limit = len(WEATHER) if target == "weather" else 24
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"invalid {target} label: {value!r}")
    return value


def descriptors(manifest, root):
    content = manifest.read_bytes()
    rows, features = [], []
    seen_ids, seen_pixels = set(), set()
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid, parent = row["sample_id"], row["parent_session_id"]
        if not all(isinstance(s, str) and s.strip() for s in (sid, parent)) or row.get("view") != "front":
            raise ValueError("expected a sample ID, parent session and front camera")
        if sid in seen_ids:
            raise ValueError("duplicate sample ID: " + sid)
        seen_ids.add(sid)
        image, file_hash = read_image(root, row["image"], cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(color_channels(image, "RGB"), cv2.COLOR_BGR2RGB)
        pixel_hash = decoded_pixel_sha256(rgb)
        if row.get("expected_sha256", {}).get("rgb", file_hash) != file_hash or row.get("pixel_sha256", pixel_hash) != pixel_hash:
            raise ValueError("RGB identity mismatch: " + sid)
        if pixel_hash in seen_pixels:
            raise ValueError("duplicate decoded RGB: " + sid)
        seen_pixels.add(pixel_hash)
        conditions = row.get("condition") or {}
        labels = {target: condition_label(conditions.get(target), target) for target in TARGETS}
        rows.append(dict(sample_id=sid, parent_session_id=parent, rgb_sha256=file_hash,
                         pixel_sha256=pixel_hash, **labels))
        features.append(extract_environment(rgb))
    if not rows:
        raise ValueError("empty scene manifest")
    return rows, np.stack(features), hashlib.sha256(content).hexdigest()


def check_holdout(training, evaluation):
    for key in ("sample_id", "parent_session_id", "pixel_sha256"):
        overlap = {r[key] for r in training}.intersection(r[key] for r in evaluation)
        if overlap:
            raise ValueError(f"training and evaluation share {key} ({len(overlap)})")


def scores(rows, target, predictions, trained_classes):
    known = [(row[target], int(p)) for row, p in zip(rows, predictions) if row[target] is not None]
    labels = sorted(set(trained_classes).union(truth for truth, _ in known))
    indices = {v: i for i, v in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in known:
        counts[indices[truth], indices[pred]] += 1
    totals = counts.sum(axis=1)
    recalls = [float(counts[i, i] / n) if n else None for i, n in enumerate(totals)]
    return dict(labels=labels, confusion_true_by_predicted=counts.tolist(), per_class_recall=recalls,
                known=len(known), unknown=len(rows) - len(known),
                unseen_in_training=sum(y not in trained_classes for y, _ in known),
                accuracy=float(np.trace(counts) / len(known)) if known else None,
                balanced_accuracy=float(np.mean([r for r in recalls if r is not None])) if known else None)


def evaluate(train_manifest, eval_manifest, root, output):
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(1)
    training, X, train_digest = descriptors(train_manifest, root)
    models = {}
    for target in TARGETS:
        kept = [i for i, r in enumerate(training) if r[target] is not None]
        labels = [training[i][target] for i in kept]
        if len(set(labels)) < 2:
            raise ValueError("training requires two known classes for " + target)
        models[target] = RandomForestClassifier(n_estimators=128, min_samples_leaf=2, random_state=0,
                                                n_jobs=2).fit(X[kept], labels)
    held, test_X, eval_digest = descriptors(eval_manifest, root)
    check_holdout(training, held)
    report = {}
    for target, clf in models.items():
        pred = clf.predict(test_X)
        report[target] = scores(held, target, pred, clf.classes_.tolist())
        for row, value in zip(held, pred):
            row["predicted_" + target] = int(value)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(models, output / "classifiers.joblib", compress=3)
    write_json(output / "training_rows.json", training)
    write_json(output / "predictions.json", held)
    write_json(output / "metrics.json", report)
    hashes = {p.name: sha256_file(p) for p in output.iterdir()}
    write_json(output / "run.json", dict(status="complete", train_manifest_sha256=train_digest,
               evaluation_manifest_sha256=eval_digest, sklearn_version=sklearn.__version__,
               camera_profile=PROFILE, feature_names=NAMES, weather_classes=WEATHER,
               params=dict(n_estimators=128, min_samples_leaf=2, random_state=0),
               train_rows=len(training), evaluation_rows=len(held), artifact_sha256=hashes))
    (output / "_SUCCESS").write_text(sha256_file(output / "run.json") + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-manifest", "eval-manifest", "data-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.train_manifest, args.eval_manifest, args.data_root, args.output_dir)))


if __name__ == "__main__":
    main()
