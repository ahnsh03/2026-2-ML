"""Build RGB-only or paired ROI10 study packs, fit the regional challenger, and score independent packs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from .roi10 import FrontProfile, PROFILE_DIR, extract_region_nodes, reduce_heads, summarize_head_maps, weather_region_views
from .roi10_quality import RegionalQualityModel, label_counts, local_inputs


def _schema():
    from common.bridge import DEFAULT_CHECKPOINT_SHA256
    return {"schema": "f0-roi10-regional-v1", "checkpoint_sha256": DEFAULT_CHECKPOINT_SHA256,
            "profile_sha256": hashlib.sha256((PROFILE_DIR / "profile.json").read_bytes()).hexdigest()}


def _publish(folder, record, files):
    record["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    run = folder / "run.json"
    run.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (folder / "_SUCCESS").write_text(hashlib.sha256(run.read_bytes()).hexdigest() + "\n")


def _completed(folder, filename):
    run = folder / "run.json"
    if (folder / "_SUCCESS").read_text().strip() != hashlib.sha256(run.read_bytes()).hexdigest():
        raise ValueError("incomplete ROI10 artifact")
    meta = json.loads(run.read_text())
    if any(meta.get(k) != v for k, v in _schema().items()):
        raise ValueError("ROI10 extractor or perception identity differs")
    if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != meta["sha256"][filename]:
        raise ValueError("ROI10 artifact checksum mismatch")
    return meta


def build(args):
    import cv2
    import torch
    from torch.utils.data import DataLoader

    from common.bridge import load_target_model
    from .build_dataset import collate, native_marking, worker_init
    from .features import extract_features
    from .image_io import color_channels, decoded_pixel_sha256, read_image
    from .infer_images import RGBSamples
    from .perception_identity import resolve_quality_perception

    # Initialize CPU thread counts before OpenCV resizes the profile and workers fork.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    for row in rows:
        if "input_sha256" in row:
            if "expected_sha256" in row and row["expected_sha256"] != row["input_sha256"]:
                raise ValueError("conflicting source hashes")
            row["expected_sha256"] = row["input_sha256"]
        if args.purpose == "training" and row.get("preregistered_quality_role") not in (None, "meta_train"):
            raise ValueError("non-training role in training manifest")
    ids = [r["sample_id"] for r in rows]
    split_groups = [r.get("split_group") for r in rows]
    has_groups = any(g is not None for g in split_groups)
    if has_groups and not all(isinstance(g, str) and g.strip() for g in split_groups):
        raise ValueError("split_group must be present and nonempty on every row when supplied")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate RGB IDs")
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("invalid batch size or workers")
    args.output.mkdir(parents=True, exist_ok=False)
    profile = FrontProfile.load()
    regions = profile.native[::2, ::2]
    region_pixels = np.bincount(regions.ravel(), minlength=10)
    checkpoint, _, _ = resolve_quality_perception(checkpoint=args.checkpoint)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_target_model(checkpoint, device).model
    loader = DataLoader(RGBSamples(args.data_root, rows), batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate, worker_init_fn=worker_init)
    xs, nodes, heads, predicted, gt, pixels = [], [], [], [], [], []
    weather_views = []
    with torch.inference_mode():
        for images, items in loader:
            da, mark = model(images.to(device))
            drive = da.argmax(1).cpu().numpy()
            marking = mark.float().softmax(1).cpu().numpy()
            for item, drivable, q in zip(items, drive, marking):
                row = rows[item["index"]]
                rgb, geometry = item["rgb"], item["geometry"]
                if row.get("pixel_sha256", item["pixel_sha256"]) != item["pixel_sha256"]:
                    raise ValueError("decoded RGB mismatch for " + row["sample_id"])
                if row.get("expected_pixel_sha256"):
                    order = row.get("expected_pixel_color_order", "RGB")
                    if order not in ("RGB", "BGR"):
                        raise ValueError("unrecognized source pixel ordering")
                    source_hash = item["pixel_sha256"] if order == "RGB" else decoded_pixel_sha256(rgb[:, :, ::-1])
                    if source_hash != row["expected_pixel_sha256"]:
                        raise ValueError("original decoded pixel digest differs")
                pixels.append(item["pixel_sha256"])
                xs.append(extract_features(rgb, q, geometry))
                nodes.append(extract_region_nodes(rgb, profile))
                maps = reduce_heads(drivable, q, geometry)
                heads.append(summarize_head_maps(maps, profile))
                if args.weather_views:
                    weather_views.append(weather_region_views(rgb, maps[0], profile))
                cls = q[:, 12:372].argmax(0)
                lane = (cls == 1) | (cls == 2)
                predicted.append(np.bincount(regions[lane], minlength=10))
                if args.paired:
                    sem, digest = read_image(args.data_root, row["semantic"], cv2.IMREAD_UNCHANGED)
                    if row.get("expected_sha256", {}).get("semantic", digest) != digest:
                        raise ValueError("Semantic hash mismatch for " + row["sample_id"])
                    if sem.shape[:2] != rgb.shape[:2]:
                        raise ValueError("Semantic and RGB dimensions differ")
                    classes = native_marking(color_channels(sem, "Semantic"))[::2, ::2]
                    gt.append(label_counts(lane, (classes == 1) | (classes == 2), regions))
    pack = dict(ids=ids, pixel_sha256=pixels, x56=xs, nodes=nodes, heads=heads,
                predicted_pixels=predicted, region_pixels=region_pixels)
    if args.weather_views:
        pack["weather_views"] = weather_views
        if args.purpose == "training":
            from .evaluate_scene_conditions import condition_label
            labels = [condition_label((r.get("condition") or {}).get("weather"), "weather") for r in rows]
            pack["weather"] = [-1 if label is None else label for label in labels]
    if has_groups:
        pack["split_group"] = split_groups
    if args.paired:
        pack["counts"] = gt
    output = args.output / "rows.npz"
    np.savez_compressed(output, **pack)
    _publish(args.output, dict(**_schema(), purpose=args.purpose, paired=args.paired, rows=len(ids),
                               manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest()), [output])


def _inputs(z):
    return local_inputs(z["x56"], z["nodes"], z["heads"], z["predicted_pixels"], z["region_pixels"])


def fit(args):
    spec = _completed(args.data, "rows.npz")
    if spec["purpose"] != "training" or not spec["paired"]:
        raise ValueError("fit needs a paired training pack")
    with np.load(args.data / "rows.npz", allow_pickle=False) as z:
        if len(np.unique(z["pixel_sha256"])) != len(z["ids"]):
            raise ValueError("duplicate decoded training pixels")
        all_counts = z["counts"]
        valid = all_counts.sum((1, 2)) > 0
        model = RegionalQualityModel.fit(_inputs(z)[valid], z["predicted_pixels"][valid], z["region_pixels"],
                                         all_counts[valid], jobs=args.jobs)
        membership = {"ids": z["ids"], "pixel_sha256": z["pixel_sha256"]}
        if "split_group" in z:
            membership["split_group"] = np.unique(z["split_group"])
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "model.joblib"
    joblib.dump(model, output, compress=2)
    members = args.output / "training.npz"
    np.savez_compressed(members, **membership)
    import sklearn
    _publish(args.output, dict(**_schema(), fit_rows=int(valid.sum()), sklearn=sklearn.__version__,
                               training_pack_sha256=spec["sha256"]["rows.npz"]), [output, members])


def fit_centers(args):
    from threadpoolctl import threadpool_limits
    from .environment_clusters import ROI10_HAZE, fit_environment_clusters

    spec = _completed(args.data, "rows.npz")
    if spec["purpose"] != "training":
        raise ValueError("fit-centers needs a training pack")
    with np.load(args.data / "rows.npz", allow_pickle=False) as pack:
        if len(np.unique(pack["pixel_sha256"])) != len(pack["ids"]):
            raise ValueError("duplicate decoded training pixels")
        features = pack["nodes"][:, :, [1, 3, 4]].reshape(-1, 30)
    with threadpool_limits(limits=1):
        centers = fit_environment_clusters(features, args.clusters, method=args.method,
                                          seed=args.seed, profile=ROI10_HAZE)
    args.output.mkdir(parents=True, exist_ok=False)
    dest = args.output / "environment_centers.json"
    dest.write_text(json.dumps(centers, indent=2, allow_nan=False) + "\n")
    _publish(args.output, dict(**_schema(), method=args.method, k=args.clusters, seed=args.seed,
                               training_rows=len(features), training_pack_sha256=spec["sha256"]["rows.npz"]), [dest])


def _check_membership(z, tr, purpose):
    overlap = int(np.count_nonzero(np.isin(z["ids"], tr["ids"]) |
                                   np.isin(z["pixel_sha256"], tr["pixel_sha256"])))
    independent = purpose in ("development", "calibration", "evaluation")
    if independent and overlap:
        raise ValueError("independent scoring pack overlaps training IDs or pixels")
    if independent and "split_group" in tr:
        if "split_group" not in z:
            raise ValueError("training used split groups; the independent scoring pack also needs groups")
        if np.intersect1d(tr["split_group"], z["split_group"]).size:
            raise ValueError("independent scoring pack overlaps training split groups")
    return overlap


def fit_weather(args):
    from .roi10_weather import BandWeatherModel

    spec = _completed(args.data, "rows.npz")
    if spec["purpose"] != "training":
        raise ValueError("fit-weather needs a training pack")
    with np.load(args.data / "rows.npz", allow_pickle=False) as z:
        if not {"weather_views", "weather", "split_group"}.issubset(z.files):
            raise ValueError("build a training pack with --weather-views, condition.weather and split_group")
        if len(np.unique(z["pixel_sha256"])) != len(z["ids"]):
            raise ValueError("duplicate decoded training pixels")
        labels = z["weather"]
        keep = labels >= 0
        model = BandWeatherModel.fit(z["weather_views"][keep], labels[keep], z["split_group"][keep],
                                     jobs=args.jobs)
        membership = dict(ids=z["ids"], pixel_sha256=z["pixel_sha256"], split_group=np.unique(z["split_group"]))
    args.output.mkdir(parents=True, exist_ok=False)
    output, members = args.output / "weather.joblib", args.output / "training.npz"
    joblib.dump(model, output, compress=2)
    np.savez_compressed(members, **membership)
    import sklearn
    _publish(args.output, dict(**_schema(), labeled_training_rows=int(keep.sum()), sklearn=sklearn.__version__,
                               training_pack_sha256=spec["sha256"]["rows.npz"]), [output, members])


def score_weather(args):
    fitted = _completed(args.model, "weather.joblib")
    _completed(args.model, "training.npz")
    data = _completed(args.data, "rows.npz")
    model = joblib.load(args.model / "weather.joblib")
    with np.load(args.data / "rows.npz", allow_pickle=False) as z, np.load(
            args.model / "training.npz", allow_pickle=False) as training:
        if "weather_views" not in z:
            raise ValueError("build the input pack with --weather-views")
        overlap = _check_membership(z, training, data["purpose"])
        probabilities = model.predict_proba(z["weather_views"])
        classes = model.combiner.classes_
        out = dict(ids=z["ids"], pixel_sha256=z["pixel_sha256"], classes=classes,
                   probabilities=probabilities, predicted_weather=classes[probabilities.argmax(1)])
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "weather_scores.npz"
    np.savez_compressed(output, **out)
    _publish(args.output, dict(**_schema(), uses_weather_label_for_prediction=False,
                               uses_gt_for_prediction=False, overlapping_training_rows=overlap,
                               input_sha256=data["sha256"]["rows.npz"], model_sha256=fitted["sha256"]["weather.joblib"]),
             [output])


def score(args):
    fit_run = _completed(args.model, "model.joblib")
    _completed(args.model, "training.npz")
    data = _completed(args.data, "rows.npz")
    # Trained bundles are trusted local scikit-learn artifacts. The numeric data packs disable pickle.
    model = joblib.load(args.model / "model.joblib")
    with np.load(args.data / "rows.npz", allow_pickle=False) as z:
        with np.load(args.model / "training.npz", allow_pickle=False) as tr:
            overlap = _check_membership(z, tr, data["purpose"])
        out = model.predict(_inputs(z), z["predicted_pixels"], z["region_pixels"])
        out.update(ids=z["ids"], pixel_sha256=z["pixel_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    dest = args.output / "scores.npz"
    np.savez_compressed(dest, **out)
    _publish(args.output, dict(**_schema(), uses_gt_for_prediction=False, overlapping_training_rows=overlap,
                               input_sha256=data["sha256"]["rows.npz"], model_sha256=fit_run["sha256"]["model.joblib"],
                               training_pack_sha256=fit_run["training_pack_sha256"]), [dest])


def evaluate_paired_scores(labels, scores, alarm_cutoff=None):
    """Join delayed Semantic labels by both frame ID and decoded RGB, preserving the original review pool."""
    from .roi10 import REGIONS

    source_ids, pairs = scores["ids"], labels["ids"]
    if len(set(source_ids)) != len(source_ids) or len(set(pairs)) != len(pairs) or not len(pairs):
        raise ValueError("scored and paired frame IDs must be unique and the label pack nonempty")
    lookup = {sample_id: i for i, sample_id in enumerate(source_ids)}
    if any(sample_id not in lookup for sample_id in pairs):
        raise ValueError("a paired frame was never scored")
    indices = np.asarray([lookup[sid] for sid in pairs])
    if not np.array_equal(scores["pixel_sha256"][indices], labels["pixel_sha256"]):
        raise ValueError("paired RGB pixels differ from the scored frame")
    predicted = scores["estimated_iou"]
    if (predicted.shape != (len(source_ids),) or not np.isfinite(predicted).all()
            or np.any((predicted < 0) | (predicted > 1))):
        raise ValueError("invalid IoU predictions")
    counts = labels["counts"]
    if counts.shape != (len(pairs), 10, 3) or np.any(counts < 0):
        raise ValueError("invalid paired regional tallies")
    union = counts.sum(axis=2)
    denominator = union.sum(axis=1)
    defined = denominator > 0
    truth = np.divide(counts[:, :, 0].sum(1), denominator, out=np.full(len(pairs), np.nan), where=defined)
    error = predicted[indices][defined] - truth[defined]
    budget = (len(predicted) + 9) // 10
    selected = np.lexsort((source_ids, predicted))[:budget]
    joined = np.full(len(predicted), np.nan)
    joined[indices] = truth
    usable = np.isfinite(joined[selected])
    regional = {}
    for j, name in enumerate(REGIONS):
        keep = union[:, j] >= 32
        local = counts[keep, j, 0] / union[keep, j]
        est = scores["per_region_iou"][indices[keep], j]
        regional[name] = dict(supported=int(keep.sum()),
                              mae=float(np.mean(np.abs(est - local))) if keep.any() else None)
    report = dict(scored=len(predicted), paired=len(pairs), defined=int(defined.sum()),
                  scored_without_pair=len(predicted) - len(pairs), empty_true_union=int((~defined).sum()),
                  mae=float(np.abs(error).mean()) if error.size else None,
                  bias=float(error.mean()) if error.size else None, low_iou_truth=0.7,
                  pooled_lowest_ten_percent=dict(budget=budget, known=int(usable.sum()),
                      confirmed_low=int(np.sum(joined[selected] < .7)), unknown_truth=int((~usable).sum())),
                  region_mae_minimum_union_32=regional)
    if alarm_cutoff is not None:
        if not 0 <= alarm_cutoff <= 1:
            raise ValueError("alarm cutoff must be a finite number between zero and one")
        low = truth[defined] < .7
        alarm = predicted[indices][defined] < alarm_cutoff
        report["alarm"] = dict(cutoff=alarm_cutoff, tp=int((alarm & low).sum()), fn=int((~alarm & low).sum()),
                               fp=int((alarm & ~low).sum()), tn=int((~alarm & ~low).sum()))
    return report


def evaluate(args):
    label_meta = _completed(args.data, "rows.npz")
    if not label_meta["paired"]:
        raise ValueError("evaluation needs a paired Semantic pack")
    score_meta = _completed(args.scores, "scores.npz")
    if score_meta.get("uses_gt_for_prediction") is not False:
        raise ValueError("the scores have no record of ground-truth-free prediction")
    with np.load(args.data / "rows.npz", allow_pickle=False) as labels, np.load(
            args.scores / "scores.npz", allow_pickle=False) as scores:
        result = evaluate_paired_scores(labels, scores, args.alarm_cutoff)
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = args.output / "metrics.json"
    metrics.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _publish(args.output, dict(**_schema(), score_sha256=score_meta["sha256"]["scores.npz"],
                               paired_pack_sha256=label_meta["sha256"]["rows.npz"]), [metrics])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifest", required=True, type=Path)
    b.add_argument("--data-root", required=True, type=Path)
    b.add_argument("--purpose", required=True, choices=("training", "development", "calibration", "evaluation", "inference"))
    b.add_argument("--paired", action="store_true", help="Read Semantic labels under the existing v6 target policy")
    b.add_argument("--weather-views", action="store_true", help="Also extract ten views for the band weather model")
    b.add_argument("--checkpoint")
    b.add_argument("--device", default="cuda")
    b.add_argument("--batch-size", default=4, type=int)
    b.add_argument("--workers", default=2, type=int)
    f = sub.add_parser("fit")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--jobs", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--model", type=Path, required=True)
    centers = sub.add_parser("fit-centers")
    centers.add_argument("--data", type=Path, required=True)
    centers.add_argument("--method", choices=("emeans", "kmeans"), required=True)
    centers.add_argument("--clusters", type=int, default=8)
    centers.add_argument("--seed", type=int, default=0)
    fw = sub.add_parser("fit-weather")
    fw.add_argument("--data", type=Path, required=True)
    fw.add_argument("--jobs", type=int, default=1)
    sw = sub.add_parser("score-weather")
    sw.add_argument("--data", type=Path, required=True)
    sw.add_argument("--model", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--scores", type=Path, required=True)
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--alarm-cutoff", type=float)
    for step in (b, f, s, centers, fw, sw, ev):
        step.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output
    if destination.exists():
        parser.error("output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".roi10-") as work:
        args.output = Path(work) / "artifact"
        {"build": build, "fit": fit, "score": score, "fit-centers": fit_centers,
         "fit-weather": fit_weather, "score-weather": score_weather, "evaluate": evaluate}[args.command](args)
        args.output.rename(destination)


if __name__ == "__main__":
    main()
