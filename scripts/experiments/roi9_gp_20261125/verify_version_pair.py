"""Audit a completed f1/g1 functional replay; this does not approve a model update."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from common.bridge import DEFAULT_CHECKPOINT_SHA256
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.native import NativeSamples
from quality_estimation.infer_spatial_images import SpatialRGBScorer
from quality_estimation.join_spatial_gt import join_candidates
from study import load, read, save, sha


def rejected(call):
    try:
        call()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("mismatched model versions were accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "study",
        "data-root",
        "adaptation-run",
        "baseline-checkpoint",
        "output",
    ]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root = args.study / "version_pair"
    bundle = root / "f1_g1_bundle"
    packs, schema = load(root / "f1_packs", ["release26_followup_test"])
    recipe = read(bundle / "feature_recipe.json")
    columns = recipe["tasks"]["quality_spatial_gp"]["ordered_features"]
    replay = root / "isolated_replay_outputs/f1_run96_batch16"
    run = read(replay / "run.json")
    if run["status"] != "complete" or (replay / "_SUCCESS").read_text().strip() != sha(
        replay / "run.json"
    ):
        raise ValueError("incomplete f1 replay")
    for name, digest in run["artifact_sha256"].items():
        if sha(replay / name) != digest:
            raise ValueError("f1 replay artifact changed")
    X = np.load(replay / "X.npy")
    expected = feature_matrix(packs, schema, columns)
    delta = float(abs(X - expected).max())
    assert delta < 1e-6
    other = root / "isolated_replay_outputs/f1_run96"
    other_X = np.load(other / "X.npy")
    other_lines = [
        json.loads(line)
        for line in (other / "predictions.jsonl").read_text().splitlines()
    ]
    cross_batch_difference = float(abs(other_X - X).max())
    cross_batch_point_difference = float(
        abs(
            np.asarray([line["raw_mean"] for line in other_lines])
            - np.asarray(
                [
                    line["raw_mean"]
                    for line in [
                        json.loads(v)
                        for v in (replay / "predictions.jsonl").read_text().splitlines()
                    ]
                ]
            )
        ).max()
    )
    lines = [
        json.loads(v) for v in (replay / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in lines] == packs["ids"].tolist()
    records = read(root / "extraction_index.json")["release26_followup_test"]["rows"]
    dataset = NativeSamples(args.data_root, records)
    mask = SpatialProfile.load().native[::2, ::2]
    differing_counts = 0
    cross_batch_count_mismatches = 0
    label_delta = 0.0
    for i, line in enumerate(lines):
        record = dataset[i]
        path = replay / line["heads_file"]
        assert sha(path) == line["heads_sha256"]
        with np.load(path, allow_pickle=False) as z:
            pred = np.isin(z["marking_probability"][:, 12:372].argmax(0), (1, 2))
        gt = np.isin(record["target"][12:372], (1, 2))
        counts = np.stack(
            [
                np.bincount(mask[m], minlength=9)
                for m in [pred & gt, pred & ~gt, ~pred & gt]
            ],
            1,
        )
        differing_counts += int(not np.array_equal(counts, packs["counts"][i]))
        with np.load(other / other_lines[i]["heads_file"], allow_pickle=False) as z:
            other_pred = np.isin(z["marking_probability"][:, 12:372].argmax(0), (1, 2))
        cross_batch_count_mismatches += int(not np.array_equal(other_pred, pred))
        tp, fp, fn = counts.sum(0)
        q = float(tp / (tp + fp + fn))
        label_delta = max(label_delta, abs(q - float(packs["y"][i])))
    assert differing_counts == 0 and label_delta < 1e-6
    with np.load(bundle / "diagnostic.npz") as z:
        raw = np.asarray([line["raw_mean"] for line in lines])
        prediction_delta = float(abs(raw - z["mean"]).max())
    assert prediction_delta < 1e-6
    f0 = root / "isolated_replay_outputs/f0_run16"
    original = args.study / "gallery26_replay"
    baseline_X_difference = float(
        abs(np.load(f0 / "X.npy") - np.load(original / "X.npy")).max()
    )
    assert baseline_X_difference < 1e-8
    candidates = [
        json.loads(v)
        for v in (root / "f1_queue/candidates.jsonl").read_text().splitlines()
    ]
    gtrows = [
        json.loads(v)
        for v in (root / "f1_gt_review/gt_review.jsonl").read_text().splitlines()
    ]
    assert len(candidates) == len(gtrows) == 12 and all(
        not row["admission_approved"] for row in gtrows
    )
    baseline = args.study / "frozen_bundle_final"
    mismatch = {
        "f1_with_g0": rejected(
            lambda: SpatialRGBScorer(
                baseline / "feature_recipe.json",
                baseline / "quality_spatial_gp.npz",
                None,
                "cpu",
                adaptation_run=args.adaptation_run,
            )
        ),
        "f0_with_g1": rejected(
            lambda: SpatialRGBScorer(
                bundle / "feature_recipe.json",
                bundle / "quality_spatial_gp.npz",
                args.baseline_checkpoint,
                "cpu",
            )
        ),
        "f0_labels_with_f1_queue": rejected(
            lambda: join_candidates(
                candidates,
                packs,
                records,
                checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                profile=schema["profile"],
                regions_sha256=schema["regions_sha256"],
            )
        ),
    }
    result = dict(
        scope="Functional version-pair integration on previously inspected same-map data. No new perception training, fair f0/f1 quality comparison, adoption or live ROS acceptance.",
        replay_rows=96,
        f1_feature_max_difference=delta,
        f1_count_mismatch_frames=differing_counts,
        f1_iou_label_max_difference=label_delta,
        f1_gp_point_max_difference=prediction_delta,
        cross_batch_16_vs_8=dict(
            maximum_feature_difference=cross_batch_difference,
            maximum_gp_point_difference=cross_batch_point_difference,
            changed_lane_mask_frames=cross_batch_count_mismatches,
        ),
        f0_regression_rows=16,
        f0_feature_max_difference=baseline_X_difference,
        queued_and_gt_joined=12,
        admission_approved=False,
        mismatch_rejections=mismatch,
        perception_identity=recipe["perception_identity"],
        summary=read(bundle / "summary.json"),
        source_sha256=sha(Path(__file__)),
    )
    save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
