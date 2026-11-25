"""Build a GT-free offline candidate queue from a verified nine-region GP replay."""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from .spatial_gp import SpatialQualityGP
from .spatial_risk import conditional_batch
from .spatial_predictor import file_sha
from .spatial_identity import validate_prediction_perception


def select_window(predictions, ids, budget, policy, *, X=None, model=None):
    """Return indices/reasons; inputs contain model scores, never reference targets."""
    ids = np.asarray(ids)
    if not 0 <= budget <= len(ids) or len(predictions) != len(ids):
        raise ValueError("invalid candidate window")
    low = np.asarray(
        [
            -p["predicted_iou"]
            if policy == "global_mean"
            else p["risk_scores"]["global_low"]
            for p in predictions
        ]
    )

    def ranked(score, count, excluded=()):
        return [int(i) for i in np.lexsort((ids, -score)) if i not in excluded][:count]

    if policy in ("global_mean", "global_probability"):
        selected = ranked(low, budget)
        return [(i, "whole_frame_low_overlap_review") for i in selected]
    if policy not in ("dual_probability", "dual_conditional"):
        raise ValueError("unknown spatial review policy")
    first_budget = math.ceil(0.7 * budget)
    remaining_budget = budget - first_budget
    far = np.asarray([p["risk_scores"]["normal_far_only"] for p in predictions])
    if policy == "dual_probability":
        first = ranked(low, first_budget)
        last = ranked(far, remaining_budget, first)
    else:
        if model is None or X is None:
            raise ValueError("conditional policy requires matching GP and ordered X")
        qcov = model.posterior_covariance(X)
        dcov = model.posterior_covariance(X, [0, 1, -1])
        qnoise = model.noise_variance * model.y_scale_[0] ** 2
        dnoise = model.noise_variance * float((model.y_scale_[1:] ** 2).sum())
        first = conditional_batch(low, qcov, first_budget, ids, noise_variance=qnoise)
        for i in first:
            dcov -= np.outer(dcov[:, i], dcov[i, :]) / (max(dcov[i, i], 0) + dnoise)
        available = np.array([i for i in range(len(ids)) if i not in first], int)
        last = [
            int(available[i])
            for i in conditional_batch(
                far[available],
                dcov[np.ix_(available, available)],
                remaining_budget,
                ids[available],
                noise_variance=dnoise,
            )
        ]
    return [(i, "whole_frame_low_overlap_review") for i in first] + [
        (i, "near_recall_good_far_recall_weak_diagnostic") for i in last
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument(
        "--manifest", type=Path, help="For older replays without a saved input manifest"
    )
    ap.add_argument(
        "--policy",
        choices=[
            "global_mean",
            "global_probability",
            "dual_probability",
            "dual_conditional",
        ],
        default="global_probability",
    )
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--budget-fraction", type=float, default=0.1)
    a = ap.parse_args()
    root = a.replay_dir
    run = json.loads((root / "run.json").read_text())
    if run["status"] != "complete" or (
        root / "_SUCCESS"
    ).read_text().strip() != file_sha(root / "run.json"):
        raise ValueError("incomplete replay")
    for name, digest in run["artifact_sha256"].items():
        if file_sha(root / name) != digest:
            raise ValueError("replay artifact checksum differs")
    manifest = a.manifest or root / "input_manifest.jsonl"
    if file_sha(manifest) != run["input_manifest_sha256"]:
        raise ValueError("original RGB manifest differs")
    rows = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [p["sample_id"] for p in predictions] != [p["sample_id"] for p in rows]:
        raise ValueError("replay row order differs")
    perception_identity, perception_weights = validate_prediction_perception(
        predictions, run
    )
    if a.window_size < 1 or not 0 < a.budget_fraction <= 1:
        raise ValueError("invalid window/budget")
    x = np.load(root / "X.npy", allow_pickle=False)
    model = None
    if a.policy == "dual_conditional":
        if a.model is None or file_sha(a.model) != run["model_sha256"]:
            raise ValueError("matching GP is required")
        model = SpatialQualityGP.load(a.model)
        if model.feature_names_.tolist() != run["ordered_features"]:
            raise ValueError("GP feature order differs")
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    selected = []
    start = 0
    windows = []
    while start < len(rows):
        session = rows[start].get("session_id", "recorded_stream")
        end = start
        while (
            end < len(rows)
            and end - start < a.window_size
            and rows[end].get("session_id", "recorded_stream") == session
        ):
            end += 1
        budget = math.ceil(a.budget_fraction * (end - start))
        window = predictions[start:end]
        ids = [p["sample_id"] for p in window]
        choices = select_window(
            window, ids, budget, a.policy, X=x[start:end], model=model
        )
        for index, reason in choices:
            row = rows[start + index]
            prediction = window[index]
            selected.append(
                dict(
                    sample_id=row["sample_id"],
                    image=row["image"],
                    view="front",
                    rgb_sha256=prediction["rgb_sha256"],
                    predicted_iou=prediction["predicted_iou"],
                    risk_scores=prediction["risk_scores"],
                    selection_reason=reason,
                    window=len(windows),
                    session_id=session,
                    status="awaiting_gt_review",
                    admission_approved=False,
                    profile=prediction["profile"],
                    regions_sha256=prediction["regions_sha256"],
                    model_sha256=run["model_sha256"],
                    recipe_sha256=run["recipe_sha256"],
                    checkpoint_sha256=run["checkpoint_sha256"],
                    perception_identity=perception_identity,
                    perception_weights=perception_weights,
                )
            )
        windows.append(dict(session_id=session, rows=end - start, budget=budget))
        start = end
    output = a.output_dir / "candidates.jsonl"
    output.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in selected)
    )
    metadata = dict(
        status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        uses_ground_truth=False,
        policy=a.policy,
        window_size=a.window_size,
        budget_fraction=a.budget_fraction,
        replay_run_sha256=file_sha(root / "run.json"),
        candidates_sha256=file_sha(output),
        rows=len(rows),
        selected=len(selected),
        windows=windows,
        checkpoint_sha256=run["checkpoint_sha256"],
        perception_identity=perception_identity,
        perception_weights=perception_weights,
        scope="Offline review candidates only. Far-band low recall is a spatial/visibility diagnostic. GT admission and perception update remain separate.",
    )
    (a.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (a.output_dir / "_SUCCESS").write_text(file_sha(a.output_dir / "run.json") + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
