"""Audit environment-policy sample budget and near-band retention, without fitting policy.

Near retention is a spatial diagnostic, not an operational-normal annotation.
Only the existing calibration cohorts are used; all thresholds are sensitivity checks.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from study import load, save, sha


def divide(a, b):
    return np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    r = args.study
    data, _ = load(r / "packs", ["release26_calibration", "release26_far_calibration"])
    path = r / "environment_transfer26_cpu/per_frame.npz"
    with np.load(path, allow_pickle=False) as z:
        lookup = {str(sid): i for i, sid in enumerate(z["ids"])}
        clusters = z["cluster"][[lookup[str(sid)] for sid in data["ids"]]]
    k = 16
    near = data["counts"][:, 6:8].sum(1)
    total = data["counts"].sum(1)
    near_gt = near[:, 0] + near[:, 2]
    near_pred = near[:, 0] + near[:, 1]
    near_recall = divide(near[:, 0], near_gt)
    near_precision = divide(near[:, 0], near_pred)
    precision = divide(total[:, 0], total[:, 0] + total[:, 1])
    benchmark = data["y"] >= np.float32(0.7)
    criteria = {"benchmark_Q_ge_0p7": benchmark}
    for threshold in [0.8, 0.9, 0.95]:
        retained = (
            (near_gt >= 50) & (near_recall >= threshold) & (near_precision >= threshold)
        )
        tag = str(threshold).replace(".", "p")
        criteria["near_retained_" + tag] = retained
        criteria["near_retained_global_precision_" + tag] = retained & (
            precision >= threshold
        )
    aggregate = {}
    for name, mask in criteria.items():
        by_cluster = []
        for cluster in range(k):
            eligible = mask & (clusters == cluster)
            sites = len(set(data["site"][eligible]))
            by_cluster.append(
                dict(
                    cluster=cluster,
                    rows=int(eligible.sum()),
                    sites=sites,
                    eligible20rows2sites=bool(eligible.sum() >= 20 and sites >= 2),
                )
            )
        aggregate[name] = dict(
            rows=int(mask.sum()),
            eligible_groups=sum(row["eligible20rows2sites"] for row in by_cluster),
            by_cluster=by_cluster,
            low_Q_rows=int((mask & ~benchmark).sum()),
            by_hour={
                str(int(hour)): dict(
                    rows=int((data["hour"] == hour).sum()),
                    criterion_rows=int((mask & (data["hour"] == hour)).sum()),
                    low_Q_and_criterion=int(
                        (mask & ~benchmark & (data["hour"] == hour)).sum()
                    ),
                )
                for hour in np.unique(data["hour"])
            },
        )
    capacity = []
    for cluster in range(k):
        members = clusters == cluster
        sites = len(set(data["site"][members]))
        capacity.append(
            dict(
                cluster=cluster,
                rows=int(members.sum()),
                sites=sites,
                eligible_ignoring_normality=bool(members.sum() >= 20 and sites >= 2),
            )
        )
    result = dict(
        scope="Calibration-only support/sensitivity audit. Near-band precision/recall preservation does not certify operational normality or justify lowering environment IoU thresholds.",
        calibration_rows=len(data["ids"]),
        sites=len(set(data["site"])),
        clusters=k,
        required_rows_for_all_groups_at_min20=k * 20,
        minimum_total_with_all_rows_normal=320,
        minimum_normal_support_rule=dict(rows=20, sites=2),
        cluster_counts_before_normal_filter=capacity,
        max_groups_eligible_even_if_all_rows_normal=sum(
            v["eligible_ignoring_normality"] for v in capacity
        ),
        criteria=aggregate,
        near_gt_support_minimum=50,
        normal_policy_fitted=False,
        operational_normal_labels_available=False,
        input_source_sha256=sha(path),
        pack_sha256={
            key: sha(r / "packs" / (key + ".npz"))
            for key in ["release26_calibration", "release26_far_calibration"]
        },
        source_sha256=sha(Path(__file__)),
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output, result)
    print(
        "capacity",
        result["calibration_rows"],
        "for16x20=320; maximum eligible groups before normal filter",
        result["max_groups_eligible_even_if_all_rows_normal"],
    )
    print(
        {
            name: (v["rows"], v["eligible_groups"], v["low_Q_rows"])
            for name, v in aggregate.items()
        }
    )


if __name__ == "__main__":
    main()
