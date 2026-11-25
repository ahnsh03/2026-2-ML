"""Audit calibration support for fixed environment groups; does not fit thresholds."""

from pathlib import Path
import argparse, json, numpy as np
from study import load, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["packs", "environment-run", "rgb-index", "output"]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    data, _ = load(a.packs, a.keys)
    with np.load(a.environment_run / "per_frame.npz") as z:
        lookup = {sid: i for i, sid in enumerate(z["ids"])}
        indices = [lookup[sid] for sid in data["ids"]]
        cluster = z["cluster"][indices]
        index = json.loads(a.rgb_index.read_text())
        rows = {
            row["sample_id"]: row for value in index.values() for row in value["rows"]
        }
        hours = np.asarray([rows[str(sid)]["hour"] for sid in z["ids"]])
        coarse_pred = (z["hour"] < 6) | (z["hour"] >= 18)
        coarse_true = (hours < 6) | (hours >= 18)
        coarse_accuracy = float((coarse_pred == coarse_true).mean())
    groups = []
    for k in range(16):
        mask = cluster == k
        normal = mask & (data["y"] >= 0.7)
        groups.append(
            dict(
                cluster=k,
                frames=int(mask.sum()),
                sites=len(set(data["site"][mask])),
                benchmark_normal=int(normal.sum()),
                benchmark_normal_sites=len(set(data["site"][normal])),
                eligible_for_existing_group_rule=int(normal.sum()) >= 20
                and len(set(data["site"][normal])) >= 2,
            )
        )
    save(
        a.output,
        dict(
            calibration_frames=len(data["ids"]),
            cluster_support=groups,
            eligible_clusters=sum(
                row["eligible_for_existing_group_rule"] for row in groups
            ),
            coarse_clock_proxy_accuracy=coarse_accuracy,
            clock_proxy_definition="Hour<6 or>=18 versus other hours. This is a clock proxy, not visual visibility or operational normality.",
            scope="Support audit only; no environment-specific thresholds fitted. IoU>=.7 is a benchmark, not operational normality. Existing rule requires20benchmark-normal frames over2sites.",
            environment_run_sha256=sha(a.environment_run / "summary.json"),
            source_sha256=sha(Path(__file__)),
        ),
    )
    print(
        "eligible groups",
        sum(row["eligible_for_existing_group_rule"] for row in groups),
        "of16",
    )


if __name__ == "__main__":
    main()
