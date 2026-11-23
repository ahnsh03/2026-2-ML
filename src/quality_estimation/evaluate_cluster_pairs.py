"""Compare frozen cluster assignments using pose-matched weather and hour pairs."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path


def evaluate(rows, method):
    """Count paired relations and their expectation if assignments shuffle within each pose."""
    if not rows:
        raise ValueError("no pose observations")
    clusters = defaultdict(Counter)
    for row in rows:
        if any(row.get(key) is None or not str(row[key]).strip() for key in ("pose", "weather", "hour", method)):
            raise ValueError("missing pose, weather, hour or assignment")
        clusters[row["pose"]][row[method]] += 1

    sample_counts = {pose: sum(hist.values()) for pose, hist in clusters.items()}
    names = ("same_weather_hour_different_pose", "only_weather_changed_same_pose", "only_hour_changed_same_pose")
    results = {name: dict(pairs=0, observed=0, shuffled_expected=0.0) for name in names}
    same_prob = {}
    for left, right in combinations(rows, 2):
        pose_a, pose_b = left["pose"], right["pose"]
        same_pose = pose_a == pose_b
        same_weather = left["weather"] == right["weather"]
        same_hour = left["hour"] == right["hour"]
        if not same_pose and same_weather and same_hour:
            name = names[0]
        elif same_pose and not same_weather and same_hour:
            name = names[1]
        elif same_pose and same_weather and not same_hour:
            name = names[2]
        else:
            continue
        pair = frozenset((pose_a, pose_b))
        if pair not in same_prob:
            a, b = clusters[pose_a], clusters[pose_b]
            na, nb = sample_counts[pose_a], sample_counts[pose_b]
            if same_pose:
                same_prob[pair] = sum(n * (n - 1) for n in a.values()) / (na * (na - 1))
            else:
                same_prob[pair] = sum(n * b[c] for c, n in a.items()) / (na * nb)
        out = results[name]
        out["pairs"] += 1
        same_center = left[method] == right[method]
        out["observed"] += int(not same_center if same_pose else same_center)
        out["shuffled_expected"] += 1 - same_prob[pair] if same_pose else same_prob[pair]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Fixed-pose observations with pose, weather, hour and cluster columns")
    parser.add_argument("--methods", nargs="+", default=["emeans", "kmeans"])
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    print(json.dumps({method: evaluate(rows, method) for method in args.methods}, indent=2))


if __name__ == "__main__":
    main()
