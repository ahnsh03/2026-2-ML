"""Derive an immutable remaining stationary plan from accepted manifests, never orphan files."""

from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from datetime import datetime, timezone


def remaining_plan(plan, accepted, *, max_frames=None):
    expected = {}
    for pose in plan["poses"]:
        conditions = pose.get(
            "conditions", [(w, h) for w in plan["weather_ids"] for h in plan["hours"]]
        )
        for weather, hour in conditions:
            sid = "release26_site%02d_w%d_h%02d" % (pose["id"], weather, hour)
            if sid in expected:
                raise ValueError("duplicate planned sample ID")
            expected[sid] = (pose, weather, hour)
    if max_frames is not None and max_frames < 1:
        raise ValueError("positive capture chunk size required")
    seen = set()
    for row in accepted:
        sid = row["sample_id"]
        if sid not in expected or sid in seen:
            raise ValueError("unknown or duplicate accepted sample")
        if row["simulator_release"] != plan["simulator_release"]:
            raise ValueError("capture release differs")
        if (row["weather"], row["hour"]) != expected[sid][1:]:
            raise ValueError("accepted condition differs from planned sample ID")
        if set(row["files"]) != {"Intensity", "Semantic", "Depth", "Instance"}:
            raise ValueError("incomplete accepted front channels")
        seen.add(sid)
    output = copy.deepcopy(plan)
    poses = []
    remaining = 0
    scheduled = 0
    for original in plan["poses"]:
        pose = copy.deepcopy(original)
        conditions = []
        for weather, hour in original.get(
            "conditions", [(w, h) for w in plan["weather_ids"] for h in plan["hours"]]
        ):
            if (
                "release26_site%02d_w%d_h%02d" % (original["id"], weather, hour)
            ) in seen:
                continue
            remaining += 1
            if max_frames is None or scheduled < max_frames:
                conditions.append([weather, hour])
                scheduled += 1
        if conditions:
            pose["conditions"] = conditions
            poses.append(pose)
    output.update(
        poses=poses,
        expected_frames=scheduled,
        already_accepted=len(seen),
        remaining_after_this_plan=remaining - scheduled,
    )
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--collections", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    rows = []
    receipts = {}
    for collection in a.collections:
        path = collection / "manifest.jsonl"
        raw = path.read_bytes()
        receipts[collection.name] = hashlib.sha256(raw).hexdigest()
        rows.extend(
            json.loads(line) for line in raw.decode().splitlines() if line.strip()
        )
    plan = remaining_plan(json.loads(a.plan.read_text()), rows, max_frames=a.max_frames)
    plan.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        accepted_manifest_sha256=receipts,
        resume_scope="Only accepted four-front-channel manifest rows are skipped. Orphans remain unregistered. This plan does not restart a stalled simulator RPC service.",
    )
    with a.output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "remaining scheduled",
        plan["expected_frames"],
        "pending afterward",
        plan["remaining_after_this_plan"],
    )


if __name__ == "__main__":
    main()
