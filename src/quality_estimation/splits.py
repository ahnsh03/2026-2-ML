"""Explicit parent-group roles and GT-independent candidate thinning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path

from common.bridge import sha256_file
from .artifacts import write_json
from .dataset_io import load_dataset, require
from .perception_identity import identity_from_adaptation

ROLES = {"meta_train", "meta_dev", "meta_calibration", "candidate_pool", "final_test", "diagnostic_only"}


def _time(sample):
    value = sample.get("capture_requested_at_utc")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None
    except ValueError:
        return None


def _pose(sample):
    ego = sample.get("ego") or {}
    pos = ego.get("position") or {}
    if all(k in pos and math.isfinite(pos[k]) for k in ("x", "y")):
        return float(pos["x"]), float(pos["y"]), float(ego.get("heading_deg", 0))
    return None


def eligible_rows(samples, min_distance=5., min_time=1., min_heading=10.):
    """GT-free candidate thinning shared by training splits and operational selection."""
    require(min_distance >= 0 and min_time >= 0 and 0 <= min_heading <= 180, "invalid thinning settings")
    eligible = [False] * len(samples)
    reasons = ["temporally_or_spatially_redundant"] * len(samples)
    by_run = defaultdict(list)
    for i, r in enumerate(samples):
        by_run[r["run_id"]].append(i)
    kept_pixels = set()
    for run, indices in sorted(by_run.items()):
        indices.sort(key=lambda i: (samples[i].get("frame_id", i), samples[i]["sample_id"]))
        previous = None
        for i in indices:
            row = samples[i]
            pixel = row.get("pixel_sha256", row["input_sha256"]["rgb"])
            if pixel in kept_pixels:
                reasons[i] = "exact_duplicate_pixels"
                continue
            keep = previous is None
            if previous is not None:
                old = samples[previous]
                now_t, old_t = _time(row), _time(old)
                current, last = _pose(row), _pose(old)
                time_ok = now_t is None or old_t is None or now_t-old_t >= min_time
                if current is not None and last is not None:
                    distance = math.hypot(current[0]-last[0], current[1]-last[1])
                    heading = abs((current[2]-last[2]+180) % 360-180)
                    keep = time_ok and (distance >= min_distance or heading >= min_heading)
                else:
                    keep = time_ok
                # Different explicitly recorded conditions are not duplicate captures.
                if row.get("condition") != old.get("condition"):
                    keep = True
            if keep:
                eligible[i] = True
                reasons[i] = "first_or_sufficient_motion_or_condition_change"
                previous = i
                kept_pixels.add(pixel)
    return eligible, reasons


def build_split(data, group_roles, min_distance=5., min_time=1., min_heading=10., allow_exposed=False):
    require(set(group_roles.values()).issubset(ROLES), "invalid role")
    require(min_distance >= 0 and min_time >= 0 and 0 <= min_heading <= 180, "invalid thinning settings")
    parents = {r["parent_session_id"] for r in data.samples}
    require(set(group_roles) == parents, "group roles must exactly cover dataset parent groups")
    for row in data.samples:
        planned = row.get("preregistered_quality_role")
        assigned = group_roles[row["parent_session_id"]]
        require(assigned not in ("meta_calibration", "final_test") or row["source_role"] == "after_f0_training",
                assigned + " cannot use exposed or unknown-source data")
        if planned is not None:
            require(assigned == planned or (assigned == "diagnostic_only" and planned != "final_test"),
                    "assigned role differs from preregistered collection role")
    if not allow_exposed:
        require(all(r["source_role"] == "after_f0_training" or group_roles[r["parent_session_id"]] == "diagnostic_only"
                    for r in data.samples), "exposed/unknown data requires explicit diagnostic authorization")
    # Pixel hashes, when available, catch re-encoded copies too.
    by_pixel = defaultdict(list)
    for i, r in enumerate(data.samples):
        by_pixel[r.get("pixel_sha256", r["input_sha256"]["rgb"])].append(i)
    crossing = [indices for indices in by_pixel.values()
                if len({group_roles[data.samples[i]["parent_session_id"]] for i in indices}) > 1]
    require(not crossing, "exact source pixels cross roles; merge source groups before training")
    eligible, reasons = eligible_rows(data.samples, min_distance, min_time, min_heading)
    assignments = [{"sample_id": sid, "parent_session_id": row["parent_session_id"],
                    "role": group_roles[row["parent_session_id"]], "eligible": eligible[i], "reason": reasons[i]}
                   for i, (sid, row) in enumerate(zip(data.ids, data.samples))]
    return {"schema_version": "ml-quality-split-1", "dataset_metadata_sha256": sha256_file(data.path / "dataset.json"),
            "group_roles": group_roles, "assignment_unit": "parent_session_id",
            "thinning": {"minimum_displacement_m": min_distance, "minimum_elapsed_seconds": min_time,
                         "heading_change_degrees": min_heading, "uses_GT": False,
                         "missing_pose": "time spacing only; missing time retains source samples"},
            "exposed_data_authorized_for_diagnostic": allow_exposed, "exact_pixel_role_overlap": 0,
            "counts": dict(Counter(a["role"] for a in assignments)),
            "eligible_counts": dict(Counter(a["role"] for a in assignments if a["eligible"])),
            "assignments": assignments}


def read_split(path, data):
    split = json.loads(Path(path).read_text())
    require(split["dataset_metadata_sha256"] == sha256_file(data.path / "dataset.json"), "split/dataset mismatch")
    a = split["assignments"]
    require([r["sample_id"] for r in a] == data.ids, "split row order mismatch")
    parents = {row["parent_session_id"] for row in data.samples}
    require(set(split["group_roles"]) == parents, "split group map does not cover dataset parents")
    roles = defaultdict(set)
    pixels = defaultdict(set)
    for assignment, sample in zip(a, data.samples):
        require(assignment["parent_session_id"] == sample["parent_session_id"], "parent group changed")
        require(assignment["role"] in ROLES and isinstance(assignment["eligible"], bool), "invalid split assignment")
        require(assignment["role"] == split["group_roles"][sample["parent_session_id"]], "assignment differs from group map")
        planned = sample.get("preregistered_quality_role")
        require(assignment["role"] not in ("meta_calibration", "final_test") or sample["source_role"] == "after_f0_training",
                assignment["role"] + " cannot use exposed or unknown-source data")
        if planned is not None:
            require(assignment["role"] == planned or (assignment["role"] == "diagnostic_only" and planned != "final_test"),
                    "split differs from preregistered collection role")
        if not split.get("exposed_data_authorized_for_diagnostic", False):
            require(sample["source_role"] == "after_f0_training" or assignment["role"] == "diagnostic_only",
                    "exposed/unknown source in an evaluation split")
        pixel = sample.get("pixel_sha256", sample["input_sha256"]["rgb"])
        pixels[pixel].add(assignment["role"])
        roles[sample["parent_session_id"]].add(assignment["role"])
    require(all(len(v) == 1 for v in roles.values()), "parent group crosses roles")
    require(all(len(v) == 1 for v in pixels.values()), "source pixels cross roles")
    return split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--group-roles", required=True, help="JSON parent group -> role mapping")
    p.add_argument("--output", required=True)
    p.add_argument("--min-distance", type=float, default=5.)
    p.add_argument("--min-time", type=float, default=1.)
    p.add_argument("--min-heading", type=float, default=10.)
    p.add_argument("--allow-exposed-development", action="store_true")
    p.add_argument("--adaptation-run", type=Path)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    identity = identity_from_adaptation(args.adaptation_run) if args.adaptation_run else None
    data = load_dataset(args.dataset_dir, perception_identity=identity)
    result = build_split(data, json.loads(Path(args.group_roles).read_text()), args.min_distance,
                         args.min_time, args.min_heading, args.allow_exposed_development)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != "assignments"}, indent=2))


if __name__ == "__main__":
    main()
