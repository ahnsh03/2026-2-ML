"""Choose new release-transfer road sites using map geometry alone, without RGB/GT."""

from __future__ import annotations
import argparse, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, distance
from study import read, save, sha
from register_geometry import clearance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("packs", "index", "map", "template", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument(
        "--exclude-plan",
        type=Path,
        action="append",
        default=[],
        help="Exclude already captured/planned links and nearby poses",
    )
    ap.add_argument("--site-id-start", type=int, default=200)
    ap.add_argument(
        "--training-selection",
        type=Path,
        help="Restrict the clearance reference to frozen quality-training membership",
    )
    a = ap.parse_args()
    exclusions = [pose for path in a.exclude_plan for pose in read(path)["poses"]]
    excluded_links = {p["requested_link"] for p in exclusions if "requested_link" in p}
    excluded_xy = np.asarray(
        [[p["pose"]["location"][axis] for axis in "xy"] for p in exclusions]
    )
    points = []
    membership = (
        set(read(a.training_selection)["ids"]) if a.training_selection else None
    )
    for key in read(a.index):
        with np.load(a.packs / (key + ".npz")) as z:
            indices = (
                np.isin(z["ids"], list(membership))
                if membership is not None
                else np.ones(len(z["ids"]), bool)
            )
            points.extend(z["xy"][indices])
    tree = cKDTree(np.unique(points, axis=0))
    candidates = []
    for record in read(a.map):
        if record["idx"] in excluded_links:
            continue
        if (
            record.get("lazy_init")
            or record.get("opp_traffic")
            or not record.get("lane_mark_left")
            or not record.get("lane_mark_right")
        ):
            continue
        vertices = np.asarray(record["points"])
        if len(vertices) < 35 or tree.query(vertices[:, :2])[0].min() < 25:
            continue
        link_distance = clearance(vertices[:, :2], tree)
        if link_distance < 25:
            continue
        stations = np.r_[
            0, np.cumsum(np.linalg.norm(np.diff(vertices[:, :2], axis=0), axis=1))
        ]
        for i in range(4, len(vertices) - 4, 12):
            if (
                len(excluded_xy)
                and np.linalg.norm(excluded_xy - vertices[i, :2], axis=1).min() < 50
            ):
                continue
            if stations[i] < 3 or stations[-1] - stations[i] < 22:
                continue
            pts = np.column_stack(
                [
                    np.interp(
                        stations[i] + np.array([0, 10, 20]), stations, vertices[:, axis]
                    )
                    for axis in range(3)
                ]
            )
            delta = np.diff(pts, axis=0)
            span = np.linalg.norm(delta[:, :2], axis=1)
            if min(span) < 7 or np.max(abs(delta[:, 2]) / span) > 0.065:
                continue
            headings = np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0]))
            curve = abs(float((headings[1] - headings[0] + 180) % 360 - 180))
            prior = float(tree.query(vertices[i, :2])[0])
            direction = vertices[i + 2] - vertices[i - 2]
            if prior < 35 or np.linalg.norm(direction[:2]) < 0.5:
                continue
            candidates.append(
                dict(
                    link=record["idx"],
                    index=i,
                    station=float(stations[i]),
                    xyz=vertices[i].tolist(),
                    yaw=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
                    prior_m=prior,
                    link_distance=link_distance,
                    curve_deg=curve,
                )
            )
    chosen = []
    used = set()
    for i in range(a.sites):
        eligible = [
            r
            for r in candidates
            if r["link"] not in used
            and all(
                np.linalg.norm(np.array(r["xyz"][:2]) - q["xyz"][:2]) >= 35
                for q in chosen
            )
        ]
        if not eligible:
            break
        subset = [
            r for r in eligible if (r["curve_deg"] >= 6) == bool(i % 2)
        ] or eligible
        item = max(subset, key=lambda r: (r["prior_m"], r["link"]))
        chosen.append(item)
        used.add(item["link"])
    if len(chosen) != a.sites:
        raise ValueError(
            f"only {len(chosen)} eligible distinct links; do not relax the predeclared rule silently"
        )
    plan = read(a.template)
    plan.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        role="Prospective 26R1 new-road test, geometry-only selection after model freeze; no RGB, GT or candidate prediction inspected.",
        source_map_sha256=sha(a.map),
        prior_index_sha256=sha(a.index),
        prior_rows=len(points),
        clearance_reference="frozen quality-training membership only"
        if membership is not None
        else "all accessible archived camera positions",
        training_selection_sha256=sha(a.training_selection)
        if a.training_selection
        else None,
        expected_frames=24 * len(chosen),
        exclusion_plan_sha256=[sha(path) for path in a.exclude_plan],
        poses=[],
    )
    for i, r in enumerate(chosen):
        plan["poses"].append(
            dict(
                id=a.site_id_start + i,
                group=f"release26_newroad{i}",
                requested_link=r["link"],
                station=r["station"],
                minimum_prior_m=r["prior_m"],
                link_distance_m=r["link_distance"],
                curve_deg=r["curve_deg"],
                pose=dict(
                    location=dict(zip("xyz", [*r["xyz"][:2], r["xyz"][2] + 0.35])),
                    rotation=dict(x=0, y=0, z=r["yaw"]),
                ),
            )
        )
    save(a.output, plan)
    print(
        "new roads",
        len(chosen),
        "minimum prior",
        min(r["prior_m"] for r in chosen),
        "minimum link clearance",
        min(r["link_distance"] for r in chosen),
    )


if __name__ == "__main__":
    main()
