"""Register external static links against all accessible prior camera positions, before scoring."""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from study import read, save, sha


def clearance(vertices, tree):
    best = float(tree.query(vertices)[0].min())
    for a, b in zip(vertices[:-1], vertices[1:]):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if best == 0 or length == 0:
            continue
        candidates = tree.query_ball_point((a + b) / 2, best + length / 2 + 1e-8)
        if candidates:
            offset = tree.data[candidates] - a
            t = np.clip(offset.dot(delta) / (length * length), 0, 1)
            best = min(
                best, float(np.linalg.norm(offset - t[:, None] * delta, axis=1).min())
            )
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("packs", "index", "map", "plans", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    index = read(a.index)
    static = ["extra_cal", "newroads", "shapes"]
    external = static + [k for k in index if k.startswith("archive_")]
    prior = [k for k in index if k not in external]
    points = []
    seenids = set()
    seenpixels = set()
    for key in prior:
        with np.load(a.packs / (key + ".npz")) as z:
            points.extend(z["xy"])
            seenids.update(z["ids"])
            seenpixels.update(z["pixel_sha256"])
    tree = cKDTree(np.unique(points, axis=0))
    links = {v["idx"]: np.asarray(v["points"])[:, :2] for v in read(a.map)}
    rows = []
    cache = {}
    for key in external:
        plan = (
            read(
                a.plans
                / (
                    "road_shape_transfer.json"
                    if key == "shapes"
                    else "real_g1_extension.json"
                )
            )
            if key in static
            else None
        )
        if plan:
            assert sha(a.map) == plan["source_link_sha256"]
            locations = {r["id"]: r for r in plan["poses"]}
        with np.load(a.packs / (key + ".npz")) as z:
            for i, sid in enumerate(z["ids"]):
                assert sid not in seenids and z["pixel_sha256"][i] not in seenpixels
                seenids.add(sid)
                seenpixels.add(z["pixel_sha256"][i])
                xy = z["xy"][i]
                row = dict(
                    id=str(sid),
                    key=key,
                    point_distance=float(tree.query(xy)[0]),
                    link="",
                    link_distance=None,
                )
                if plan:
                    p = locations[int(re.search("_p(\\d+)_", str(sid))[1])]
                    link = p["requested_link"]
                    requested = np.array([p["pose"]["location"][axis] for axis in "xy"])
                    assert np.linalg.norm(xy - requested) < 0.1
                    assert (
                        np.linalg.norm(links[link][p["waypoint_index"]] - requested)
                        < 0.1
                    )
                    if link not in cache:
                        cache[link] = clearance(links[link], tree)
                    row.update(link=link, link_distance=cache[link])
                rows.append(row)
    save(
        a.output,
        dict(
            prior_keys=prior,
            prior_rows=len(points),
            prior_unique_xy=len(tree.data),
            map_sha256=sha(a.map),
            index_sha256=sha(a.index),
            source_sha256=sha(Path(__file__)),
            rows=rows,
        ),
    )
    isolated = [
        r
        for r in rows
        if r["link_distance"] is not None
        and r["link_distance"] >= 25
        and r["point_distance"] >= 25
    ]
    print(
        "external",
        len(rows),
        "isolated static",
        len(isolated),
        "links",
        len(set(r["link"] for r in isolated)),
    )


if __name__ == "__main__":
    main()
