"""Register accepted stationary capture parts, preserving hashes and release provenance."""

from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from study import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--collections", nargs="+", type=Path, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--expected-rows", type=int, required=True)
    ap.add_argument("--existing-index", type=Path)
    ap.add_argument(
        "--sample-id-prefix",
        default="",
        help="Collection namespace for logical sample/site IDs; raw capture IDs are preserved",
    )
    a = ap.parse_args()
    if (
        a.sample_id_prefix
        and re.fullmatch(r"[A-Za-z0-9_.-]+", a.sample_id_prefix) is None
    ):
        raise ValueError("use a path-independent sample ID prefix")
    index = read(a.existing_index) if a.existing_index else {}
    reserved = [row["sample_id"] for value in index.values() for row in value["rows"]]
    if len(set(reserved)) != len(reserved):
        raise ValueError("existing registry contains duplicate sample IDs")
    reserved = set(reserved)
    rows = []
    seen = set()
    receipts = {}
    for collection in a.collections:
        manifest = collection / "manifest.jsonl"
        receipts[str(collection.relative_to(a.data_root))] = sha(manifest)
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            sid = a.sample_id_prefix + r["sample_id"]
            if sid in reserved:
                raise ValueError(
                    "captured sample ID already registered; use a unique collection prefix: "
                    + sid
                )
            if sid in seen:
                raise ValueError("duplicate captured sample ID")
            seen.add(sid)
            if set(r["files"]) != {"Intensity", "Semantic", "Depth", "Instance"}:
                raise ValueError(
                    "four raw capture channels required before registration"
                )
            for kind, f in r["files"].items():
                if sha(collection / f["path"]) != f["sha256"]:
                    raise ValueError("captured source checksum differs")
            before = r["before"]
            after = r["after"]
            if before["weather"] != after["weather"] or before["time"] != after["time"]:
                raise ValueError("condition changed within capture")
            for state in (before, after):
                if state["ego"]["vehicle_state"]["gear_mode"] != "GEAR_MODE_P":
                    raise ValueError("accepted frame was not Park")
            b = before["ego"]["transform"]["location"]
            c = after["ego"]["transform"]["location"]
            if math.dist([b[k] for k in "xyz"], [c[k] for k in "xyz"]) >= 0.005:
                raise ValueError("capture position changed")

            def relative(kind):
                return str(
                    (collection / r["files"][kind]["path"]).relative_to(a.data_root)
                )

            rows.append(
                dict(
                    sample_id=sid,
                    capture_sample_id=r["sample_id"],
                    image=relative("Intensity"),
                    semantic=relative("Semantic"),
                    expected_sha256=dict(
                        rgb=r["files"]["Intensity"]["sha256"],
                        semantic=r["files"]["Semantic"]["sha256"],
                    ),
                    view="front",
                    group=a.key,
                    site=a.sample_id_prefix + r["site"],
                    capture_site=r["site"],
                    cohort=a.key,
                    weather=r["weather"],
                    hour=r["hour"],
                    xy=[b["x"], b["y"]],
                    capture_completion_basis=r["completion_basis"],
                    simulator_release=r["simulator_release"],
                )
            )
    if len(rows) != a.expected_rows:
        raise ValueError(f"expected {a.expected_rows}, accepted {len(rows)}")
    rows.sort(key=lambda r: r["sample_id"])
    if a.key in index:
        raise ValueError("index key already exists")
    index[a.key] = dict(simulator_release="26.R1", rows=rows, source_manifests=receipts)
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output, index)
    print(a.key, len(rows), sha(a.output))


if __name__ == "__main__":
    main()
