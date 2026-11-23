#!/usr/bin/env python3
"""Register existing front captures, list sealed sessions, verify payloads or export offline inputs."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros/src/data_collection/src"))
sys.path.insert(0, str(REPO / "src"))
from data_collection.capture_sync import CaptureSyncError, png_hw
from data_collection.session_archive import (
    RECEIPT, completed_receipt, file_sha, json_rows, relative_file, session_context, verified_frames, write_receipt,
)

SUFFIXES = {"Intensity": "rgb", "Semantic": "semantic", "Instance": "instance", "Depth": "depth"}


@contextmanager
def new_directory(target):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".capture-stage-", dir=target.parent) as tmp:
        stage = Path(tmp) / "session"
        stage.mkdir()
        yield stage
        stage.rename(target)


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def front_files(row, camera_dir):
    name = row["name"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("expected a capture name without directories")
    expected = {str(Path(camera_dir) / (name + "_" + suffix + ".png")): "front/" + kind
                for suffix, kind in SUFFIXES.items()}
    if "files" in row:
        files = {}
        for item in row["files"]:
            channel = expected.get(item["path"])
            if channel:
                if channel in files:
                    raise ValueError("duplicate front channel for " + name)
                files[channel] = {"path": item["path"], "sha256": item["sha256"]}
        if len(files) != len(expected):
            raise ValueError("incomplete front capture: " + name)
        return files
    return {"front/" + kind: {"path": str(Path(camera_dir) / (name + "_" + suffix + ".png")),
                              "sha256": row["sha256"][suffix]} for suffix, kind in SUFFIXES.items()}


def register_front(args):
    if not args.session_id.strip():
        raise ValueError("empty session ID")
    context = session_context(json.loads(args.context.read_text()) if args.context else None)
    with new_directory(args.output) as stage:
        snapshot = stage / "source.jsonl"
        shutil.copyfile(args.manifest, snapshot)
        ids, capture_names = set(), set()
        with (stage / "media.jsonl").open("x", encoding="utf-8") as out:
            for row in json_rows(snapshot):
                if row.get("status", "complete") != "complete" or row.get("capture", {}).get("success") is False:
                    raise ValueError("incomplete input row; use the admitted capture manifest")
                sid = row.get("sample_id") or row.get("id") or row["name"]
                if not isinstance(sid, str) or not sid or sid in ids:
                    raise ValueError("missing or duplicate sample ID: " + str(sid))
                ids.add(sid)
                if row["name"] in capture_names:
                    raise ValueError("duplicate capture name: " + row["name"])
                capture_names.add(row["name"])
                files = front_files(row, args.camera_dir)
                dimensions = set()
                for item in files.values():
                    path = relative_file(args.media_root, item["path"])
                    dimensions.add(png_hw(path))
                    if file_sha(path) != item["sha256"]:
                        raise ValueError("source digest mismatch: " + item["path"])
                if len(dimensions) != 1 or min(next(iter(dimensions))) <= 0:
                    raise ValueError("incompatible front channel dimensions: " + sid)
                role = row.get("preregistered_quality_role") or row.get("role")
                if role == "meta_train" and row.get("role") not in (None, "meta_train", "train", "training"):
                    role = row["role"]
                write_row(out, dict(sample_id=sid, files=files, source_role=role,
                                    parent_session_id=row.get("parent_session_id"),
                                    pose_id=row.get("pose_id"), source_row=len(ids)-1))
        if not ids:
            raise ValueError("empty capture manifest")
        seal = write_receipt(stage, session_id=args.session_id, source="native_front", context=context,
                             rows="media.jsonl", frames=len(ids), channels=files,
                             manifests=["media.jsonl", "source.jsonl"], retention=args.population,
                             clock="see_original_source_row", media_root="external")
        (stage / "_SUCCESS").write_text(json.dumps({"session_sha256": seal}) + "\n")
    return {"session_id": args.session_id, "frames": len(ids), "registered_files": 4 * len(ids)}


def export_front(args):
    from quality_estimation.image_io import color_channels, decoded_pixel_sha256, read_image
    import cv2

    receipt = completed_receipt(args.session)
    root = args.media_root or args.session
    context = receipt["context"]
    count = paired_count = 0
    with new_directory(args.output) as stage:
        with (stage / "rgb.jsonl").open("x") as images, (stage / "paired.jsonl").open("x") as pairs:
            for sid, files, row in verified_frames(args.session, args.media_root):
                rgb = files["front/rgb"]
                data, digest = read_image(root, rgb["path"], cv2.IMREAD_UNCHANGED)
                if digest != rgb["sha256"]:
                    raise ValueError("RGB changed during export: " + sid)
                pixel = decoded_pixel_sha256(cv2.cvtColor(color_channels(data, "RGB"), cv2.COLOR_BGR2RGB))
                if row.get("pixel_sha256", pixel) != pixel:
                    raise ValueError("decoded RGB digest mismatch: " + sid)
                image = dict(sample_id=sid, view="front", image=rgb["path"], expected_sha256={"rgb": digest})
                write_row(images, image)
                count += 1
                if "front/semantic" in files:
                    sem = files["front/semantic"]
                    role = row.get("source_role") or context.get("split_role") or "unassigned"
                    # Both the archived role and any new session restriction must permit training.
                    if context.get("split_role") not in (None, "meta_train") and role == "meta_train":
                        role = context["split_role"]
                    paired = dict(image, semantic=sem["path"], pixel_sha256=pixel,
                                  expected_sha256={"rgb": digest, "semantic": sem["sha256"]},
                                  preregistered_quality_role=role)
                    group = context.get("parent_session_id") or row.get("parent_session_id")
                    if group:
                        paired["split_group"] = group
                    write_row(pairs, paired)
                    paired_count += 1
        if completed_receipt(args.session) != receipt:
            raise ValueError("session changed during export")
        info = dict(source_receipt_sha256=receipt["receipt_sha256"], rgb=count, paired=paired_count,
                    manifest_sha256={n: file_sha(stage / n) for n in ("rgb.jsonl", "paired.jsonl")})
        (stage / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        (stage / "_SUCCESS").write_text(file_sha(stage / "export.json") + "\n")
    return info


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    reg = commands.add_parser("register-front", help="Index existing four-channel capture rows without copying images")
    reg.add_argument("--manifest", required=True, type=Path)
    reg.add_argument("--media-root", required=True, type=Path)
    reg.add_argument("--camera-dir", default="captures/CAMERA_1", help="Front camera directory relative to media root")
    reg.add_argument("--session-id", required=True)
    reg.add_argument("--population", choices=("unknown", "all_captured", "selected_only"), default="unknown",
                     help="Original frame population as established by the capture protocol")
    reg.add_argument("--context", type=Path)
    reg.add_argument("--output", required=True, type=Path)
    listing = commands.add_parser("list", help="Print one JSON row per receipt, with its root-relative location")
    listing.add_argument("root", type=Path)
    for name in ("verify", "export"):
        cmd = commands.add_parser(name)
        cmd.add_argument("session", type=Path)
        cmd.add_argument("--media-root", type=Path)
        if name == "export":
            cmd.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "register-front":
        result = register_front(args)
    elif args.command == "list":
        if not args.root.is_dir():
            cli.error("catalog root does not exist")
        for folder, sub, names in os.walk(args.root):
            sub[:] = sorted(name for name in sub if not name.startswith(".capture-stage-"))
            if RECEIPT in names:
                directory = Path(folder)
                try:
                    record = dict(completed_receipt(directory), status="complete")
                except (OSError, KeyError, ValueError) as exc:
                    record = dict(status="unsealed_or_invalid", error=str(exc))
                write_row(sys.stdout, dict(record, directory=str(directory.relative_to(args.root))))
                sub.clear()
            elif ".sync.lock" in names or ("rgb.jsonl" in names and "run.json" in names):
                directory = Path(folder)
                write_row(sys.stdout, dict(status="unregistered", directory=str(directory.relative_to(args.root))))
                sub.clear()
        return
    elif args.command == "export":
        result = export_front(args)
    else:
        frames = files = 0
        for _, media, _ in verified_frames(args.session, args.media_root):
            frames += 1
            files += len(media)
        result = dict(verified_frames=frames, verified_files=files)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, CaptureSyncError) as exc:
        sys.exit("capture session: " + str(exc))
