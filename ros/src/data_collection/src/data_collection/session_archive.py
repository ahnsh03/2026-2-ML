"""Portable capture receipts. Paths in a receipt are relative to the session or its media root."""
import hashlib
import json
import os
from pathlib import Path


RECEIPT = "capture_session.json"
SCHEMA = "collection-session-v1"
CONTEXT_FIELDS = {
    "purpose", "parent_session_id", "road_id", "site_id", "split_role", "camera_profile",
    "condition_source", "sampling_policy", "root_alias", "checkpoint_version", "roi_version", "gt_policy_version", "review_policy_version",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_context(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - CONTEXT_FIELDS:
        raise ValueError("invalid collection session context fields")
    if any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError("context values must be nonempty strings")
    return dict(value)


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("expected a root-relative file path")
    path = (Path(root) / name).resolve()
    path.relative_to(Path(root).resolve())
    return path


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_receipt(directory, *, session_id, source, context, rows, frames, channels,
                  manifests, retention, clock, media_root="session"):
    directory = Path(directory)
    record = dict(schema=SCHEMA, session_id=session_id, source=source, context=session_context(context),
                  frame_manifest=rows, frame_count=frames, channels=sorted(channels),
                  retention=retention, clock=clock, media_root=media_root,
                  manifest_sha256={name: file_sha(relative_file(directory, name)) for name in manifests})
    path = directory / RECEIPT
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return file_sha(path)


def completed_receipt(directory):
    directory = Path(directory)
    receipt = directory / RECEIPT
    payload = receipt.read_bytes()
    record = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if record["frame_manifest"] not in record["manifest_sha256"]:
        raise ValueError("unregistered frame manifest")
    if record["schema"] != SCHEMA:
        raise ValueError("unknown session schema")
    marker = (directory / "_SUCCESS").read_text().strip()
    if record["source"] == "ros_rgb":
        run = directory / "run.json"
        if marker != file_sha(run):
            raise ValueError("RGB completion digest mismatch")
        sealed = json.loads(run.read_text())
    else:
        sealed = json.loads(marker)
    if sealed["session_sha256"] != digest:
        raise ValueError("session receipt digest mismatch")
    return dict(record, receipt_sha256=digest)


def frame_media(row, source):
    """Return the front identity and every registered relative payload and byte digest."""
    if source == "capture_sync":
        media = {}
        for modality, views in row["paths"].items():
            for view, path in views.items():
                channel = view + "/" + ("rgb" if modality == "intensity" else modality)
                if modality == "state":
                    channel = "state/" + view
                media[channel] = {"path": path, "sha256": row["file_sha256"][path]}
        return "{}/front/{:06d}".format(row["run_id"], row["frame_id"]), media
    if source == "ros_rgb":
        return row["sample_id"], {"front/rgb": {"path": row["image"], "sha256": row["sha256"]}}
    if source == "native_front":
        return row["sample_id"], row["files"]
    raise ValueError("unknown collection source: " + source)


def verified_frames(directory, media_root=None):
    """Check completion, all registered manifests/media and identities before consumer closeout."""
    directory = Path(directory)
    record = completed_receipt(directory)
    if record["media_root"] == "external" and media_root is None:
        raise ValueError("this collection requires its external media root")
    root = Path(media_root) if media_root is not None else directory
    for name, expected in record["manifest_sha256"].items():
        if file_sha(relative_file(directory, name)) != expected:
            raise ValueError("manifest digest mismatch: " + name)
    ids = set()
    for row in json_rows(relative_file(directory, record["frame_manifest"])):
        sid, files = frame_media(row, record["source"])
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("missing or duplicate sample ID: " + str(sid))
        if sorted(files) != record["channels"]:
            raise ValueError("channel set mismatch for " + sid)
        ids.add(sid)
        for item in files.values():
            if file_sha(relative_file(root, item["path"])) != item["sha256"]:
                raise ValueError("media digest mismatch: " + item["path"])
        yield sid, files, row
    if not ids or len(ids) != record["frame_count"]:
        raise ValueError("frame count mismatch")
    if completed_receipt(directory) != record:
        raise ValueError("session changed during verification")
