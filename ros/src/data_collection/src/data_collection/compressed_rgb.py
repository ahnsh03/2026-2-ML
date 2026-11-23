"""Store one ROS compressed color frame unchanged and emit a GT-free inference manifest."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .session_archive import session_context, write_receipt


def _json_line(stream, record):
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def _encoding(payload, declaration):
    if payload.startswith(b"\xff\xd8\xff"):
        format_name, suffix = "jpeg", ".jpg"
    elif payload.startswith(b"\x89PNG\r\n\x1a\n"):
        format_name, suffix = "png", ".png"
    else:
        raise ValueError("expected JPEG or PNG image data")
    declared = declaration.lower().replace("jpg", "jpeg")
    if "depth" in declared or any(name in declared for name in ("jpeg", "png") if name != format_name):
        raise ValueError("compressed image encoding differs from the payload")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("expected a decodable 8-bit color camera frame")
    return suffix, image


class CompressedRGBSession:
    """A new session per publisher run. The caller serializes appends and closes on shutdown."""
    def __init__(self, directory, topic, context=None):
        self.directory = Path(directory)
        self.topic = topic
        self.context = session_context(context)
        self.rows = 0
        self.sequence_gaps = 0
        self.repeated_sequences = 0
        self.sequence_resets = 0
        self.previous_seq = None
        self.started = datetime.now(timezone.utc).isoformat()
        self.files = ExitStack()
        self.manifest = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            (self.directory / "rgb").mkdir()
            self.manifest = self.files.enter_context((self.directory / "rgb.jsonl").open("x"))
            self.transport = self.files.enter_context((self.directory / "transport.jsonl").open("x"))
            self._status("running")
        except BaseException:
            self.files.close()
            raise
        return self

    def _status(self, state, error=None, **extra):
        record = dict(status=state, topic=self.topic, context=self.context, started_utc=self.started, rows=self.rows, **extra,
                      sequence_gaps=self.sequence_gaps, repeated_sequences=self.repeated_sequences,
                      sequence_resets=self.sequence_resets, updated_utc=datetime.now(timezone.utc).isoformat())
        if error is not None:
            record["error"] = str(error)
        tmp = self.directory / "run.pending"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(self.directory / "run.json")

    def append(self, payload, declaration, sequence, stamp_ns, frame_id, stream=None):
        received = datetime.now(timezone.utc).isoformat()
        suffix, image = _encoding(payload, declaration)
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        pixel_digest = hashlib.sha256(np.asarray(rgb.shape, dtype=np.int64).tobytes())
        pixel_digest.update(rgb)
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"{self.rows:08d}{suffix}"
        dest = self.directory / "rgb" / filename
        with dest.with_suffix(".pending").open("xb") as pending:
            pending.write(payload)
        dest.with_suffix(".pending").replace(dest)
        record = dict(sample_id=f"{self.directory.name}_{self.rows:08d}", view="front",
                      image="rgb/" + filename, expected_sha256={"rgb": digest})
        if self.previous_seq is not None:
            step = (sequence - self.previous_seq) % (2**32)
            if step == 0:
                self.repeated_sequences += 1
            elif step >= 2**31:
                self.sequence_resets += 1
            else:
                self.sequence_gaps += step - 1
        self.previous_seq = sequence
        _json_line(self.transport, dict(sample_id=record["sample_id"], ros_header_seq=sequence,
                   image=record["image"], pixel_sha256=pixel_digest.hexdigest(),
                   ros_stamp_ns=stamp_ns, ros_frame_id=frame_id, received_utc=received, format=declaration,
                   width=width, height=height, bytes=len(payload), sha256=digest))
        _json_line(self.manifest, record)
        self.rows += 1
        if stream is not None:
            _json_line(stream, record)
        return record

    def __exit__(self, kind, error, trace):
        self.files.close()
        if error is not None or not self.rows:
            self._status("failed", error if error is not None else "no RGB frames")
            if error is None:
                raise ValueError("no RGB frames")
            return
        seal = write_receipt(self.directory, session_id=self.directory.name, source="ros_rgb", context=self.context,
                             rows="transport.jsonl", frames=self.rows, channels=["front/rgb"],
                             manifests=["transport.jsonl", "rgb.jsonl"], retention="all_received_rgb",
                             clock="ros_header_ns_and_host_receive_utc")
        self._status("complete", session_sha256=seal)
        digest = hashlib.sha256((self.directory / "run.json").read_bytes()).hexdigest()
        (self.directory / "_SUCCESS").write_text(digest + "\n")
