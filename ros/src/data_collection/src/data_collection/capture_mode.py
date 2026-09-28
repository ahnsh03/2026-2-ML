# -*- coding: utf-8 -*-
# 이관: 인지 코드 레포 d0922bb `src/data_collection/src/data_collection/capture_mode.py` (원작성: 안승현)
"""Pure-Python helpers for MORAI Capture Mode collection.

This module deliberately has no ROS or gRPC import so its scheduling and
manifest behavior can be tested outside the ROS container.
"""

from __future__ import print_function

import json
import math
import os
import re
import time
from datetime import datetime, timezone


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def make_run_id(prefix="capture"):
    """Return a filename-safe run id with millisecond UTC precision."""
    now = datetime.now(timezone.utc)
    return "{}_{}_{:03d}".format(
        sanitize_name(prefix, fallback="capture", max_length=24),
        now.strftime("%Y%m%dT%H%M%S"),
        now.microsecond // 1000,
    )


def sanitize_name(value, fallback="capture", max_length=80):
    """Normalize user supplied values before sending them to MORAI as names."""
    value = _SAFE_NAME.sub("_", str(value or "").strip()).strip("_-")
    if not value:
        value = fallback
    return value[:max_length]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def default_capture_data_root():
    if os.environ.get("ML_DATA_ROOT"):
        return os.path.expanduser(os.environ["ML_DATA_ROOT"])
    return "/data"


class CaptureSchedule(object):
    """Wall-clock fixed-rate schedule that never emits catch-up bursts."""

    def __init__(self, hz, start_delay_sec=1.0, monotonic_fn=None):
        hz = float(hz)
        if not math.isfinite(hz) or hz <= 0.0 or hz > 10.0:
            raise ValueError("capture_hz must be in (0, 10]")
        self.hz = hz
        self.period_sec = 1.0 / hz
        self._monotonic = monotonic_fn or time.monotonic
        self.next_deadline = self._monotonic() + max(0.0, float(start_delay_sec))

    def remaining(self):
        return max(0.0, self.next_deadline - self._monotonic())

    def advance(self):
        """Advance once; skip missed slots instead of firing them back-to-back."""
        now = self._monotonic()
        self.next_deadline += self.period_sec
        if self.next_deadline <= now:
            self.next_deadline = now + self.period_sec
        return self.next_deadline


class CaptureRunWriter(object):
    """Append-only trigger manifest for one Capture Mode run."""

    def __init__(self, data_root, run_id, metadata):
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.run_id = sanitize_name(run_id, fallback="capture_run")
        self.root = os.path.join(self.data_root, "capture_runs", self.run_id)
        self.manifest_path = os.path.join(self.root, "capture_manifest.jsonl")
        self._manifest = None
        self.attempted = 0
        self.succeeded = 0
        self.failed = 0
        self.skipped_motion = 0
        self._successful_completed_ns = []
        self._rpc_latencies_ms = []

        os.makedirs(self.root, exist_ok=False)
        meta = dict(metadata or {})
        meta.update(
            {
                "schema_version": "capture-mode-0.2.0",
                "run_id": self.run_id,
                "created_at_utc": utc_now_iso(),
                "manifest": os.path.basename(self.manifest_path),
            }
        )
        self._write_json(os.path.join(self.root, "meta.json"), meta)
        self._manifest = open(self.manifest_path, "a", buffering=1)

    @staticmethod
    def _write_json(path, value):
        with open(path, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")

    def write_capture(self, record):
        rec = dict(record)
        rec.setdefault("recorded_at_utc", utc_now_iso())
        self._manifest.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self.attempted += 1
        if rec.get("success"):
            self.succeeded += 1
            completed_ns = rec.get("completed_wall_time_ns")
            if completed_ns is not None:
                self._successful_completed_ns.append(int(completed_ns))
        else:
            self.failed += 1
        latency = rec.get("rpc_latency_ms")
        if latency is not None:
            self._rpc_latencies_ms.append(float(latency))

    def mark_motion_skip(self):
        self.skipped_motion += 1

    def close(self):
        if self._manifest is None:
            return
        self._manifest.close()
        self._manifest = None
        effective_hz = None
        if len(self._successful_completed_ns) > 1:
            elapsed = (
                self._successful_completed_ns[-1]
                - self._successful_completed_ns[0]
            ) / 1000000000.0
            if elapsed > 0:
                effective_hz = (len(self._successful_completed_ns) - 1) / elapsed
        latency_summary = {}
        if self._rpc_latencies_ms:
            values = sorted(self._rpc_latencies_ms)
            latency_summary = {
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 3),
                "p95_ms": round(_percentile(values, 95), 3),
                "max_ms": round(max(values), 3),
            }
        self._write_json(
            os.path.join(self.root, "summary.json"),
            {
                "run_id": self.run_id,
                "closed_at_utc": utc_now_iso(),
                "attempted": self.attempted,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "skipped_motion": self.skipped_motion,
                "effective_capture_hz": (
                    None if effective_hz is None else round(effective_hz, 4)
                ),
                "rpc_latency": latency_summary,
            },
        )


def _percentile(sorted_values, percentile):
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def vector3_to_dict(value):
    return {
        "x": float(value.x),
        "y": float(value.y),
        "z": float(value.z),
    }


def ego_message_to_dict(msg):
    result = {
        "unique_id": int(getattr(msg, "unique_id", -1)),
        "position": vector3_to_dict(msg.position),
        "velocity": vector3_to_dict(msg.velocity),
        "heading_deg": float(msg.heading),
        "accel": float(getattr(msg, "accel", 0.0)),
        "brake": float(getattr(msg, "brake", 0.0)),
    }
    if hasattr(msg, "wheel_angle"):
        result["wheel_angle_deg"] = float(msg.wheel_angle)
    if hasattr(msg, "front_steer_angle"):
        result["front_steer_angle_deg"] = float(msg.front_steer_angle)
    header = getattr(msg, "header", None)
    if header is not None and getattr(header, "stamp", None) is not None:
        result["stamp_ns"] = int(header.stamp.secs) * 10**9 + int(header.stamp.nsecs)
    return result


def ctrl_message_to_dict(msg):
    result = {}
    for key in (
        "longlCmdType",
        "accel",
        "brake",
        "steering",
        "velocity",
        "acceleration",
    ):
        if hasattr(msg, key):
            value = getattr(msg, key)
            result[key] = int(value) if key == "longlCmdType" else float(value)
    return result


def speed_mps_from_ego(ego):
    velocity = (ego or {}).get("velocity") or {}
    x = float(velocity.get("x", 0.0))
    y = float(velocity.get("y", 0.0))
    z = float(velocity.get("z", 0.0))
    return math.sqrt(x * x + y * y + z * z)
