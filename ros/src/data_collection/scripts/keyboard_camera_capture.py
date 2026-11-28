#!/usr/bin/env python3
"""Save one original ROS CompressedImage per Space/S key in a focused terminal."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import select
import sys
import termios
import threading
import time
import tty
import uuid

import cv2
import numpy as np


class CameraCapture:
    def __init__(self, output, condition, topic, max_age=2.0):
        if not re.fullmatch(r"[\w-]+", condition):
            raise ValueError("condition must contain only letters, numbers, _ or -")
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max-age must be a finite positive number")
        self.directory = Path(output).expanduser().resolve() / condition
        self.directory.mkdir(parents=True, exist_ok=True)
        self.condition, self.topic, self.max_age = condition, topic, max_age
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.saved_sequence = None

    def receive(self, message):
        with self.lock:
            self.sequence += 1
            self.latest = (message, time.monotonic(), self.sequence)

    def save(self):
        with self.lock:
            snapshot = self.latest
        if snapshot is None:
            raise ValueError("No image received yet. Check topic and ROS connection.")
        message, received, sequence = snapshot
        age = time.monotonic() - received
        if age > self.max_age:
            raise ValueError("Latest image is {:.1f}s old. Check camera stream.".format(age))
        if sequence == self.saved_sequence:
            raise ValueError("This received frame was already saved; wait for a new frame.")
        data = bytes(message.data)
        if data.startswith(b"\xff\xd8\xff"):
            extension = ".jpg"
        elif data.startswith(b"\x89PNG\r\n\x1a\n"):
            extension = ".png"
        else:
            raise ValueError("Camera payload is not a supported JPEG/PNG image.")
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Camera image could not be decoded; nothing saved.")
        now = datetime.now(timezone.utc)
        stem = now.strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
        path = self.directory / (stem + extension)
        metadata_path = path.with_suffix(".json")
        metadata = {
            "image": path.name,
            "condition": self.condition,
            "topic": self.topic,
            "saved_at_utc": now.isoformat(),
            "ros_stamp_secs": message.header.stamp.secs,
            "ros_stamp_nsecs": message.header.stamp.nsecs,
            "frame_id": message.header.frame_id,
            "format": message.format,
            "width": int(decoded.shape[1]),
            "height": int(decoded.shape[0]),
            "receive_age_seconds": age,
        }
        # Exclusive creation prevents accidental replacement. Preserve compressed bytes.
        with path.open("xb") as stream:
            try:
                stream.write(data)
                with metadata_path.open("x", encoding="utf-8") as sidecar:
                    json.dump(metadata, sidecar, ensure_ascii=False, indent=2)
                    sidecar.write("\n")
            except Exception:
                path.unlink(missing_ok=True)
                raise
        self.saved_sequence = sequence
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/image_jpeg/compressed")
    parser.add_argument("--output", default=os.path.join(
        os.environ.get("ML_DATA_ROOT") or str(Path.home()), "rgb_samples"))
    parser.add_argument("--condition", required=True, help="e.g. clear_day, night, fog")
    parser.add_argument("--max-age", type=float, default=2.0,
                        help="Reject frames received more than this many seconds ago")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Run in an interactive terminal (Docker: use -it).")
    try:
        capture = CameraCapture(args.output, args.condition, args.topic, args.max_age)
    except PermissionError as exc:
        parser.error("Cannot write to {}. Choose a writable folder with "
                     "--output (for example: --output ~/rgb_samples). {}".format(
                         args.output, exc))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    import rospy
    from sensor_msgs.msg import CompressedImage

    rospy.init_node("keyboard_camera_capture", anonymous=True)
    subscriber = rospy.Subscriber(args.topic, CompressedImage, capture.receive,
                                  queue_size=1, buff_size=16 * 1024 * 1024)
    print("Topic: {}\nOutput: {}".format(args.topic, capture.directory), flush=True)
    print("Focus this terminal. Space/S: save latest frame | Q/Ctrl-C: quit", flush=True)
    print("Waiting for camera stream...", flush=True)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    announced = False
    try:
        tty.setcbreak(fd)
        while not rospy.is_shutdown():
            if not announced and capture.latest is not None:
                print("Camera stream received. Ready to save.", flush=True)
                announced = True
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            key = os.read(fd, 1).decode("ascii", errors="ignore").lower()
            if key in ("q", "\x03"):
                break
            if key in (" ", "s"):
                try:
                    print("Saved: {}".format(capture.save()), flush=True)
                except (ValueError, OSError, cv2.error) as exc:
                    print("Not saved: {}".format(exc), flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        subscriber.unregister()
        rospy.signal_shutdown("Keyboard capture closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
