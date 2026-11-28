"""Validate saved camera payloads and rejection of missing/stale/broken frames."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/keyboard_camera_capture.py"
SPEC = importlib.util.spec_from_file_location("keyboard_camera_capture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.capture = MODULE.CameraCapture(self.temp.name, "night", "/camera")

    def message(self, extension=".jpg"):
        pixels = np.zeros((24, 32, 3), dtype=np.uint8)
        pixels[:, :, 2] = 200
        success, encoded = cv2.imencode(extension, pixels)
        self.assertTrue(success)
        msg = CompressedImage()
        msg.data = encoded.tobytes()
        msg.header.frame_id = "Camera-1"
        msg.header.stamp.secs = 123
        msg.header.stamp.nsecs = 456
        return msg

    def test_original_bytes_dimensions_and_metadata(self):
        for extension in (".jpg", ".png"):
            msg = self.message(extension)
            self.capture.receive(msg)
            path = self.capture.save()
            self.assertEqual(path.suffix, extension)
            self.assertEqual(path.read_bytes(), msg.data)
            meta = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual((meta["width"], meta["height"]), (32, 24))
            self.assertEqual(meta["condition"], "night")
            self.assertEqual(meta["ros_stamp_secs"], 123)
            self.assertEqual(meta["ros_stamp_nsecs"], 456)

    def test_missing_duplicate_and_stale_frames(self):
        with self.assertRaisesRegex(ValueError, "No image"):
            self.capture.save()
        self.capture.receive(self.message())
        path = self.capture.save()
        with self.assertRaisesRegex(ValueError, "already saved"):
            self.capture.save()
        self.capture.receive(self.message())
        msg, _, seq = self.capture.latest
        self.capture.latest = (msg, time.monotonic() - 3, seq)
        with self.assertRaisesRegex(ValueError, "old"):
            self.capture.save()
        self.assertEqual(len(list(path.parent.glob("*.jpg"))), 1)

    def test_corrupt_payload_creates_no_files(self):
        for data in (b"not an image", b"\xff\xd8\xffbroken"):
            msg = self.message()
            msg.data = data
            self.capture.receive(msg)
            with self.assertRaises(ValueError):
                self.capture.save()
        self.assertEqual(list(self.capture.directory.iterdir()), [])

    def test_invalid_settings(self):
        for condition in ("../night", "", "night/day"):
            with self.assertRaises(ValueError):
                MODULE.CameraCapture(self.temp.name, condition, "/camera")
        for age in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MODULE.CameraCapture(self.temp.name, "night", "/camera", age)


if __name__ == "__main__":
    unittest.main()
