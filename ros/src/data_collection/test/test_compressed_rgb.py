import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data_collection.compressed_rgb import CompressedRGBSession


class CompressedRGBTests(unittest.TestCase):
    def payload(self, ext):
        image = np.zeros((8, 12, 3), np.uint8)
        image[:, 3:, 2] = 190
        ok, data = cv2.imencode(ext, image)
        self.assertTrue(ok)
        return data.tobytes()

    def test_wire_bytes_and_metadata_survive_repeated_camera_stamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "drive"
            stream = io.StringIO()
            with CompressedRGBSession(target, "/front") as writer:
                first = writer.append(self.payload(".jpg"), "bgr8; jpeg compressed bgr8", 2, 140, "front", stream)
                second = writer.append(self.payload(".png"), "png", 4, 140, "front", stream)
            manifest = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(manifest, [first, second])
            self.assertEqual(stream.getvalue(), (target / "rgb.jsonl").read_text())
            self.assertNotEqual(first["sample_id"], second["sample_id"])
            self.assertEqual((target / first["image"]).read_bytes(), self.payload(".jpg"))
            self.assertEqual(set(first), {"view", "sample_id", "image", "expected_sha256"})
            transport = [json.loads(line) for line in (target / "transport.jsonl").read_text().splitlines()]
            self.assertEqual([row["ros_stamp_ns"] for row in transport], [140, 140])
            self.assertEqual(json.loads((target / "run.json").read_text())["sequence_gaps"], 1)
            self.assertEqual((target / "_SUCCESS").read_text().strip(), hashlib.sha256((target / "run.json").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                with CompressedRGBSession(target, "/front"):
                    pass

    def test_corrupt_or_mislabeled_frame_does_not_emit_a_manifest_row(self):
        for payload, declaration in [(self.payload(".png"), "jpeg"), (b"bad", "png"), (self.payload(".png"), "compressedDepth png")]:
            with self.subTest(format=declaration), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "drive"
                with self.assertRaises(ValueError):
                    with CompressedRGBSession(target, "/front") as writer:
                        writer.append(payload, declaration, 0, 0, "front")
                self.assertEqual((target / "rgb.jsonl").read_text(), "")
                self.assertEqual(json.loads((target / "run.json").read_text())["status"], "failed")
                self.assertFalse((target / "_SUCCESS").exists())
