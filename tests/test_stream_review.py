import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from common.bridge import sha256_file
from quality_estimation.stream_review import run_stream


class RecordingScorer:
    centers = centers_bytes = None
    device = "cpu"
    metadata = {"model": "RandomForest", "uses_ground_truth": False}

    def score(self, images, items):
        item = items[0]
        value = dict(sample_id=item["sample_id"], pixel_sha256=item["pixel_sha256"],
                     rgb_sha256=item["sha256"], estimated_iou=.25)
        return [value], None, None


class StreamReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "out"

    def rgb(self, i):
        path = self.root / f"{i}.png"
        cv2.imwrite(str(path), np.full((12, 16, 3), i, np.uint8))
        return json.dumps(dict(sample_id=str(i), view="front", image=path.name,
                               expected_sha256={"rgb": sha256_file(path)}))

    def test_flushes_selected_originals_before_reading_the_next_window(self):
        def input_lines():
            yield self.rgb(0)
            yield self.rgb(1)
            first = json.loads((self.out / "queue.jsonl").read_text())
            self.assertEqual(first["sample_id"], "0")
            self.assertEqual(sha256_file(self.out / first["stored_image"]), first["rgb_sha256"])
            self.assertFalse((self.out / "_SUCCESS").exists())
            yield self.rgb(2)
        result = run_stream(input_lines(), RecordingScorer(), self.root, self.out, 2, 50, 0)
        self.assertEqual(result, dict(input_rows=3, unique_pixels=3, windows=2, selected=2))
        self.assertEqual((self.out / "_SUCCESS").read_text().strip(), sha256_file(self.out / "run.json"))

    def test_rejects_non_rgb_fields_and_marks_partial_work(self):
        lines = [self.rgb(1), json.dumps(dict(sample_id="2", image="2.png", view="front", semantic="labels.png"))]
        with self.assertRaisesRegex(ValueError, "RGB-only"):
            run_stream(lines, RecordingScorer(), self.root, self.out, 1, 50)
        self.assertFalse((self.out / "_SUCCESS").exists())
        run = json.loads((self.out / "run.json").read_text())
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["selected"], 1)
