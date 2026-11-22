import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from common.bridge import sha256_file
from quality_estimation.fit_environment import training_descriptors
from quality_estimation.image_io import decoded_pixel_sha256


class EnvironmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        image = np.full((720, 1280, 3), (50, 70, 90), dtype=np.uint8)
        cv2.imwrite(str(self.path / "frame.png"), image)
        self.member = dict(sample_id="first", rgb_sha256=sha256_file(self.path / "frame.png"),
                           pixel_sha256=decoded_pixel_sha256(image[:, :, ::-1]))
        self.write_model([self.member, dict(self.member, sample_id="copy")])
        self.manifest = self.path / "rgb.jsonl"
        self.manifest.write_text("".join(json.dumps(dict(sample_id=i, view="front", image=name)) + "\n"
                                         for i, name in [("held-out", "absent.png"), ("copy", "frame.png"), ("first", "frame.png")]))

    def write_model(self, rows):
        membership = self.path / "train_membership.json"
        membership.write_text(json.dumps(rows))
        run = self.path / "run.json"
        run.write_text(json.dumps(dict(status="complete", artifact_sha256={membership.name: sha256_file(membership)})))
        (self.path / "_SUCCESS").write_text(sha256_file(run) + "\n")

    def test_deduplicates_pixels_in_training_order_and_skips_extra_images(self):
        features, kept, source = training_descriptors(self.path, self.manifest, self.path)
        self.assertEqual(features.shape, (1, 73))
        self.assertEqual([r["sample_id"] for r in kept], ["first"])
        self.assertEqual(source["membership_rows"], 2)

    def test_checks_saved_pixels_and_membership_hash(self):
        self.write_model([dict(self.member, pixel_sha256="bad")])
        with self.assertRaisesRegex(ValueError, "pixels differ"):
            training_descriptors(self.path, self.manifest, self.path)
        (self.path / "train_membership.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "membership differs"):
            training_descriptors(self.path, self.manifest, self.path)
