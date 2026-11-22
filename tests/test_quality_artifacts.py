"""Source-policy agreement and corruption/partial-output rejection."""
import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from quality_estimation.artifacts import FeatureDatasetWriter
from quality_estimation.build_dataset import native_marking, REPO_ROOT
from quality_estimation.dataset_io import load_dataset
from quality_estimation.labels import build_label
from quality_estimation.native import NativeSamples, decoded_pixel_sha256
from data_collection.mask_baker import OFFICIAL_PALETTE_RGB, semantic_policy_masks


class SourceTests(unittest.TestCase):
    def test_native_target_agrees_with_frozen_independent_policy(self):
        colors = [rgb[::-1] for name, rgb in OFFICIAL_PALETTE_RGB.items() if name != "blue_lane"]
        semantic = np.array([colors], np.uint8)
        policy = json.loads((REPO_ROOT / "config/perception_target_policy_v3_actor_negative.json").read_text())
        np.testing.assert_array_equal(native_marking(semantic), semantic_policy_masks(semantic, policy)["road_marking"])

    def test_unknown_or_blue_semantic_is_not_silently_background(self):
        for bgr in ((1, 1, 1), (255, 178, 0)):
            with self.assertRaises(ValueError):
                native_marking(np.array([[bgr]], np.uint8))

    def test_pair_rejects_stale_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = np.full((6, 4, 3), 127, np.uint8)
            cv2.imwrite(str(root / "rgb.png"), image)
            cv2.imwrite(str(root / "gt.png"), image)
            rows = [{"sample_id": "run/front/0", "view": "front", "image": "rgb.png", "semantic": "gt.png",
                     "expected_sha256": {"rgb": "0" * 64}}]
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                NativeSamples(root, rows)[0]

    def test_pixel_identity_is_shape_and_color_sensitive(self):
        a = np.arange(72, dtype=np.uint8).reshape(6, 4, 3)
        self.assertNotEqual(decoded_pixel_sha256(a), decoded_pixel_sha256(a.reshape(4, 6, 3)))
        self.assertNotEqual(decoded_pixel_sha256(a), decoded_pixel_sha256(a[..., ::-1]))
        self.assertEqual(decoded_pixel_sha256(a), decoded_pixel_sha256(a.copy()))

    def test_source_bgr_hash_does_not_change_operational_rgb(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bgr = np.zeros((6, 4, 3), np.uint8)
            bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2] = 10, 40, 200
            cv2.imwrite(str(root / "rgb.png"), bgr)
            cv2.imwrite(str(root / "gt.png"), np.full_like(bgr, 127))
            row = {"sample_id": "r/front/0", "view": "front", "image": "rgb.png", "semantic": "gt.png",
                   "expected_pixel_sha256": decoded_pixel_sha256(bgr), "expected_pixel_color_order": "BGR"}
            item = NativeSamples(root, [row])[0]
            np.testing.assert_array_equal(item["rgb"][0, 0], [200, 40, 10])
            self.assertNotEqual(item["pixel_sha256"], row["expected_pixel_sha256"])


class ArtifactTests(unittest.TestCase):
    def write_one(self, out):
        pred = np.ones((3, 2), np.uint8)
        label = build_label(pred, pred, np.ones_like(pred, bool), np.ones_like(pred, bool))
        metadata = {"checkpoint_sha256": DEFAULT_CHECKPOINT_SHA256}
        with FeatureDatasetWriter(out, metadata) as writer:
            writer.add(np.zeros(56, np.float32), label,
                       {"sample_id": "r/front/0", "run_id": "r", "parent_session_id": "p"},
                       {"probabilities": np.ones((4, 3, 2), np.float32) / 4})
            writer.finalize()

    def test_probability_fixture_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "complete"
            self.write_one(out)
            self.assertEqual(load_dataset(out).X.shape, (1, 56))
            fixture = next((out / "audit").glob("*.npz"))
            fixture.write_bytes(fixture.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_dataset(out)

    def test_failure_cannot_look_like_a_completed_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "partial"
            with self.assertRaisesRegex(RuntimeError, "source broke"):
                with FeatureDatasetWriter(out, {}) as writer:
                    raise RuntimeError("source broke")
            self.assertFalse((out / "_SUCCESS").exists())
            self.assertEqual(json.loads((out / "run.json").read_text())["status"], "failed")
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_dataset(out)

    def test_completed_dataset_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "complete"
            self.write_one(out)
            previous = (out / "dataset.json").read_bytes()
            with self.assertRaises(FileExistsError):
                FeatureDatasetWriter(out, {})
            self.assertEqual((out / "dataset.json").read_bytes(), previous)

    def test_label_count_audit_remains_active_under_optimized_python(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'complete';self.write_one(out)
            with (out/'labels.csv').open() as stream:
                reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
            rows[0]['tp']='5'  # union remains 6; artifact hashes alone cannot check this identity.
            with (out/'labels.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            meta=json.loads((out/'dataset.json').read_text())
            meta['artifact_sha256']['labels.csv']=sha256_file(out/'labels.csv')
            (out/'dataset.json').write_text(json.dumps(meta))
            (out/'_SUCCESS').write_text(sha256_file(out/'dataset.json')+'\n')
            script='from quality_estimation.verify_dataset import verify; import sys; verify(sys.argv[1],sys.argv[1])'
            result=subprocess.run([sys.executable,'-O','-c',script,str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('union == tp + fp + fn',result.stderr)


if __name__ == "__main__":
    unittest.main()
