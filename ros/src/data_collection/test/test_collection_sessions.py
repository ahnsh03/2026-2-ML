import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from data_collection.compressed_rgb import CompressedRGBSession
from data_collection.session_archive import completed_receipt, verified_frames


CLI = Path(__file__).resolve().parents[4] / "scripts" / "capture_sessions.py"


class CollectionSessionTests(unittest.TestCase):
    def cli(self, *args, ok=True):
        result = subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def test_rgb_session_restored_with_all_pixels_and_no_labels(self):
        buf = cv2.imencode(".png", np.zeros((20, 30, 4), dtype=np.uint8))[1].tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with CompressedRGBSession(root / "drive", "/front", {"parent_session_id": "drive1"}) as writer:
                writer.append(buf, "png", 1, 0, "front")
                writer.append(buf, "png", 2, 0, "front")
            moved = Path(shutil.copytree(root / "drive", root / "restore"))
            self.assertEqual(completed_receipt(moved)["context"]["parent_session_id"], "drive1")
            self.assertEqual(len(list(verified_frames(moved))), 2)
            out = root / "export"
            self.cli("export", moved, "--output", out)
            self.assertEqual((out / "rgb.jsonl").read_bytes(), (moved / "rgb.jsonl").read_bytes())
            self.assertEqual((out / "paired.jsonl").stat().st_size, 0)
            (moved / "rgb.jsonl").write_text("{}\n")
            self.cli("verify", moved, ok=False)

    def test_native_register_relocate_and_reject_missing_or_corrupt_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = root / "data" / "cam"
            camera.mkdir(parents=True)
            raw = cv2.imencode(".png", np.full((20, 30, 3), 42, np.uint8))[1].tobytes()
            hashes = {}
            for channel in ("Intensity", "Semantic", "Instance", "Depth"):
                (camera / ("c_" + channel + ".png")).write_bytes(raw)
                hashes[channel] = hashlib.sha256(raw).hexdigest()
            src = root / "captures.jsonl"
            record = dict(name="c", id="original", role="final_test", sha256=hashes)
            src.write_text(json.dumps(record) + "\n")
            ctx = root / "context.json"
            ctx.write_text(json.dumps(dict(split_role="meta_train", parent_session_id="route_1")))
            session = root / "sessions" / "archive"
            args = ["register-front", "--manifest", src, "--media-root", camera.parent, "--camera-dir", "cam",
                    "--session-id", "drive", "--context", ctx, "--population", "all_captured"]
            self.cli(*args, "--output", session)
            self.cli(*args, "--output", session, ok=False)
            self.cli("verify", session, ok=False)
            media = root / "restored"
            camera.parent.rename(media)
            self.cli("verify", session, "--media-root", media)
            exp = root / "export"
            self.cli("export", session, "--media-root", media, "--output", exp)
            row = json.loads((exp / "paired.jsonl").read_text())
            self.assertEqual((row["sample_id"], row["split_group"], row["preregistered_quality_role"]),
                             ("original", "route_1", "final_test"))
            self.assertEqual(set(json.loads((exp / "rgb.jsonl").read_text())),
                             {"sample_id", "image", "view", "expected_sha256"})
            record["files"] = [{"path": "cam/c_" + k + ".png", "sha256": h} for k, h in hashes.items()]
            record["files"].pop()
            src.write_text(json.dumps(record) + "\n")
            args[args.index("--media-root") + 1] = media
            self.cli(*args, "--output", root / "incomplete", ok=False)
            self.assertFalse((root / "incomplete").exists())
            (media / "cam/c_Semantic.png").write_bytes(bytes(len(raw)))
            self.cli("export", session, "--media-root", media, "--output", root / "bad-export", ok=False)
            self.assertFalse((root / "bad-export").exists())
            (session / "_SUCCESS").unlink()
            status = json.loads(self.cli("list", session.parent).stdout)
            self.assertEqual(status["status"], "unsealed_or_invalid")
