import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.bridge import sha256_file


def entrypoint(name):
    directory = (
        Path(__file__).resolve().parents[1] / "scripts/experiments/roi9_gp_20261125"
    )
    spec = importlib.util.spec_from_file_location(
        "test_" + name + "_entrypoint", directory / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(directory), *sys.path]):
        spec.loader.exec_module(module)
    return module.main


class CaptureRegistrationTests(unittest.TestCase):
    def fixture(self, root):
        collection = root / "capture"
        collection.mkdir()
        files = {}
        for channel in ["Intensity", "Semantic", "Depth", "Instance"]:
            file = collection / (channel + ".png")
            file.write_bytes(channel.encode())
            files[channel] = dict(path=file.name, sha256=sha256_file(file))
        state = dict(
            weather=dict(type=0),
            time=dict(hour=13),
            ego=dict(
                vehicle_state=dict(gear_mode="GEAR_MODE_P"),
                transform=dict(location=dict(x=0, y=0, z=0)),
            ),
        )
        row = dict(
            sample_id="frame",
            site="site400",
            before=state,
            after=state,
            files=files,
            weather=0,
            hour=13,
            simulator_release="26.R1",
            completion_basis="test verified metadata",
        )
        (collection / "manifest.jsonl").write_text(json.dumps(row) + "\n")
        prior = root / "prior.json"
        prior.write_text(json.dumps(dict(old=dict(rows=[dict(sample_id="frame")]))))
        return [
            "register_capture",
            "--data-root",
            str(root),
            "--collections",
            str(collection),
            "--key",
            "new",
            "--expected-rows",
            "1",
            "--existing-index",
            str(prior),
            "--output",
            str(root / "new.json"),
        ]

    def test_existing_ids_are_rejected_and_namespace_preserves_capture_identity(self):
        main = entrypoint("register_capture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.fixture(root)
            with (
                patch("sys.argv", argv),
                self.assertRaisesRegex(ValueError, "already registered"),
            ):
                main()
            self.assertFalse((root / "new.json").exists())
            with (
                patch("sys.argv", argv + ["--sample-id-prefix", "acq__"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            row = json.loads((root / "new.json").read_text())["new"]["rows"][0]
            self.assertEqual(
                (row["sample_id"], row["capture_sample_id"]), ("acq__frame", "frame")
            )
            self.assertEqual(
                (row["site"], row["capture_site"]), ("acq__site400", "site400")
            )

    def test_incomplete_raw_channel_set_cannot_be_registered(self):
        main = entrypoint("register_capture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.fixture(root)
            manifest = root / "capture/manifest.jsonl"
            row = json.loads(manifest.read_text())
            row["files"].pop("Depth")
            manifest.write_text(json.dumps(row) + "\n")
            with (
                patch("sys.argv", argv + ["--sample-id-prefix", "new__"]),
                self.assertRaisesRegex(ValueError, "four raw"),
            ):
                main()

    def test_geometry_planner_rejects_reserved_site_ids_before_reading_map(self):
        main = entrypoint("select_transfer_sites")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "plan.json"
            prior.write_text(json.dumps(dict(poses=[dict(id=400)])))
            argv = [
                "select_transfer_sites",
                "--packs",
                "not-opened",
                "--index",
                "not-opened",
                "--map",
                "not-opened",
                "--template",
                "not-opened",
                "--output",
                str(root / "new.json"),
                "--sites",
                "4",
                "--site-id-start",
                "400",
                "--exclude-plan",
                str(prior),
            ]
            with (
                patch("sys.argv", argv),
                self.assertRaisesRegex(ValueError, "site IDs"),
            ):
                main()
            self.assertFalse((root / "new.json").exists())


if __name__ == "__main__":
    unittest.main()
