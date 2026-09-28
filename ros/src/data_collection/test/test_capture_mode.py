#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `src/data_collection/test/test_capture_mode.py` (원작성: 안승현)

import json
import os
import tempfile
import unittest

from data_collection.capture_mode import (
    CaptureRunWriter,
    CaptureSchedule,
    sanitize_name,
    speed_mps_from_ego,
)


class FakeClock(object):
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class CaptureModeTest(unittest.TestCase):
    def test_sanitize_name_blocks_paths(self):
        self.assertEqual(sanitize_name(" ../a b/c "), "a_b_c")
        self.assertEqual(sanitize_name(""), "capture")

    def test_schedule_skips_missed_slots(self):
        clock = FakeClock(10.0)
        schedule = CaptureSchedule(1.0, start_delay_sec=2.0, monotonic_fn=clock)
        self.assertEqual(schedule.remaining(), 2.0)
        clock.value = 14.5
        schedule.advance()
        self.assertAlmostEqual(schedule.next_deadline, 15.5)

    def test_writer_is_append_only_and_summarizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = CaptureRunWriter(tmp, "run-1", {"capture_hz": 1.0})
            writer.write_capture({"sequence": 0, "success": True})
            writer.write_capture({"sequence": 1, "success": False})
            writer.mark_motion_skip()
            root = writer.root
            writer.close()

            with open(os.path.join(root, "capture_manifest.jsonl")) as stream:
                rows = [json.loads(line) for line in stream]
            with open(os.path.join(root, "summary.json")) as stream:
                summary = json.load(stream)
            self.assertEqual([row["sequence"] for row in rows], [0, 1])
            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["skipped_motion"], 1)

    def test_speed_uses_vector_norm(self):
        self.assertAlmostEqual(
            speed_mps_from_ego({"velocity": {"x": 3, "y": 4, "z": 0}}), 5.0
        )


if __name__ == "__main__":
    unittest.main()
