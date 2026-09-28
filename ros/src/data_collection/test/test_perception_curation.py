#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `src/data_collection/test/test_perception_curation.py` (원작성: 안승현)

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_collection.perception_curation import curate_run, stationary_groups


class PerceptionCurationTest(unittest.TestCase):
    def test_stationary_group_and_periodic_selection(self):
        rows = []
        for frame_id in range(8):
            rows.append(
                {
                    "frame_id": frame_id,
                    "state_snapshot": {
                        "ego": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}}
                    },
                }
            )
        self.assertEqual(stationary_groups(rows), [(0, 7)])

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "datasets" / "run_001"
            for row in rows:
                row["run_id"] = "run_001"
                row["valid"] = True
                row["paths"] = {"semantic": {}}
                for view in ("front", "left", "right"):
                    relative = "frames/semantic/{}/{:06d}.png".format(
                        view, row["frame_id"]
                    )
                    path = run_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image = np.zeros((12, 16, 3), dtype=np.uint8)
                    if row["frame_id"] == 4:
                        image[2:10, 3:12] = (2, 255, 236)  # obstacle RGB reversed
                    self.assertTrue(cv2.imwrite(str(path), image))
                    row["paths"]["semantic"][view] = relative
            (run_root / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = curate_run(
                run_root,
                max_stationary_gap_frames=3,
                focus_change_ratio=0.001,
                signature_hw=(12, 16),
            )
            decisions = [
                json.loads(line)
                for line in (Path(result["output_root"]) / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            selected = {row["frame_id"] for row in decisions if row["selected"]}
            self.assertIn(0, selected)
            self.assertIn(3, selected)  # periodic
            self.assertIn(4, selected)  # focus change
            self.assertIn(7, selected)  # ending anchor or focus change back
            self.assertLess(len(selected), len(rows))


if __name__ == "__main__":
    unittest.main()
