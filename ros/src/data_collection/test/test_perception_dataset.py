#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `src/data_collection/test/test_perception_dataset.py` (원작성: 안승현)

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_collection.perception_dataset import (
    PerceptionDatasetError,
    build_dataset_version,
)


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("failed to write test PNG")


class PerceptionDatasetTest(unittest.TestCase):
    def make_cache(self, data_root, run_id, v2=False, v3=False):
        v2 = v2 or v3
        run_root = data_root / "datasets" / run_id
        cache_root = run_root / "derived/twinlite_384x640_v2"
        rows = []
        for index, view in enumerate(("front", "left", "right")):
            image_rel = "frames/intensity/{}/000000.png".format(view)
            write_png(run_root / image_rel, np.zeros((8, 12, 3), dtype=np.uint8))
            outputs = {}
            for name in ("lane", "drivable", "stopline"):
                relative = "{}_masks/{}/000000.png".format(name, view)
                write_png(cache_root / relative, np.zeros((4, 6), dtype=np.uint8))
                outputs[name] = relative
            outputs["valid"] = "valid_masks/{}/000000.png".format(view)
            write_png(cache_root / outputs["valid"], np.ones((4, 6), dtype=np.uint8))
            valid_masks = None
            if v2:
                outputs["drivable_ignore"] = "drivable_ignore_masks/{}/000000.png".format(view)
                write_png(cache_root / outputs["drivable_ignore"], np.zeros((4, 6), dtype=np.uint8))
                drivable_valid = "valid_masks/drivable/{}/000000.png".format(view)
                write_png(cache_root / drivable_valid, np.ones((4, 6), dtype=np.uint8))
                valid_masks = {
                    "lane": outputs["valid"],
                    "drivable": drivable_valid,
                    "stopline": outputs["valid"],
                }
            if v3:
                outputs["road_marking"] = (
                    "road_marking_masks/{}/000000.png".format(view)
                )
                marking = np.zeros((4, 6), dtype=np.uint8)
                marking[1, 1] = 1
                marking[2, 2] = 2
                marking[3, 3] = 3
                write_png(cache_root / outputs["road_marking"], marking)
                valid_masks["road_marking"] = outputs["valid"]
            rows.append(
                {
                    "run_id": run_id,
                    "frame_id": 0,
                    "view": view,
                    "source": {"intensity": image_rel, "semantic": "unused"},
                    "outputs": outputs,
                    "valid_masks": valid_masks,
                    "geometry": {
                        "original_hw": [8, 12],
                        "target_hw": [4, 6],
                        "scale": 0.5,
                        "scale_xy": [0.5, 0.5],
                        "resized_hw": [4, 6],
                        "pad_ltrb": [0, 0, 0, 0],
                        "valid_roi_xyxy": [0, 0, 6, 4],
                    },
                    "task_valid": {
                        "lane": True,
                        "drivable": True,
                        "stopline": True if v3 else view == "front",
                        **({"road_marking": True} if v3 else {}),
                    },
                    "target_encodings": (
                        {
                            "road_marking": {
                                "type": "categorical_uint8",
                                "class_values": {
                                    "background": 0,
                                    "white_lane": 1,
                                    "yellow_lane": 2,
                                    "stopline": 3,
                                },
                            }
                        }
                        if v3
                        else {}
                    ),
                }
            )
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "geometry.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (cache_root / "_SUCCESS").write_text(
            json.dumps(
                {
                    "sample_count": 3,
                    "transform_sha256": "test-transform",
                    "policy_id": "test_v2" if v2 else None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_builds_exclusive_run_based_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run")
            self.make_cache(data_root, "val_run")
            result = build_dataset_version(
                data_root,
                train_runs=["train_run"],
                val_runs=["val_run"],
            )
            output = Path(result["output_root"])
            self.assertTrue((output / "_SUCCESS").is_file())
            train_rows = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(train_rows), 3)
            self.assertEqual(train_rows[0]["run_id"], "train_run")
            self.assertTrue(train_rows[0]["image"].startswith("datasets/train_run/"))
            self.assertEqual(result["splits"]["val"]["sample_count"], 3)
            self.assertEqual(result["splits"]["test"]["sample_count"], 0)
            self.assertEqual(result["training_heads"], ["lane", "drivable"])

    def test_rejects_run_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PerceptionDatasetError):
                build_dataset_version(
                    Path(tmp), train_runs=["same"], val_runs=["same"]
                )

    def test_existing_version_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run")
            build_dataset_version(data_root, train_runs=["train_run"])
            with self.assertRaises(PerceptionDatasetError):
                build_dataset_version(data_root, train_runs=["train_run"])

    def test_v2_cache_propagates_head_valid_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run", v2=True)
            result = build_dataset_version(
                data_root,
                train_runs=["train_run"],
                dataset_name="v2",
            )
            row = json.loads(
                (Path(result["output_root"]) / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("drivable", row["valid_masks"])
            self.assertIn("drivable_ignore", row["auxiliary_targets"])
            self.assertEqual(result["target_policy_ids"], ["test_v2"])
            self.assertEqual(
                result["head_valid_policy"]["drivable"],
                "per-sample valid_masks.drivable (policy-specific)",
            )

    def test_v3_cache_propagates_multiclass_road_marking_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run", v3=True)
            result = build_dataset_version(
                data_root,
                train_runs=["train_run"],
                dataset_name="v3",
                training_heads=("drivable", "road_marking"),
            )
            row = json.loads(
                (Path(result["output_root"]) / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("road_marking", row["targets"])
            self.assertIn("road_marking", row["valid_masks"])
            self.assertEqual(
                row["target_encodings"]["road_marking"]["class_values"]["stopline"],
                3,
            )
            self.assertEqual(result["training_heads"], ["drivable", "road_marking"])
            self.assertEqual(
                result["target_encodings"]["road_marking"]["class_values"],
                {
                    "background": 0,
                    "white_lane": 1,
                    "yellow_lane": 2,
                    "stopline": 3,
                },
            )
            self.assertEqual(
                result["head_valid_policy"]["road_marking"],
                "geometric_valid AND per-sample task_valid",
            )

    def test_rejects_training_head_absent_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run", v2=True)
            with self.assertRaisesRegex(
                PerceptionDatasetError, "training heads are absent"
            ):
                build_dataset_version(
                    data_root,
                    train_runs=["train_run"],
                    dataset_name="missing_marking",
                    training_heads=("drivable", "road_marking"),
                )

    def test_rejects_duplicate_training_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run", v3=True)
            with self.assertRaisesRegex(PerceptionDatasetError, "duplicates"):
                build_dataset_version(
                    data_root,
                    train_runs=["train_run"],
                    dataset_name="duplicate_heads",
                    training_heads=("drivable", "drivable"),
                )

    def test_excludes_complete_capture_frame_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run")
            self.make_cache(data_root, "val_run")
            result = build_dataset_version(
                data_root,
                train_runs=["train_run"],
                val_runs=["val_run"],
                excluded_frames_by_run={"val_run": {0}},
            )
            output = Path(result["output_root"])
            self.assertEqual(result["splits"]["val"]["sample_count"], 0)
            self.assertEqual(result["explicit_frame_exclusions"], {"val_run": [0]})
            val_summary = next(
                row for row in result["runs"] if row["run_id"] == "val_run"
            )
            self.assertEqual(val_summary["excluded_frame_ids"], [0])
            self.assertEqual(val_summary["excluded_sample_count"], 3)
            self.assertEqual((output / "val.jsonl").read_text(encoding="utf-8"), "")

    def test_rejects_exclusion_for_unassigned_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.make_cache(data_root, "train_run")
            with self.assertRaises(PerceptionDatasetError):
                build_dataset_version(
                    data_root,
                    train_runs=["train_run"],
                    excluded_frames_by_run={"other_run": {0}},
                )


if __name__ == "__main__":
    unittest.main()
