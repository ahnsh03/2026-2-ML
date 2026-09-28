#!/usr/bin/env python3
# 이관: 인지 코드 레포 d0922bb `src/data_collection/test/test_mask_baker.py` (원작성: 안승현)

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_collection.mask_baker import (
    MaskBakeError,
    bake_native_run,
    build_model_cache,
    letterbox_mask,
)


VIEW_HW = {
    "front": (720, 1280),
    "left": (480, 640),
    "right": (480, 640),
}


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("failed to write test PNG")


class MaskBakerTest(unittest.TestCase):
    def make_run(self, root):
        run_root = root / "datasets" / "run_001"
        paths = {"intensity": {}, "semantic": {}}
        for view, (height, width) in VIEW_HW.items():
            rgb = np.zeros((height, width, 4), dtype=np.uint8)
            rgb[:, :, 3] = 255
            semantic = np.zeros((height, width, 4), dtype=np.uint8)
            semantic[:, :, 3] = 255
            semantic[height // 2 :, :, :3] = (127, 127, 127)  # Asphalt (BGR)
            semantic[height // 2 :, width // 3 : width // 3 + 8, :3] = (
                255,
                255,
                255,
            )
            semantic[height * 3 // 4 : height * 3 // 4 + 8, :, :3] = (
                0,
                0,
                255,
            )  # Stop Line RGB=(255,0,0)

            rgb_rel = "frames/intensity/{}/000000.png".format(view)
            sem_rel = "frames/semantic/{}/000000.png".format(view)
            write_png(run_root / rgb_rel, rgb)
            write_png(run_root / sem_rel, semantic)
            paths["intensity"][view] = rgb_rel
            paths["semantic"][view] = sem_rel

        manifest_row = {
            "run_id": "run_001",
            "frame_id": 0,
            "valid": True,
            "paths": paths,
        }
        (run_root / "manifest.jsonl").write_text(
            json.dumps(manifest_row) + "\n", encoding="utf-8"
        )
        (run_root / "dataset.json").write_text(
            json.dumps(
                {
                    "schema_version": "capture-sync-1.0.0",
                    "run_id": "run_001",
                    "frame_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return run_root

    def test_letterbox_geometry_for_front_and_side(self):
        front, front_geometry = letterbox_mask(np.ones((720, 1280), dtype=np.uint8))
        side, side_geometry = letterbox_mask(np.ones((480, 640), dtype=np.uint8))
        self.assertEqual(front.shape, (384, 640))
        self.assertEqual(front_geometry["pad_ltrb"], [0, 12, 0, 12])
        self.assertEqual(int(front[:12].sum()), 0)
        self.assertEqual(side_geometry["pad_ltrb"], [64, 0, 64, 0])
        self.assertEqual(int(side[:, :64].sum()), 0)

    def test_native_then_cache_writes_view_folders_geometry_and_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = self.make_run(Path(tmp))
            native_summary = bake_native_run(run_root, write_previews=True)
            native = run_root / "derived/perception_targets_native_v1"
            self.assertTrue((native / "_SUCCESS").is_file())
            self.assertEqual(native_summary["sample_count"], 3)
            native_lane = cv2.imread(
                str(native / "lane_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(native_lane.shape, (720, 1280))
            self.assertTrue((native / "qa_overlays/right/000000.png").is_file())

            summary = build_model_cache(run_root, write_previews=True)
            output = run_root / "derived/twinlite_384x640_v2"
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertEqual(summary["sample_count"], 3)

            stopline = cv2.imread(
                str(output / "stopline_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            drivable = cv2.imread(
                str(output / "drivable_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            valid = cv2.imread(
                str(output / "valid_masks/left/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(stopline.shape, (384, 640))
            self.assertEqual(set(np.unique(stopline).tolist()), {0, 1})
            self.assertGreater(int(stopline.sum()), 0)
            self.assertTrue(np.all(drivable[stopline == 1] == 1))
            self.assertEqual(int(valid[:, :64].sum()), 0)

            geometry_rows = [
                json.loads(line)
                for line in (output / "geometry.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(geometry_rows), 3)
            by_view = {row["view"]: row for row in geometry_rows}
            self.assertTrue(by_view["front"]["task_valid"]["stopline"])
            self.assertFalse(by_view["left"]["task_valid"]["stopline"])
            self.assertTrue((output / "qa_overlays/front/000000.png").is_file())

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = self.make_run(Path(tmp))
            bake_native_run(run_root)
            with self.assertRaises(MaskBakeError):
                bake_native_run(run_root)

    def test_shape_mismatch_is_rejected_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = self.make_run(Path(tmp))
            wrong = np.zeros((10, 10, 3), dtype=np.uint8)
            write_png(run_root / "frames/semantic/front/000000.png", wrong)
            with self.assertRaises(MaskBakeError):
                bake_native_run(run_root)
            self.assertFalse(
                (run_root / "derived/perception_targets_native_v1").exists()
            )

    def test_unknown_semantic_color_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = self.make_run(Path(tmp))
            semantic_path = run_root / "frames/semantic/front/000000.png"
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            semantic[0, 0, :3] = (3, 2, 1)  # RGB=(1,2,3), not MORAI 26.R1.
            write_png(semantic_path, semantic)
            with self.assertRaises(MaskBakeError):
                bake_native_run(run_root)

    def test_v2_policy_writes_actor_masks_and_head_specific_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = self.make_run(root)
            semantic_path = run_root / "frames/semantic/front/000000.png"
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            semantic[400:500, 500:600, :3] = (2, 255, 236)  # obstacle RGB reversed
            write_png(semantic_path, semantic)
            policy = {
                "status": "frozen",
                "policy_id": "test_v2",
                "targets": {
                    "lane": {"positive_classes": ["white_lane"]},
                    "drivable": {
                        "positive_classes": ["asphalt", "white_lane", "stopline"],
                        "ignore_classes": ["obstacle"],
                    },
                    "stopline": {"positive_classes": ["stopline"]},
                    "actor_occupancy_gt": {"positive_classes": ["obstacle"]},
                },
            }
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")
            native_summary = bake_native_run(
                run_root,
                output_name="perception_targets_native_v2",
                policy_path=policy_path,
            )
            self.assertEqual(native_summary["schema_version"], "perception-targets-native-2.0.0")
            native = run_root / "derived/perception_targets_native_v2"
            ignore = cv2.imread(
                str(native / "drivable_ignore_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            occupancy = cv2.imread(
                str(native / "actor_occupancy_gt_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(int(ignore.sum()), 10000)
            self.assertTrue(np.array_equal(ignore, occupancy))

            build_model_cache(
                run_root,
                native_output_name="perception_targets_native_v2",
                output_name="twinlite_384x640_v3",
            )
            cache = run_root / "derived/twinlite_384x640_v3"
            geometric = cv2.imread(
                str(cache / "valid_masks/geometric/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            drivable_valid = cv2.imread(
                str(cache / "valid_masks/drivable/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            baked_ignore = cv2.imread(
                str(cache / "drivable_ignore_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertGreater(int(baked_ignore.sum()), 0)
            self.assertTrue(np.all(drivable_valid[baked_ignore == 1] == 0))
            self.assertTrue(np.all(geometric[baked_ignore == 1] == 1))

    def test_v3_policy_writes_multiclass_marking_and_all_view_stopline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = self.make_run(root)
            for view, (height, width) in VIEW_HW.items():
                semantic_path = run_root / "frames/semantic/{}/000000.png".format(view)
                semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
                semantic[height // 2 :, width // 2 : width // 2 + 8, :3] = (
                    0,
                    255,
                    255,
                )  # Yellow lane RGB=(255,255,0)
                write_png(semantic_path, semantic)
            policy = {
                "status": "frozen",
                "policy_id": "test_v3",
                "audit_constraints": {"zero_pixel_classes": ["blue_lane"]},
                "targets": {
                    "lane": {
                        "positive_classes": ["white_lane", "yellow_lane"]
                    },
                    "drivable": {
                        "positive_classes": [
                            "asphalt",
                            "white_lane",
                            "yellow_lane",
                            "stopline",
                        ],
                        "ignore_classes": ["obstacle"],
                    },
                    "stopline": {
                        "positive_classes": ["stopline"],
                        "valid_views": ["front", "left", "right"],
                    },
                    "road_marking": {
                        "class_values": {
                            "background": 0,
                            "white_lane": 1,
                            "yellow_lane": 2,
                            "stopline": 3,
                        },
                        "valid_views": ["front", "left", "right"],
                    },
                    "actor_occupancy_gt": {"positive_classes": ["obstacle"]},
                },
            }
            policy_path = root / "policy_v3.json"
            policy_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")

            summary = bake_native_run(
                run_root,
                output_name="perception_targets_native_v3",
                policy_path=policy_path,
            )
            self.assertEqual(
                summary["schema_version"], "perception-targets-native-3.0.0"
            )
            native = run_root / "derived/perception_targets_native_v3"
            rows = [
                json.loads(line)
                for line in (native / "manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(all(row["task_valid"]["stopline"] for row in rows))
            self.assertTrue(all(row["task_valid"]["road_marking"] for row in rows))
            marking = cv2.imread(
                str(native / "road_marking_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(set(np.unique(marking).tolist()), {0, 1, 2, 3})

            cache_summary = build_model_cache(
                run_root,
                native_output_name="perception_targets_native_v3",
                output_name="twinlite_384x640_marking_v1",
            )
            self.assertEqual(
                cache_summary["schema_version"], "twinlite-mask-cache-3.0.0"
            )
            cached = cv2.imread(
                str(
                    run_root
                    / "derived/twinlite_384x640_marking_v1/road_marking_masks/front/000000.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(cached.shape, (384, 640))
            self.assertTrue(set(np.unique(cached).tolist()).issubset({0, 1, 2, 3}))

    def test_v3_policy_rejects_unexpected_blue_lane_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = self.make_run(root)
            semantic_path = run_root / "frames/semantic/front/000000.png"
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            semantic[100:110, 100:110, :3] = (255, 178, 0)
            write_png(semantic_path, semantic)
            policy = {
                "status": "frozen",
                "policy_id": "test_v3_blue_gate",
                "audit_constraints": {"zero_pixel_classes": ["blue_lane"]},
                "targets": {
                    "lane": {"positive_classes": ["white_lane", "yellow_lane"]},
                    "drivable": {
                        "positive_classes": ["asphalt", "white_lane", "yellow_lane"],
                        "ignore_classes": [],
                    },
                    "stopline": {"positive_classes": ["stopline"]},
                    "road_marking": {
                        "class_values": {
                            "background": 0,
                            "white_lane": 1,
                            "yellow_lane": 2,
                            "stopline": 3,
                        }
                    },
                    "actor_occupancy_gt": {"positive_classes": []},
                },
            }
            policy_path = root / "policy_v3.json"
            policy_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")
            with self.assertRaises(MaskBakeError):
                bake_native_run(
                    run_root,
                    output_name="perception_targets_native_v3",
                    policy_path=policy_path,
                )
            self.assertFalse(
                (run_root / "derived/perception_targets_native_v3").exists()
            )

    def test_actor_negative_policy_keeps_actor_pixels_valid_non_drivable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = self.make_run(root)
            semantic_path = run_root / "frames/semantic/front/000000.png"
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            semantic[400:500, 500:600, :3] = (2, 255, 236)
            write_png(semantic_path, semantic)
            policy = {
                "status": "frozen",
                "policy_id": "test_actor_negative",
                "audit_constraints": {"zero_pixel_classes": ["blue_lane"]},
                "targets": {
                    "lane": {"positive_classes": ["white_lane", "yellow_lane"]},
                    "drivable": {
                        "positive_classes": [
                            "asphalt",
                            "white_lane",
                            "yellow_lane",
                            "stopline",
                        ],
                        "ignore_classes": [],
                    },
                    "stopline": {
                        "positive_classes": ["stopline"],
                        "valid_views": ["front", "left", "right"],
                    },
                    "road_marking": {
                        "class_values": {
                            "background": 0,
                            "white_lane": 1,
                            "yellow_lane": 2,
                            "stopline": 3,
                        },
                        "valid_views": ["front", "left", "right"],
                    },
                    "actor_occupancy_gt": {"positive_classes": ["obstacle"]},
                },
            }
            policy_path = root / "policy_actor_negative.json"
            policy_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")

            bake_native_run(
                run_root,
                output_name="perception_targets_native_actor_negative",
                policy_path=policy_path,
            )
            cache_summary = build_model_cache(
                run_root,
                native_output_name="perception_targets_native_actor_negative",
                output_name="twinlite_actor_negative",
            )
            native_root = (
                run_root / "derived/perception_targets_native_actor_negative"
            )
            occupancy = cv2.imread(
                str(native_root / "actor_occupancy_gt_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            drivable = cv2.imread(
                str(native_root / "drivable_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            ignore = cv2.imread(
                str(native_root / "drivable_ignore_masks/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertGreater(int(occupancy.sum()), 0)
            self.assertTrue(np.all(drivable[occupancy == 1] == 0))
            self.assertEqual(int(ignore.sum()), 0)

            cache_root = run_root / "derived/twinlite_actor_negative"
            geometric = cv2.imread(
                str(cache_root / "valid_masks/geometric/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            drivable_valid = cv2.imread(
                str(cache_root / "valid_masks/drivable/front/000000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertTrue(np.array_equal(drivable_valid, geometric))
            self.assertEqual(
                cache_summary["policy_id"], "test_actor_negative"
            )


if __name__ == "__main__":
    unittest.main()
