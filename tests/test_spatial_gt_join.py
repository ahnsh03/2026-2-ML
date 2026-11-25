import unittest
import numpy as np
from quality_estimation.join_spatial_gt import join_candidates


class SpatialGTJoinTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.zeros((1, 9, 3), np.int64)
        self.counts[0, 0] = [30, 20, 0]
        self.counts[0, 4] = [10, 0, 50]
        self.counts[0, 6] = [60, 0, 0]
        self.pack = dict(
            ids=np.array(["frame"]),
            counts=self.counts,
            y=np.array([100 / 170], np.float32),
        )
        self.rows = [
            dict(
                sample_id="frame",
                image="front.png",
                semantic="semantic.png",
                expected_sha256=dict(rgb="a" * 64, semantic="b" * 64),
            )
        ]
        self.candidates = [
            dict(
                sample_id="frame",
                rgb_sha256="a" * 64,
                checkpoint_sha256="c" * 64,
                profile="roi9-front-v1",
                regions_sha256="d" * 64,
            )
        ]

    def join(self, candidates):
        return join_candidates(
            candidates,
            self.pack,
            self.rows,
            checkpoint_sha256="c" * 64,
            profile="roi9-front-v1",
            regions_sha256="d" * 64,
        )

    def test_gt_is_attached_after_selection_without_auto_admission(self):
        row = self.join(self.candidates)[0]
        self.assertAlmostEqual(row["true_lane_iou"], 100 / 170)
        self.assertEqual(
            row["spatial_diagnostic"], "near_recall_retained_far_recall_reduced"
        )
        self.assertFalse(row["admission_approved"])

    def test_checkpoint_and_rgb_mismatch_rejects_a_join(self):
        for field in ("checkpoint_sha256", "rgb_sha256"):
            modified = [dict(self.candidates[0], **{field: "e" * 64})]
            with self.assertRaises(ValueError):
                self.join(modified)


if __name__ == "__main__":
    unittest.main()
