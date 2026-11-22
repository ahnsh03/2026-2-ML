import unittest

from quality_estimation.admit_review import admission_rows
from quality_estimation.perception_data import validate_adaptation_selection


class EnvironmentAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.raw = []
        self.pred = []
        for i in range(4):
            sid = str(i)
            rgb = str(i)*64
            self.raw.append(dict(sample_id=sid, view="front", run_id="pool", frame_id=i, image="unopened.png",
                                 semantic="unopened_semantic.png", preregistered_quality_role="candidate_pool",
                                 expected_sha256={"rgb": rgb, "semantic": "c"*64}))
            self.pred.append(dict(sample_id=sid, rgb_sha256=rgb, pixel_sha256=rgb,
                                  estimated_iou=.1*i, environment_cluster=i%2))
        self.queue = [dict(queue_index=i+1, sample_id=str(i), purpose="candidate_pool", expected_sha256={"rgb": str(i)*64},
                           pixel_sha256=str(i)*64, estimated_iou=.1*i, environment_cluster=i,
                           review_reason=reason) for i, reason in enumerate(("environment_round_robin", "random_remainder_audit"))]
        self.evidence = dict(pool_rows=4, review_seed=8, environment_centers_sha256="b"*64,
                             queue_run_sha256="c"*64, inference_run_sha256="d"*64, model_run_sha256="e"*64)

    def test_annotation_paths_are_joined_by_identity_without_reading_labels(self):
        result = admission_rows(self.raw[::-1], self.pred, self.queue, .5, self.evidence)
        self.assertEqual([r["sample_id"] for r in result], ["0", "1"])
        validate_adaptation_selection(result, "EnvironmentAudit", 24, .5)
        result[0]["selection"]["review_seed"] = 99
        with self.assertRaisesRegex(ValueError, "review seed"):
            validate_adaptation_selection(result, "EnvironmentAudit", 24, .5)

    def test_rejects_a_held_out_frame_even_when_it_is_not_selected(self):
        self.raw[-1]["preregistered_quality_role"] = "final_test"
        with self.assertRaisesRegex(ValueError, "reserved"):
            admission_rows(self.raw, self.pred, self.queue, .5, self.evidence)

    def test_rejects_swapped_pixels_and_mismatched_budget(self):
        self.queue[0]["pixel_sha256"] = self.pred[-1]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "frozen estimate"):
            admission_rows(self.raw, self.pred, self.queue, .5, self.evidence)
        with self.assertRaisesRegex(ValueError, "budget"):
            admission_rows(self.raw, self.pred, self.queue, .25, self.evidence)
