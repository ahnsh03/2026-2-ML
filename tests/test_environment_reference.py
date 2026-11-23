import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.bridge import sha256_file
from quality_estimation.environment_reference import fit_reference, load_reference, with_shortfall
from quality_estimation.review_queue import choose_for_review


class EnvironmentReferenceTests(unittest.TestCase):
    def test_training_quintiles_and_sparse_fallback(self):
        y = np.r_[np.full(20, .8), np.full(25, .3), [.1]]
        g = np.repeat([0, 1, 2], [20, 25, 1])
        spec = fit_reference(y, g, 4)
        self.assertAlmostEqual(spec["clusters"][0]["q20"], .8)
        self.assertAlmostEqual(spec["clusters"][1]["q20"], .3)
        self.assertEqual([group["source"] for group in spec["clusters"]], ["cluster", "cluster", "global", "global"])
        self.assertEqual(spec["clusters"][2]["q20"], spec["global_q20"])
        self.assertEqual(spec["clusters"][3]["rows"], 0)

    def test_relative_queue_can_choose_a_high_iou_scene_and_break_ties_by_id(self):
        spec = {"clusters": [{"q20": .8}, {"q20": .2}]}
        rows = [dict(sample_id=s, estimated_iou=iou, environment_cluster=g, pixel_sha256=s*64)
                for s, iou, g in [("b", .5, 0), ("a", .5, 0), ("c", .1, 1)]]
        enriched = [with_shortfall(row, spec) for row in rows]
        chosen, _ = choose_for_review(enriched, 1, audit_fraction=0, priority_policy="environment_relative")
        self.assertEqual((chosen[0][0]["sample_id"], chosen[0][1]), ("a", "largest_environment_shortfall"))
        self.assertEqual(choose_for_review(enriched, 1, audit_fraction=0)[0][0][0]["sample_id"], "c")
        with self.assertRaisesRegex(ValueError, "shortfall"):
            choose_for_review(rows, 1, priority_policy="environment_relative")
        with self.assertRaisesRegex(ValueError, "cluster"):
            with_shortfall(dict(rows[0], environment_cluster=4), spec)

    def test_reference_is_bound_to_model_checkpoint_and_centers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "reference.json"
            reference.write_text(json.dumps(fit_reference(np.full(20, .5), np.zeros(20, int), 2)))
            binding = dict(model_run_sha256="a"*64, checkpoint_sha256="b"*64, environment_centers_sha256="c"*64)
            run = directory / "run.json"
            run.write_text(json.dumps(dict(binding, status="complete", artifact_sha256={"reference.json": sha256_file(reference)})))
            (directory / "_SUCCESS").write_text(sha256_file(run))
            self.assertEqual(load_reference(directory, binding)["training_rows"], 20)
            for key in binding:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    load_reference(directory, dict(binding, **{key: "0"*64}))
            reference.write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_reference(directory, binding)
