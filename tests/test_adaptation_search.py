import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from quality_estimation.search_adaptation import development_score, next_index, search, search_grid


def evaluation(directory, role):
    directory.mkdir()
    context = dict(role=role, manifest_sha256=role, checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                   target_policy_sha256="standard", adaptation_run_sha256=None, freeze_sha256=None)
    summary = directory / "evaluation.json"
    summary.write_text(json.dumps(dict(context, metrics={"lane": {"mean_frame_iou": .6}})))
    frames = directory / "per_frame.jsonl"
    frames.write_text(json.dumps(dict(sample_id=role, pixel_sha256=role)))
    record = directory / "run.json"
    record.write_text(json.dumps(dict(context, status="complete", artifact_sha256={f.name: sha256_file(f) for f in (summary, frames)})))
    (directory / "_SUCCESS").write_text(sha256_file(record))


class AdaptationSearchTests(unittest.TestCase):
    def setUp(self):
        self.design = dict(lr=[1e-6, 2e-6, 4e-6], replay=[2], lam=[0, 1], batch=8, steps=240,
                           initial=[[2e-6, 2, 0]], budget=5, max_loss=.005, random_search_seed=113)

    def test_missing_trial_returns_same_initial_step_and_validates_roles(self):
        with TemporaryDirectory() as temp:
            runs = Path(temp)
            evaluation(runs / "f0_development", "meta_dev")
            evaluation(runs / "f0_retention", "diagnostic_only")
            a, b = (search(runs, self.design, mode, 12) for mode in ("bo", "random"))
            self.assertEqual(a, b)
            self.assertEqual(a["settings"], (2e-6, 2, 0))
            self.assertEqual(len(a["required_runs"]), 3)
            final = runs / "final"
            evaluation(final, "final_test")
            with self.assertRaisesRegex(ValueError, "development evaluation"):
                development_score(final, checkpoint=DEFAULT_CHECKPOINT_SHA256, adaptation=None, role="meta_dev")
            (runs / "f0_retention" / "evaluation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                search(runs, self.design, "bo", 12)

    def test_grid_scaling_and_bo_for_feasible_and_infeasible_histories(self):
        grid, initial, coords = search_grid(self.design)
        np.testing.assert_allclose(coords[initial[0]], [.5, 0, 0])
        self.assertEqual(len(grid), 6)
        for loss in (.001, .02):
            history = [dict(index=initial[0], new_gain=.02, retention_loss=loss)]
            for _ in range(len(grid)-1):
                proposal = next_index(coords, initial, history, self.design, "bo")
                self.assertNotIn(proposal, [r["index"] for r in history])
                history.append(dict(index=proposal, new_gain=.01, retention_loss=loss))
            self.assertEqual(len({r["index"] for r in history}), 6)
