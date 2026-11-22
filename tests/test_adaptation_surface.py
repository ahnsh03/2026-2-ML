import unittest

from quality_estimation.replay_adaptation_search import replay


class SurfaceReplayTests(unittest.TestCase):
    def setUp(self):
        self.surface = dict(search=dict(budgets=[1, 2], shared_initial=1, initializations=2,
                            first_seed=1, max_loss=.005, random_seed_offset=100, steps=2, batch=2),
                            grids={"tiny":dict(lr=[.001, .002], replay=[1], lam=[0])},
                            observations=[dict(settings=[x, 1, 0], new_gain=.2, retention_loss=.02)
                                          for x in (.001, .002)])

    def test_no_feasible_update_returns_incumbent(self):
        report = replay(self.surface, "tiny")
        for row in report:
            self.assertEqual(row["starts_with_feasible"], 0)
            self.assertEqual(row["mean_best_gain"], 0)
        self.surface["observations"][0]["retention_loss"] = .003
        report = replay(self.surface, "tiny")
        for row in report:
            if row["budget"] == 2:
                self.assertEqual(row["median_best_gain"], .2)
                self.assertEqual(row["starts_reaching_exact_best"], 2)

    def test_missing_or_repeated_trials_are_rejected(self):
        record = self.surface["observations"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay(self.surface, "tiny")
        self.surface["observations"].extend([record, record])
        with self.assertRaisesRegex(ValueError, "duplicate trial"):
            replay(self.surface, "tiny")
