import unittest

from quality_estimation.shadow_replay import make_batch_plan, validate_batch_plan


class ShadowReplayTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id=str(i)) for i in range(5)]
        self.predictions = [dict(sample_id=str(i), batch_rows=3 if i < 3 else 2,
                                 batch_seconds=.2 if i < 3 else .1) for i in range(5)]

    def test_variable_live_batches_are_preserved(self):
        plan = make_batch_plan(self.predictions, self.rows)
        self.assertEqual(validate_batch_plan(plan, self.rows), [3, 2])

    def test_reordered_or_duplicate_population_is_rejected(self):
        for rows in [self.rows[::-1], [self.rows[0]] * 5]:
            with self.assertRaises(ValueError):
                make_batch_plan(self.predictions, rows)

    def test_incomplete_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            make_batch_plan(self.predictions[:-1], self.rows[:-1])

    def test_inconsistent_batch_is_rejected(self):
        self.predictions[1]['batch_seconds'] = .3
        with self.assertRaises(ValueError):
            make_batch_plan(self.predictions, self.rows)

    def test_changed_batch_plan_is_rejected(self):
        plan = make_batch_plan(self.predictions, self.rows)
        for sizes in [[True] * 5, [0, 5], [4], [2.5, 2.5]]:
            plan['batch_sizes'] = sizes
            with self.assertRaises(ValueError):
                validate_batch_plan(plan, self.rows)

    def test_manifest_replacement_is_rejected(self):
        plan = make_batch_plan(self.predictions, self.rows)
        with self.assertRaises(ValueError):
            validate_batch_plan(plan, self.rows[::-1])
