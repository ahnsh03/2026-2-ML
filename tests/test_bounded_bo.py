import unittest
import numpy as np
from quality_estimation.bounded_bo import next_expected_improvement


class BoundedBOTests(unittest.TestCase):
    def test_initial_order_unique_proposals_and_exhaustion(self):
        coords = np.random.default_rng(4).uniform(size=(12, 4))
        indices = []
        losses = []
        for step in range(12):
            i = next_expected_improvement(coords, indices, losses, [7, 2, 9])
            self.assertNotIn(i, indices)
            if step < 3:
                self.assertEqual(i, [7, 2, 9][step])
            indices.append(i)
            losses.append(float(((coords[i] - 0.3) ** 2).sum()))
        with self.assertRaisesRegex(ValueError, "exhausted"):
            next_expected_improvement(coords, indices, losses, [7, 2, 9])


if __name__ == "__main__":
    unittest.main()
