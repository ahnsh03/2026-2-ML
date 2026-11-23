from itertools import permutations, product
import unittest

from quality_estimation.evaluate_cluster_pairs import evaluate


class ClusterPairTests(unittest.TestCase):
    def test_null_equals_all_within_pose_permutations(self):
        conditions = [(0, 13), (1, 13), (0, 23)]
        centers = [(0, 0, 1), (2, 0, 2)]
        rows = [dict(pose=p, weather=w, hour=h, assignment=c)
                for p, labels in enumerate(centers) for (w, h), c in zip(conditions, labels)]
        observed = evaluate(rows, "assignment")
        self.assertEqual([v["pairs"] for v in observed.values()], [3, 2, 2])
        self.assertEqual([v["observed"] for v in observed.values()], [1, 1, 1])
        alternatives = []
        for a, b in product(permutations(centers[0]), permutations(centers[1])):
            alternatives.append(evaluate([dict(r, assignment=c) for r, c in zip(rows, a + b)], "assignment"))
        for name, value in observed.items():
            expected = sum(result[name]["observed"] for result in alternatives) / len(alternatives)
            self.assertAlmostEqual(value["shuffled_expected"], expected)

    def test_no_eligible_pair_and_missing_assignment(self):
        one = dict(pose="A", weather="sunny", hour="13", center="0")
        self.assertTrue(all(r["pairs"] == 0 for r in evaluate([one], "center").values()))
        with self.assertRaises(ValueError):
            evaluate([one], "absent")
