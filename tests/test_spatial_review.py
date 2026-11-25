import unittest
import numpy as np
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.review_spatial import select_window


class SpatialReviewTests(unittest.TestCase):
    def test_budget_is_exact_disjoint_and_does_not_use_attached_reference_fields(self):
        rng = np.random.default_rng(67)
        x = rng.normal(size=(12, 3))
        model = SpatialQualityGP().fit(x, rng.uniform(size=(12, 3)))
        rows = [
            dict(
                predicted_iou=0.1 + i * 0.05,
                risk_scores=dict(global_low=1 - i * 0.07, normal_far_only=i * 0.07),
            )
            for i in range(12)
        ]
        for policy in (
            "global_mean",
            "global_probability",
            "dual_probability",
            "dual_conditional",
        ):
            a = select_window(
                rows, list(map(str, range(12))), 5, policy, X=x, model=model
            )
            decorated = [
                dict(row, y=1.0, weather="unrelated", gt_path="absent") for row in rows
            ]
            b = select_window(
                decorated, list(map(str, range(12))), 5, policy, X=x, model=model
            )
            self.assertEqual(a, b)
            self.assertEqual(len(a), 5)
            self.assertEqual(len({i for i, _ in a}), 5)
            if policy.startswith("dual"):
                self.assertEqual(sum("diagnostic" in reason for _, reason in a), 1)


if __name__ == "__main__":
    unittest.main()
