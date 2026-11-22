import unittest
from quality_estimation.review_queue import choose_for_review


class ReviewQueueTests(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=str(i),pixel_sha256=format(i,'064x'),estimated_iou=i/10) for i in range(10)]

    def test_priority_budget_and_remaining_audit_are_reproducible(self):
        a,p=choose_for_review(self.rows(),5,.4,123)
        self.assertEqual([r['sample_id'] for r,reason in a[:3]],['0','1','2'])
        self.assertEqual([reason for r,reason in a[3:]],['random_remainder_audit']*2)
        self.assertEqual(len({r['sample_id'] for r,reason in a}),5)
        self.assertEqual(a,choose_for_review(list(reversed(self.rows())),5,.4,123)[0])
        self.assertEqual((p['priority_count'],p['audit_count']),(3,2))

    def test_exact_pixel_duplicates_spend_only_one_budget(self):
        rows=self.rows();rows.append(dict(rows[0],sample_id='z'))
        a,p=choose_for_review(rows,10,0)
        self.assertEqual(p['pixel_unique_rows'],10)
        self.assertNotIn('z',[r['sample_id'] for r,reason in a])
        with self.assertRaisesRegex(ValueError,'exceeds'):choose_for_review(rows,11)

    def test_bad_scores_and_duplicate_ids_are_rejected(self):
        for bad in [float('nan'),float('inf'),-.1,1.1]:
            rows=self.rows();rows[0]['estimated_iou']=bad
            with self.assertRaisesRegex(ValueError,'IoU'):choose_for_review(rows,2)
        with self.assertRaisesRegex(ValueError,'sample ID'):choose_for_review(self.rows()+self.rows()[:1],2)

    def test_audit_only_and_priority_only_policies(self):
        self.assertTrue(all(reason=='random_remainder_audit' for r,reason in choose_for_review(self.rows(),3,1)[0]))
        self.assertTrue(all(reason=='low_estimated_iou' for r,reason in choose_for_review(self.rows(),3,0)[0]))

    def test_invalid_budget_or_fraction_is_rejected(self):
        for budget,fraction in [(0,.2),(2,-.1),(2,1.1),(2,float('nan'))]:
            with self.assertRaises(ValueError):choose_for_review(self.rows(),budget,fraction)

    def test_cluster_rounds_rank_by_quality_and_keep_random_audit_separate(self):
        rows = self.rows()
        for row, group in zip(rows, [2, 2, 3, 3, 3, 3, 5, 5, 5, 5]):
            row["environment_cluster"] = group
        queue, _ = choose_for_review(rows, 5, .2, 123, priority_policy="cluster_round_robin")
        self.assertEqual([row["sample_id"] for row, _ in queue[:4]], ["0", "2", "6", "1"])
        self.assertEqual(queue[-1][1], "random_remainder_audit")
        self.assertEqual(len({row["sample_id"] for row, _ in queue}), 5)
        with self.assertRaisesRegex(ValueError, "environment cluster"):
            choose_for_review(self.rows(), 3, priority_policy="cluster_round_robin")

    def test_gp_exploration_uses_remaining_uncertainty_and_separate_audit(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row["prediction_std"] = float(i)
        ranked = sorted(rows, key=lambda r: (r["estimated_iou"], r["sample_id"]))
        expected_risk = [row["sample_id"] for row in ranked[:4]]
        expected_exploration = sorted(ranked[4:], key=lambda r: (-r["prediction_std"], r["sample_id"]))[:2]
        queue, policy = choose_for_review(rows, 8, .25, 77, priority_policy="quality_uncertainty")
        self.assertEqual([row["sample_id"] for row, _ in queue[:4]], expected_risk)
        self.assertEqual(queue[4:6], [(r, "high_prediction_std") for r in expected_exploration])
        self.assertEqual(policy["uncertainty_count"], 2)
        self.assertEqual(policy["audit_count"], 2)
        self.assertEqual(len({r["sample_id"] for r, _ in queue}), 8)
        rows[0]["prediction_std"] = float("nan")
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            choose_for_review(rows, 8, priority_policy="quality_uncertainty")
