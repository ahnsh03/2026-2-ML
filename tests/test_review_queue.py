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
