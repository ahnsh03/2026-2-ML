import unittest
from quality_estimation.retrain_streaming import BalancedBatches

class StreamingBatchesTest(unittest.TestCase):
    def test_every_pool_member_seen_and_replay_draws_match_between_arms(self):
        left=list(BalancedBatches(8,12,2,2,8,23));right=list(BalancedBatches(10,12,2,2,8,23))
        self.assertEqual(set(i for b in left[:4] for i in b[:2]),set(range(8)))
        self.assertEqual([[i-8 for i in b[2:]] for b in left],[[i-10 for i in b[2:]] for b in right])
    def test_full_original_only_control_supports_empty_new_pool(self):
        batches=list(BalancedBatches(0,11,0,4,7,42))
        self.assertTrue(all(len(b)==4 for b in batches))
        flat=[i for b in batches for i in b]
        self.assertEqual(set(flat[:11]),set(range(11)))
        self.assertEqual(set(flat[11:22]),set(range(11)))
        self.assertEqual(batches[3:],list(BalancedBatches(0,11,0,4,7,42,start=3)))
    def test_resume_matches_remaining_index_stream(self):
        full=list(BalancedBatches(7,11,2,2,12,98));resumed=list(BalancedBatches(7,11,2,2,12,98,start=5))
        self.assertEqual(full[5:],resumed)

if __name__=='__main__':unittest.main()
