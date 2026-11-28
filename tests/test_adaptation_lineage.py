import copy
import unittest
from quality_estimation.adaptation_lineage import validate_cumulative_membership


class AdaptationLineageTest(unittest.TestCase):
    def fixture(self):
        parent=[dict(sample_id='old',pixel_sha256='a'),dict(sample_id='admitted',pixel_sha256='b')]
        replay=[dict(parent[0],source_role='f0_train'),dict(parent[1],source_role='previously_admitted')]
        new=[dict(sample_id='new',pixel_sha256='c')]
        return new,replay,parent

    def test_preserves_original_and_previously_admitted_members(self):
        new,replay,parent=self.fixture();result=validate_cumulative_membership(new,replay,parent)
        self.assertEqual(result['cumulative_training_rows'],3)
        self.assertTrue(result['parent_membership_preserved'])

    def test_rejects_dropped_changed_duplicate_or_unverified_parent(self):
        for mode in ['dropped','changed','duplicate','role']:
            with self.subTest(mode=mode):
                new,replay,parent=self.fixture()
                if mode=='dropped':replay.pop()
                elif mode=='changed':replay[0]['pixel_sha256']='changed'
                elif mode=='duplicate':replay.append(copy.deepcopy(replay[0]))
                else:replay[1]['source_role']='unverified'
                with self.assertRaises(ValueError):validate_cumulative_membership(new,replay,parent)

    def test_rejects_previously_exposed_new_pixels_and_ids(self):
        for field in ['sample_id','pixel_sha256']:
            with self.subTest(field=field):
                new,replay,parent=self.fixture();new[0][field]=parent[0][field]
                with self.assertRaises(ValueError):validate_cumulative_membership(new,replay,parent)
