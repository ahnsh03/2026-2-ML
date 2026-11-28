import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/scene_cycle_20261128/assess_perception_confirmation.py'
spec=importlib.util.spec_from_file_location('confirmation_neural_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ConfirmationNeuralMembershipTests(unittest.TestCase):
    source=[dict(sample_id='frame_a',pixel_sha256='a'),dict(sample_id='frame_b',pixel_sha256='b')]

    def rows(self):return [dict(row,adaptation_role='confirmation')for row in self.source]

    def test_exact_registered_population_is_accepted(self):
        module.validate_source_membership(self.rows(),self.source)

    def test_order_or_pixel_change_does_not_reuse_confirmation_status(self):
        for field,value in [('sample_id','other'),('pixel_sha256','other'),('adaptation_role','development')]:
            rows=self.rows();rows[0][field]=value
            with self.assertRaises(ValueError):module.validate_source_membership(rows,self.source)
        with self.assertRaises(ValueError):module.validate_source_membership(self.rows()[::-1],self.source)

    def test_subset_is_not_the_frozen_test(self):
        with self.assertRaises(ValueError):module.validate_source_membership(self.rows()[:1],self.source)


if __name__=='__main__':unittest.main()
