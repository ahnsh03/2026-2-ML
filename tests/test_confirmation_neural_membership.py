import importlib.util
from pathlib import Path
import unittest
import numpy as np

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

    def test_neural_band_report_preserves_missing_output_and_empty_gt(self):
        counts=np.zeros((3,9,3),int);counts[0,6]=[80,0,20];counts[1,6]=[0,60,0];counts[2,6]=[0,0,100]
        result=module.band_summary(counts)['near']
        self.assertEqual(result['gt_supported_rows'],2)
        self.assertEqual(result['supported_fn_frames'],1);self.assertEqual(result['supported_fp_frames'],1)
        self.assertEqual(result['empty_gt_fp_frames'],1);self.assertEqual(result['missing_output_fn_frames'],1)
        self.assertAlmostEqual(result['pooled_precision'],80/140)
        self.assertAlmostEqual(result['pooled_recall'],80/200)
        self.assertIsNone(module.band_summary(np.zeros((2,9,3),int))['near']['pooled_precision'])


if __name__=='__main__':unittest.main()
