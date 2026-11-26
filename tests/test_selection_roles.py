import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/selection_cycle_20261126/prepare.py'
spec=importlib.util.spec_from_file_location('selection_cycle_prepare_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class SelectionRoleTest(unittest.TestCase):
    def test_opened_shift_never_enters_calibration_via_shared_prefix(self):
        self.assertEqual(module.source_role('drive_calibration_shift_drive26_r02'),'heldout_diagnostic')
        self.assertEqual(module.source_role('drive_calibration_drive26_r01'),'calibration')
        self.assertEqual(module.source_role('drive_confirmation_test_confirm26_c04'),'heldout_diagnostic')
    def test_existing_quality_fit_exposure_takes_precedence(self):
        self.assertEqual(module.source_role('drive_development_example',True),'quality_fit')

if __name__=='__main__':unittest.main()
