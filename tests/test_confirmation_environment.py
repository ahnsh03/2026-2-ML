import importlib.util
from pathlib import Path
import sys
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/scene_cycle_20261128/assess_environment_context.py'
original_path=sys.path.copy();sys.path.insert(0,str(path.parent))
try:
    spec=importlib.util.spec_from_file_location('confirmation_environment_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
finally:sys.path[:]=original_path


class ConfirmationEnvironmentTests(unittest.TestCase):
    def test_class_imbalance_does_not_inflate_balanced_accuracy(self):
        result=module.classification([0,0,0,1],[0,0,0,0],3)
        self.assertEqual(result['accuracy'],.75)
        self.assertEqual(result['balanced_accuracy'],.5)
        self.assertEqual(result['supported_classes'],[0,1])
        self.assertEqual(result['confusion_matrix'],[[3,0,0],[1,0,0],[0,0,0]])

    def test_fixed_sdk_clock_mapping_matches_training(self):
        self.assertEqual(module.lighting([5,6,7,8,13,17,18,19,20]).tolist(),[2,1,1,0,0,0,1,1,2])

    def test_invalid_classes_are_not_silently_dropped_from_confusion(self):
        with self.assertRaises(ValueError):module.classification([0,1],[0,3],3)

    def test_empty_or_misaligned_population_is_not_zero_accuracy(self):
        for truth,pred in [([],[]),([0],[0,1])]:
            with self.assertRaises(ValueError):module.classification(truth,pred,3)


if __name__=='__main__':unittest.main()
