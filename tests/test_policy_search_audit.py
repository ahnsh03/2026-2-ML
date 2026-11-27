"""The Dev audit must not silently discard better controls or adopt test changes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

scripts = Path(__file__).resolve().parents[1] / 'scripts/experiments/environment_cycle_20261127'
if str(scripts) not in sys.path:
    sys.path.insert(0, str(scripts))
from audit_policy_search import audit_policy_search, write_audit


class PolicySearchAuditTest(unittest.TestCase):
    def fixture(self):
        def trial(loss, floor=0):
            return dict(loss=loss, parameters=dict(balanced_floor=floor, uncertainty_weight=0),
                        reference=dict(absolute_threshold=.7, groups={'0': dict(threshold=.6)}))
        trials = {'0': trial(.4), '1': trial(.39), '2': trial(.35, .5)}
        study = dict(bo_indices=[0, 1], bo_losses=[.4, .39], random_indices=[0, 2],
                     random_losses=[.4, .35], trials=trials, selected=trials['1'])
        return dict(common_objective=.4, positive_support_passes=True,
                    runtime_policy=dict(grouping='global', audit_fraction=.1, score_tie_break='standardized_margin'),
                    studies={'rgb_weather': study}, perception_identity={'kind': 'test'})

    def test_better_random_control_is_retained_but_not_adopted(self):
        policy = self.fixture(); before = copy.deepcopy(policy)
        result = audit_policy_search(policy)
        candidate = result['proposed_candidate']
        self.assertEqual(candidate['trial_index'], 2)
        self.assertEqual(candidate['runtime_policy']['balanced_floor'], .5)
        self.assertFalse(candidate['runtime_policy']['environment_thresholds_adopted'])
        self.assertTrue(result['requires_new_confirmation'])
        self.assertFalse(result['fresh_assessment_read'])
        self.assertFalse(result['inference_bundle_created'])
        self.assertEqual(policy, before)

    def test_simulator_oracle_is_not_a_deployable_candidate(self):
        policy = self.fixture(); policy['studies'] = {'sim_oracle': policy['studies']['rgb_weather']}
        self.assertIsNone(audit_policy_search(policy)['proposed_candidate'])

    def test_insufficient_positive_support_prevents_candidate(self):
        policy = self.fixture(); policy['positive_support_passes'] = False
        self.assertIsNone(audit_policy_search(policy)['proposed_candidate'])

    def test_no_minimum_gain_keeps_no_candidate(self):
        policy = self.fixture(); policy['common_objective'] = .355
        self.assertIsNone(audit_policy_search(policy)['proposed_candidate'])

    def test_equal_losses_prefer_less_balance_then_stable_index(self):
        policy = self.fixture(); study = policy['studies']['rgb_weather']
        study['trials']['2']['loss'] = .39; study['random_losses'][1] = .39
        self.assertEqual(audit_policy_search(policy)['proposed_candidate']['trial_index'], 1)

    def test_rejects_corrupt_or_unequal_search_history(self):
        for mode in ['loss', 'nan', 'budget', 'duplicate', 'missing', 'selected']:
            with self.subTest(mode=mode):
                policy = self.fixture(); study = policy['studies']['rgb_weather']
                if mode == 'loss': study['random_losses'][1] = .2
                elif mode == 'nan': study['random_losses'][1] = float('nan')
                elif mode == 'budget': study['random_indices'].pop()
                elif mode == 'duplicate': study['random_indices'] = [0, 0]
                elif mode == 'missing': del study['trials']['2']
                else: study['selected'] = dict(loss=.1)
                with self.assertRaises(ValueError): audit_policy_search(policy)

    def test_output_is_exclusive_and_source_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'policy.json'; output = Path(tmp)/'audit.json'
            source.write_text(json.dumps(self.fixture())); original = source.read_bytes()
            write_audit(source, output); expected = output.read_bytes()
            with self.assertRaises(FileExistsError): write_audit(source, output)
            with self.assertRaises(FileExistsError): write_audit(source, source)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), expected)


if __name__ == '__main__':
    unittest.main()
