import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/scene_cycle_20261128/decide_offline_adoption.py'
spec=importlib.util.spec_from_file_location('confirmation_adoption_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ConfirmationAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.parent={'checkpoint_sha256':'parent'};self.child={'checkpoint_sha256':'child'}
        self.freeze=dict(perception_models=dict(r1=dict(perception_identity=self.parent),selected=dict(perception_identity=self.child)),
            primary_review_name='primary',parent_review_name='parent_r1',
            review_candidates=[dict(name=name,perception_identity=identity,bundle=name,bundle_manifest_sha256=name+'_sha')for name,identity in [('primary',self.child),('parent_r1',self.parent)]])
        self.neural=dict(selected_is_existing_parent=False,new_child_passes=True,failed_gates=[])
        self.review=dict(results={name:dict(perception_identity=identity,bundle_manifest_sha256=name+'_sha',review_gate=dict(passes=True,failed_gates=[]),quality_handoff_gate=dict(passes=True,failed_gates=[]))for name,identity in [('primary',self.child),('parent_r1',self.parent)]})

    def test_accepted_child_requires_its_exact_review_pair(self):
        result=module.decide(self.freeze,self.neural,self.review)
        self.assertEqual(result['perception_version'],'m-v6-r2-20261128')
        self.assertEqual(result['review_name'],'primary');self.assertTrue(result['offline_review_criteria_pass'])

    def test_good_primary_review_cannot_rescue_failed_child(self):
        self.neural.update(new_child_passes=False,failed_gates=['marked_macro_gain'])
        self.review['results']['parent_r1']['review_gate']=dict(passes=False,failed_gates=['insufficient_near_error_support'])
        result=module.decide(self.freeze,self.neural,self.review)
        self.assertEqual(result['review_name'],'parent_r1');self.assertFalse(result['offline_review_criteria_pass'])
        self.assertEqual(result['perception_identity'],self.parent)

    def test_review_failure_does_not_undo_a_valid_perception_only_update(self):
        self.review['results']['primary']['review_gate']=dict(passes=False,failed_gates=['utility_recall_vs_random'])
        result=module.decide(self.freeze,self.neural,self.review)
        self.assertEqual(result['perception_decision'],'accept_frozen_child');self.assertFalse(result['offline_review_criteria_pass'])
        self.assertFalse(result['automatic_training_admission'])

    def test_coverage_guard_is_required_even_with_high_review_recall(self):
        self.review['results']['primary']['quality_handoff_gate']=dict(passes=False,failed_gates=['quality_interval_coverage'])
        self.assertFalse(module.decide(self.freeze,self.neural,self.review)['offline_review_criteria_pass'])

    def test_parent_fallback_cannot_silently_use_child_quality_weights(self):
        self.neural.update(new_child_passes=False,failed_gates=['marked_macro_gain'])
        self.review['results']['parent_r1']['perception_identity']=copy.deepcopy(self.child)
        with self.assertRaises(ValueError):module.decide(self.freeze,self.neural,self.review)

    def test_a_better_diagnostic_family_does_not_become_the_primary(self):
        self.review['results']['GP_only']=dict(test_score=1.)
        self.review['results']['primary']['review_gate']=dict(passes=False,failed_gates=['utility_recall_vs_random'])
        result=module.decide(self.freeze,self.neural,self.review)
        self.assertEqual(result['review_name'],'primary');self.assertFalse(result['offline_review_criteria_pass'])

    def test_no_new_child_keeps_the_prospectively_frozen_parent_policy(self):
        self.freeze['perception_models']['selected']['perception_identity']=self.parent
        self.neural.update(selected_is_existing_parent=True,new_child_passes=False)
        self.freeze['review_candidates'][0]['perception_identity']=self.parent
        self.review['results']['primary']['perception_identity']=self.parent
        result=module.decide(self.freeze,self.neural,self.review)
        self.assertEqual(result['review_name'],'parent_r1')
        self.assertEqual(result['perception_version'],'m-v6-r1-20261128')

    def test_same_neural_model_does_not_allow_a_changed_policy_bundle(self):
        self.review['results']['primary']['bundle_manifest_sha256']='another_policy'
        with self.assertRaises(ValueError):module.decide(self.freeze,self.neural,self.review)


if __name__=='__main__':unittest.main()
