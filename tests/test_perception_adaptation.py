"""Protect target parity, reserved evaluation roles and masked gradients."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from quality_estimation.native import NativeSamples
from quality_estimation.perception_data import (NativePerceptionSamples, validate_adaptation_roles,
                                                validate_adaptation_selection)
from quality_estimation.freeze_perception import create_perception_freeze
from common.bridge import DEFAULT_CHECKPOINT_SHA256
from data_collection.mask_baker import OFFICIAL_PALETTE_RGB
from twinlite_morai import MaskedTwinLiteLoss, ROAD_MARKING_TASK


class AdaptationTests(unittest.TestCase):
    def test_training_arm_matches_admission_method_budget_seed_and_order(self):
        selection=dict(method='Random',budget=.1,seed=12,rank=1,
                       inference_run_sha256='a'*64,model_run_sha256='b'*64)
        rows=[dict(selection=selection)]
        validate_adaptation_selection(rows,'Random',12,.1)
        for method in ('FrozenQualityRank','Confidence','Entropy'):
            validate_adaptation_selection([dict(selection=dict(selection,method=method,seed=None))],method,12,.1)
            with self.assertRaisesRegex(ValueError,'method differs'):
                validate_adaptation_selection(rows,method,12,.1)
        for seed,budget,pattern in ((13,.1,'seed differs'),(12,.2,'budget differs')):
            with self.assertRaisesRegex(ValueError,pattern):
                validate_adaptation_selection(rows,'Random',seed,budget)
        with self.assertRaisesRegex(ValueError,'ranks'):
            validate_adaptation_selection(rows+rows,'Random',12,.1)
        mixed=rows+[dict(selection=dict(selection,rank=2,inference_run_sha256='c'*64))]
        with self.assertRaisesRegex(ValueError,'share one valid'):
            validate_adaptation_selection(mixed,'Random',12,.1)
        validate_adaptation_selection([],'ReplayOnly',12,.1)
        with self.assertRaisesRegex(ValueError,'cannot contain'):
            validate_adaptation_selection(rows,'ReplayOnly',12,.1)

    def test_final_registry_rejects_reused_training_parent_and_keeps_all_runs(self):
        run=dict(train_parent_groups=['candidate'],protocol_sha256='protocol',arm='Random',seed=12,
                 steps=240,selection='fixed step 240')
        membership=[dict(parent_session_id='replay')]
        with patch('quality_estimation.freeze_perception.resolve_checkpoint',return_value=Path('f0.pt')), \
             patch('quality_estimation.freeze_perception.sha256_file',return_value=DEFAULT_CHECKPOINT_SHA256), \
             patch('quality_estimation.freeze_perception.load_adaptation',return_value=(Path('f1.pt'),run,membership)):
            for reused in ('candidate','replay'):
                with self.assertRaisesRegex(ValueError,'used for adaptation'):
                    create_perception_freeze([reused],['run_a'])
            frozen=create_perception_freeze(['final'],['run_a','run_b'])
            self.assertEqual(set(frozen['perception_checkpoints']),{'f0','run_a','run_b'})
            self.assertFalse(frozen['final_test_accessed_before_freeze'])
            self.assertEqual(frozen['adaptation_provenance']['run_b']['steps'],240)
            with self.assertRaisesRegex(ValueError,'duplicate checkpoint'):
                create_perception_freeze(['final'],['run_a','run_a'])

    def test_native_two_head_targets_keep_actor_negative_and_marking_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);semantic=np.empty((12,30,3),np.uint8)
            for i,name in enumerate(('white_lane','yellow_lane','stopline','asphalt','vehicle')):
                semantic[:,i*6:(i+1)*6]=OFFICIAL_PALETTE_RGB[name][::-1]
            cv2.imwrite(str(root/'semantic.png'),semantic)
            cv2.imwrite(str(root/'rgb.png'),np.full_like(semantic,100))
            row=dict(sample_id='one',view='front',image='rgb.png',semantic='semantic.png')
            quality=NativeSamples(root,[row])[0];training=NativePerceptionSamples(root,[row])[0]
            np.testing.assert_array_equal(training['targets']['road_marking'],quality['target'])
            x,y,_,_=training['geometry'].pad_ltrb;h,w=training['geometry'].resized_hw
            marking=training['targets']['road_marking'][y:y+h,x:x+w]
            drivable=training['targets']['drivable'][y:y+h,x:x+w]
            self.assertEqual(set(np.unique(marking)),{0,1,2,3})
            self.assertTrue(np.all(drivable[:,int(.82*w):]==0))
            self.assertTrue(np.all(drivable[:,:int(.78*w)]==1))
            self.assertTrue(np.all(training['valid_mask'][y:y+h,x:x+w]))

    def test_training_roles_reject_final_dev_and_unknown_replay(self):
        new=dict(sample_id='new',parent_session_id='candidate',preregistered_quality_role='candidate_pool')
        replay=dict(sample_id='old',parent_session_id='old',source_role='f0_train')
        validate_adaptation_roles([new],[replay])
        for role in ('meta_train','meta_dev','final_test'):
            with self.assertRaisesRegex(ValueError,'candidate pool'):
                validate_adaptation_roles([dict(new,preregistered_quality_role=role)],[replay])
        with self.assertRaisesRegex(ValueError,'known f0 training'):
            validate_adaptation_roles([new],[dict(replay,source_role='unknown')])

    def test_replay_only_control_cannot_silently_include_new_labels(self):
        replay=dict(sample_id='old',parent_session_id='old',source_role='f0_train')
        validate_adaptation_roles([],[replay],replay_only=True)
        with self.assertRaisesRegex(ValueError,'replay-only control'):
            validate_adaptation_roles([],[replay])
        new=dict(sample_id='new',parent_session_id='new',preregistered_quality_role='candidate_pool')
        with self.assertRaisesRegex(ValueError,'replay-only control'):
            validate_adaptation_roles([new],[replay],replay_only=True)

    def test_two_head_loss_has_finite_gradients_and_ignores_padding(self):
        drivable=torch.randn(2,2,4,6,requires_grad=True)
        marking=torch.randn(2,4,4,6,requires_grad=True)
        targets=dict(drivable=torch.zeros(2,4,6,dtype=torch.long),road_marking=torch.zeros(2,4,6,dtype=torch.long))
        targets['road_marking'][:,1:3,2:4]=1
        valid=torch.zeros(2,4,6,dtype=torch.bool);valid[:,1:3]=True
        losses=MaskedTwinLiteLoss(task_spec=ROAD_MARKING_TASK)((drivable,marking),targets,valid)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        for tensor in (drivable,marking):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertTrue(torch.equal(tensor.grad[:,:,0],torch.zeros_like(tensor.grad[:,:,0])))
            self.assertGreater(float(tensor.grad[:,:,1:3].abs().sum()),0)


if __name__=='__main__':
    unittest.main()
