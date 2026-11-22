import unittest
import torch
from quality_estimation.distillation import masked_replay_distillation,distillation_settings


class ReplayDistillationTests(unittest.TestCase):
    def heads(self,value=0.,gradient=True):
        return {name:torch.full((1,n,2,2),value,requires_grad=gradient)
                for name,n in [('drivable',2),('road_marking',4)]}

    def test_identical_predictions_have_zero_penalty(self):
        x=self.heads();loss=masked_replay_distillation(x,x,torch.ones(1,2,2,dtype=torch.bool))
        self.assertAlmostEqual(float(loss),0.,places=6)

    def test_teacher_is_detached_and_padding_receives_no_gradient(self):
        student=self.heads();teacher=self.heads()
        with torch.no_grad():
            for v in teacher.values():v[:,0]=2.
        valid=torch.zeros(1,2,2,dtype=torch.bool);valid[:,0,0]=True
        loss=masked_replay_distillation(student,teacher,valid);self.assertGreater(float(loss),0.)
        loss.backward()
        for name,v in student.items():
            self.assertGreater(float(v.grad[:,:,0,0].abs().sum()),0.)
            self.assertEqual(float(v.grad[:,:,1,:].abs().sum()),0.)
            self.assertEqual(float(v.grad[:,:,0,1].abs().sum()),0.)
            self.assertIsNone(teacher[name].grad)

    def test_empty_mask_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'mask'):
            masked_replay_distillation(self.heads(),self.heads(),torch.zeros(1,2,2,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError,'non-finite'):
            masked_replay_distillation(self.heads(float('nan')),self.heads(),torch.ones(1,2,2,dtype=torch.bool))

    def test_absent_protocol_preserves_original_training(self):
        self.assertIsNone(distillation_settings({}))
        with self.assertRaises(ValueError):distillation_settings({'replay_distillation':{'weight':1}})
