import unittest
import torch
from quality_estimation.blend_perception import interpolate_parameters

class ParameterInterpolationTest(unittest.TestCase):
    def states(self):
        return {'w':torch.tensor([1.,2.]),'running_mean':torch.tensor([.5]),'counter':torch.tensor(3)}, {'w':torch.tensor([3.,6.]),'running_mean':torch.tensor([.5]),'counter':torch.tensor(3)}
    def test_only_parameters_interpolated_without_mutating_parents(self):
        initial,adapted=self.states();result=interpolate_parameters(initial,adapted,{'w'},.25)
        torch.testing.assert_close(result['w'],torch.tensor([1.5,3.]));self.assertTrue(torch.equal(result['counter'],initial['counter']))
        result['running_mean'].add_(1);self.assertEqual(initial['running_mean'].item(),.5)
        self.assertTrue(torch.equal(interpolate_parameters(initial,adapted,{'w'},0)['w'],initial['w']))
    def test_changed_buffers_or_dtype_cannot_be_silently_blended(self):
        initial,adapted=self.states();adapted['running_mean'].add_(1)
        with self.assertRaisesRegex(ValueError,'buffers changed'):interpolate_parameters(initial,adapted,{'w'},.5)
        initial,adapted=self.states();adapted['w']=adapted['w'].double()
        with self.assertRaisesRegex(ValueError,'dtype'):interpolate_parameters(initial,adapted,{'w'},.5)

if __name__=='__main__':unittest.main()
