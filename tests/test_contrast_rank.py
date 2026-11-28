import unittest
import numpy as np
from quality_estimation.contrast_rank import standardized_contrast

class ContrastRankTest(unittest.TestCase):
    def test_zero_sd_is_bounded_and_does_not_invent_probability(self):
        z=standardized_contrast(np.array([.1,-.1,0]),np.zeros(3));np.testing.assert_array_equal(z,[10,-10,0])
    def test_sign_and_covariance_scale_preserved(self):
        np.testing.assert_allclose(standardized_contrast([.2,-.3],[.1,.1]),[2,-3])
    def test_invalid_sd_and_alignment_rejected(self):
        for mean,sd in [([1],[-1]),([1],[float('nan')]),([1,2],[1])]:
            with self.assertRaises(ValueError):standardized_contrast(mean,sd)
