import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from quality_estimation.expected_deficit import expected_deficit,far_deficit_with_near_retained,density,bounded_expected_deficit,bounded_far_deficit_with_near_retained

class ExpectedDeficitTests(unittest.TestCase):
    def test_zero_variance_limit_and_boundary_value(self):
        np.testing.assert_allclose(expected_deficit([.2,.7,.9],[0.,0.,0.],.7),[.5,0.,0.],atol=1e-10)
        self.assertAlmostEqual(expected_deficit(.7,.04,.7),.2/np.sqrt(2*np.pi))

    def test_joint_truncation_matches_direct_conditional_integration(self):
        for rho in [-.7,0.,.7]:
            m=np.array([[.7,.83,.46]]);sn=.12;sf=.18;c=np.array([[[.01,0,0],[0,sn*sn,rho*sn*sf],[0,rho*sn*sf,sf*sf]]])
            actual=far_deficit_with_near_retained(m,c)[0]
            expected=quad(lambda y:(.5-y)*density((y-.46)/sf)/sf*ndtr((.83+rho*sn/sf*(y-.46)-.8)/(sn*np.sqrt(1-rho*rho))),-np.inf,.5,epsabs=1e-10)[0]
            self.assertAlmostEqual(actual,expected,places=8)
            if rho==0:self.assertAlmostEqual(actual,ndtr((.83-.8)/sn)*expected_deficit(.46,sf*sf,.5),places=10)

    def test_bounded_severity_does_not_exaggerate_negative_gaussian_tail(self):
        m=np.array([[.5,.83,.05]]);sn=.12;sf=.3;rho=.5;c=np.array([[[.01,0,0],[0,sn*sn,rho*sn*sf],[0,rho*sn*sf,sf*sf]]])
        value=bounded_far_deficit_with_near_retained(m,c)[0]
        integrand=lambda y:max(.5-np.clip(y,0,1),0)*density((y-.05)/sf)/sf*ndtr((.83+rho*sn/sf*(y-.05)-.8)/(sn*np.sqrt(1-rho*rho)))
        expected=quad(integrand,-np.inf,0,epsabs=1e-10)[0]+quad(integrand,0,.5,epsabs=1e-10)[0]
        self.assertAlmostEqual(value,expected,delta=1e-7)  # Existing finite-node bivariate CDF quadrature.
        self.assertGreaterEqual(value,0);self.assertLessEqual(value,.5)
        self.assertAlmostEqual(bounded_expected_deficit(-5.,.01,.7),.7)

    def test_cdf_saturation_does_not_erase_severity(self):
        a=expected_deficit(.1,.0001,.7);b=expected_deficit(.3,.0001,.7)
        self.assertGreater(a,b+.19)

if __name__=='__main__':unittest.main()
