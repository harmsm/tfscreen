
import pytest
import numpy as np
from tfscreen.models.occupancy_growth_model import OccupancyGrowthModel

class TestOccupancyGrowthModel:

    def test_calc_baryani_integral_values(self):
        """Test integral calculation for known values."""
        model = OccupancyGrowthModel()
        
        # Test t=0 -> integral should be 0
        val = model.calc_baryani_integral(0, tau=10, k=1.0)
        assert np.isclose(val, 0.0)
        
        # Test large t, integral should look like t - tau (linear part) + constant?
        # S(t) -> 1. Integral of S(t) -> t. 
        # Wait, S(t) is 1/(1+exp(-k(t-tau))).
        # For large t, S(t) approx 1. Integral approx t.
        # Let's check a numerical integration.
        
        # Numerical integration of S(t) from 0 to 5 for tau=2, k=5
        t_end = 5.0
        tau = 2.0
        k = 5.0
        
        # S(t)
        def sigmoid(t):
            return 1.0 / (1.0 + np.exp(-k * (t - tau)))
            
        from scipy.integrate import quad
        expected, _ = quad(sigmoid, 0, t_end)
        
        calculated = model.calc_baryani_integral(t_end, tau, k)
        assert np.isclose(calculated, expected, rtol=1e-4)

    def test_predict_trajectory_dilution_step(self):
        """Verify the dilution step creates the correct offset."""
        model = OccupancyGrowthModel()
        
        t_pre = 100.0
        ln_cfu0 = 10.0
        mu1 = 0.01
        mu2 = 0.02
        dilution = 0.1 # ln(0.1) ~ -2.3
        
        # Prediction just before dilution (t_sel = -epsilon) -> linear growth
        pred_pre = model.predict_trajectory(
            t_pre, -1e-6, ln_cfu0, mu1, mu2, dilution
        )
        expected_pre = ln_cfu0 + mu1 * (t_pre - 1e-6)
        assert np.isclose(pred_pre, expected_pre)
        
        # Prediction just after dilution (t_sel = 0)
        pred_zero = model.predict_trajectory(
            t_pre, 0.0, ln_cfu0, mu1, mu2, dilution
        )
        expected_zero = (ln_cfu0 + mu1 * t_pre) + np.log(dilution)
        assert np.isclose(pred_zero, expected_zero)
        
        # Check gap size
        gap = pred_pre - pred_zero
        # gap should be approx -ln(dilution) - mu1*epsilon
        assert np.isclose(gap, -np.log(dilution), atol=1e-4)

    def test_predict_trajectory_vectorization(self):
        """Test that the model handles vectorized inputs correctly."""
        model = OccupancyGrowthModel()
        
        # Vectorized t_sel
        t_sel = np.array([-10, 0, 10, 20])
        t_pre = 100.0
        ln_cfu0 = 10.0
        mu1 = 0.01
        mu2 = 0.02
        dilution = 0.1
        
        preds = model.predict_trajectory(t_pre, t_sel, ln_cfu0, mu1, mu2, dilution)
        assert len(preds) == 4
        assert preds[0] > preds[1] # t=-10 (10.9) vs t=0 (8.7)
        # Wait, t=-10 is pre-growth. t=0 is post-dilution.
        # pre-growth end: 10 + 0.01*100 = 11.
        # pre-growth at -10: 10 + 0.01*(90) = 10.9
        # post-dilution start: 11 + ln(0.1) = 11 - 2.3 = 8.7
        # So preds[0] (10.9) > preds[1] (8.7)
        assert preds[0] > preds[1]
        
        # Vectorized parameters
        dilutions = np.array([0.1, 0.01])
        t_sel_single = 0.0
        preds_dil = model.predict_trajectory(t_pre, t_sel_single, ln_cfu0, mu1, mu2, dilutions)
        assert len(preds_dil) == 2
        # Lower dilution factor (0.01) means more drop, so lower value based on log
        assert preds_dil[1] < preds_dil[0]

    def test_growth_bound_check(self):
        """Check behavior when tau is large (delayed transition)."""
        model = OccupancyGrowthModel()
        
        t_end = 10.0
        tau = 100.0 # Transition happens way later
        k = 10.0
        mu1 = 0.1
        mu2 = 0.5
        
        # Should grow at mu1 mostly
        pred = model.predict_trajectory(
            t_pre=0, t_sel=t_end, ln_cfu0=0, mu1=mu1, mu2=mu2, dilution=1.0, tau=tau, k_sharp=k
        )
        
        # mu(t) approx mu1. integral approx mu1 * t
        expected = mu1 * t_end
        assert np.isclose(pred, expected, rtol=0.1)
