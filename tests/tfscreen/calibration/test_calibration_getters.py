
import pytest
import numpy as np
import pandas as pd
from tfscreen.calibration import (
    get_background,
    get_k_vs_theta,
    get_wt_theta,
    get_wt_k
)

@pytest.fixture
def dummy_calibration():
    # Construct a calibration dictionary matching the structure expected
    # by read_calibration and used by getters.
    
    # k_bg_df: m, b for each titrant
    k_bg_df = pd.DataFrame({
        "m": [1.0, 2.0],
        "b": [0.1, 0.2]
    }, index=["t1", "t2"])
    
    # dk_cond_df: m, b for each condition
    dk_cond_df = pd.DataFrame({
        "m": [0.5, 0.0],
        "b": [0.05, 0.0]
    }, index=["c1", "background"])
    
    # theta_param: [baseline, amplitude, lnK, n] for hill_repressor guesses
    # model: baseline + amplitude / (1 + (x/K)^n)
    # But wait, hill_repressor model in tfscreen uses lnK?
    # get_wt_theta docstring: "calibration stores lnK"
    # let's assume standard hill repressor
    theta_param = {
        "t1": [0.0, 1.0, np.log(10.0), 2.0], # K=10, n=2
        "t2": [0.0, 1.0, np.log(5.0), 1.0]   # K=5, n=1
    }
    
    return {
        "k_bg_df": k_bg_df,
        "dk_cond_df": dk_cond_df,
        "theta_param": theta_param,
        # Raw dicts might be needed if read_calibration re-parses specific keys, 
        # but getters use _df mostly if passed a dict that already has them?
        # read_calibration checks if input is dict, returns it. 
        # So we need to ensure the input dict has what read_calibration would put there.
        # read_calibration ADDS _df keys. If we pass a dict with _df keys, it works.
    }

def test_get_background(dummy_calibration):
    # m*conc + b
    # t1: 1.0*conc + 0.1
    titrants = np.array(["t1", "t1"])
    concs = np.array([0.0, 10.0])
    
    bg = get_background(titrants, concs, dummy_calibration)
    
    assert np.allclose(bg, [0.1, 10.1])

def test_get_k_vs_theta(dummy_calibration):
    # c1: m=0.5, b=0.05
    conds = np.array(["c1", "background"])
    
    slopes, intercepts = get_k_vs_theta(conds, dummy_calibration)
    
    assert np.allclose(slopes, [0.5, 0.0])
    assert np.allclose(intercepts, [0.05, 0.0])

def test_get_wt_theta(dummy_calibration):
    # t1: K=10, n=2. baseline=0, amp=1.
    # theta = 0 + 1 / (1 + (x/10)^2)
    
    titrants = np.array(["t1"])
    concs = np.array([10.0]) # at x=K, theta should be 0.5 for simple repressor?
    # Checking tfscreen.models.generic.MODEL_LIBRARY["hill_repressor"] definition implicitly via this test.
    # Usually repressor: 1 / (1 + (x/K)^n)
    
    theta = get_wt_theta(titrants, concs, dummy_calibration)
    assert np.allclose(theta, [0.5])
    
    # Test overrides
    # Override K=100. Then at x=10, (10/100)^2 = 0.01. theta ~ 1/(1.01) ~ 0.99
    theta_ov_K = get_wt_theta(titrants, concs, dummy_calibration, override_K=100.0)
    # K=100, x=10. x << K. Result should be small (near 0).
    # Model is x^n / (x^n + K^n). 10^2 / (10^2 + 100^2) = 100 / 10100 = 1/101 ~ 0.0099
    assert np.allclose(theta_ov_K, [1.0 / 101.0])

def test_get_wt_k(dummy_calibration):
    # k = background + dk_intercept + dk_slope * theta
    
    conds = np.array(["c1"])
    titrants = np.array(["t1"])
    concs = np.array([10.0])
    
    # theta at 10.0 is 0.5 (see above)
    # background at 10.0 is 10.1 (see above)
    # c1: slope=0.5, intercept=0.05
    
    # k = 10.1 + 0.05 + 0.5 * 0.5 = 10.15 + 0.25 = 10.4
    
    k = get_wt_k(conds, titrants, concs, dummy_calibration)
    assert np.allclose(k, [10.4])
    
    # Test with explict theta
    k_explicit = get_wt_k(conds, titrants, concs, dummy_calibration, theta=np.array([0.0]))
    # k = 10.1 + 0.05 + 0.5 * 0 = 10.15
    assert np.allclose(k_explicit, [10.15])
