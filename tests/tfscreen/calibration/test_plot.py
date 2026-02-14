
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from tfscreen.calibration.plot import (
    growth_rate_fit,
    A0_hist,
    k_vs_titrant,
    k_pred_corr,
    fit_summary,
    indiv_replicates
)

@pytest.fixture
def example_data():
    n = 10
    obs = np.linspace(10, 20, n)
    obs_std = np.ones(n) * 0.1
    calc = obs + np.random.normal(0, 0.1, n)
    calc_std = np.ones(n) * 0.1
    return obs, obs_std, calc, calc_std

def test_growth_rate_fit(example_data):
    obs, obs_std, calc, calc_std = example_data
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        
        # Test 1: Let function create axis
        ret_ax = growth_rate_fit(obs, obs_std, calc, calc_std)
        # Should call scatter, errorbar, plot...
        ret_ax.scatter.assert_called()
        ret_ax.errorbar.assert_called()
        
        # Test 2: Pass axis
        growth_rate_fit(obs, obs_std, calc, calc_std, ax=ax)
        ax.scatter.assert_called()

def test_A0_hist():
    A0 = np.random.normal(16, 1, 100)
    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        
        ret_ax = A0_hist(A0, ax=None)
        ret_ax.fill.assert_called()

def test_indiv_replicates():
    df = pd.DataFrame({
        "time": np.tile(np.arange(5), 2),
        "replicate": 1,
        "genotype": "wt",
        "condition_pre": "bg",
        "titrant_name": "t1",
        "titrant_conc": np.concatenate([np.ones(5), np.ones(5)*10.0]),
        "y_obs": np.tile(np.arange(5) + 10, 2),
        "y_std": 0.1,
        "calc_est": np.tile(np.arange(5) + 10, 2),
        "t_sel": np.tile(np.arange(5), 2),
        "condition_sel": "cond1"
    })
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        fig = MagicMock()
        ax = MagicMock()
        
        # Create object array 
        val = np.empty((1,1), dtype=object)
        val[0,0] = ax
        
        mock_subplots.return_value = (fig, val)
        
        indiv_replicates(df)
        
        ax.scatter.assert_called()

def test_indiv_replicates_bad_rgb_map():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="rgb_map should be a list"):
        indiv_replicates(df, rgb_map="bad")


def test_indiv_replicates_with_dilution():
    df = pd.DataFrame({
        "time": [-5.0, 0.0, 5.0],
        "replicate": 1,
        "genotype": "wt",
        "titrant_name": "t1",
        "titrant_conc": 1.0,
        "y_obs": [10.0, 11.0, 12.0],
        "y_std": 0.1,
        "calc_est": [10.0, 11.0, 12.0],
        "t_pre": 5.0,
        "t_sel": [-5.0, 0.0, 5.0],  # Includes -t_pre point
        "condition_pre": "bg",
        "condition_sel": "cond1"
    })
    
    cal_dict = {
        "dilution": 0.1,
        "dk_cond": {"tau": {"cond1": 0.0}, "k_sharp": {"cond1": 1.0}}
    } 
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots, \
         patch("tfscreen.calibration.plot.get_wt_k") as mock_get_wt_k:
        
        mock_get_wt_k.return_value = 0.2 # k_pre and k_sel
        
        fig = MagicMock()
        ax = MagicMock()
        val = np.empty((1,1), dtype=object)
        val[0,0] = ax
        mock_subplots.return_value = (fig, val)
        
        indiv_replicates(df, calibration_dict=cal_dict)
        
        # Verify integrated plot was called (length 151: 50 pre + 1 NaN + 100 smooth)
        found_curve = False
        for call in ax.plot.call_args_list:
            args, _ = call
            if len(args[0]) == 151:
                found_curve = True
                # Check start and end X
                assert np.allclose(args[0][0], -5)
                assert np.allclose(args[0][-1], 5)
                # Check NaN separator at index 50
                assert np.isnan(args[1][50])
        assert found_curve

def test_indiv_replicates_broadcast_repro():
    """Reproduce the ValueError: could not broadcast input array from shape (2,1) into (2,)"""
    df = pd.DataFrame({
        "time": [-5.0, 0.0, 5.0],
        "replicate": 1,
        "genotype": "wt",
        "titrant_name": "t1",
        "titrant_conc": 1.0,
        "y_obs": [10.0, 11.0, 12.0],
        "y_std": 0.1,
        "calc_est": [10.0, 11.0, 12.0],
        "t_pre": 5.0,
        "t_sel": [-5.0, 0.0, 5.0],
        "condition_pre": "bg",
        "condition_sel": "cond1"
    })
    
    # Create duplicate calc_est columns. 
    bad_df = pd.concat([df, pd.Series([10, 11, 12], name="calc_est")], axis=1)
    
    cal_dict = {
        "dilution": 0.1,
        "dk_cond": {"tau": {"cond1": 0.0}, "k_sharp": {"cond1": 1.0}}
    }
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots, \
         patch("tfscreen.calibration.plot.get_wt_k") as mock_get_wt_k:
        
        mock_get_wt_k.return_value = 0.2
        fig = MagicMock(); ax = MagicMock()
        val = np.empty((1,1), dtype=object); val[0,0] = ax
        mock_subplots.return_value = (fig, val)
        
        # This should NO LONGER trigger the bug because of .ravel()
        indiv_replicates(bad_df, calibration_dict=cal_dict)
        
        # Check that the plot calls happened
        found_curve = False
        for call in ax.plot.call_args_list:
            args, _ = call
            if len(args[0]) == 151:
                found_curve = True
        assert found_curve

def test_indiv_replicates_with_dk_geno():
    """Verify that dk_geno from calibration_dict is used in growth rate calculation."""
    df = pd.DataFrame({
        "time": [-5.0, 0.0, 5.0],
        "replicate": 1,
        "genotype": "mutant",
        "titrant_name": "t1",
        "titrant_conc": 1.0,
        "y_obs": [10.0, 11.0, 12.0],
        "y_std": 0.1,
        "calc_est": [10.0, 11.0, 12.0],
        "t_pre": 5.0,
        "t_sel": [-5.0, 0.0, 5.0],
        "condition_pre": "bg",
        "condition_sel": "cond1"
    })
    
    # Define calibration with a significant dk_geno for 'mutant'
    # Base growth (k_bg) = 0.2 (from mock_get_wt_k)
    # dk_geno = 0.1
    # Expected mu = 0.3
    
    cal_dict = {
        "dilution": 1.0, # No dilution for simplicity
        "dk_cond": {"tau": {"cond1": 0.0}, "k_sharp": {"cond1": 1.0}},
        "dk_geno": {"mutant": 0.1}
    } 
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots, \
         patch("tfscreen.calibration.plot.get_wt_k") as mock_get_wt_k:
        
        mock_get_wt_k.return_value = 0.2 
        
        fig = MagicMock(); ax = MagicMock()
        val = np.empty((1,1), dtype=object); val[0,0] = ax
        mock_subplots.return_value = (fig, val)
        
        # We need to spy on OccupancyGrowthModel or calculate expectation.
        # Let's mock OccupancyGrowthModel to check inputs?
        
        with patch("tfscreen.calibration.plot.OccupancyGrowthModel") as MockOGM:
            instance = MockOGM.return_value
            # return some dummy value for predict to avoid errors
            instance.predict_trajectory.return_value = np.zeros(100) # dummy

            indiv_replicates(df, calibration_dict=cal_dict)
            
            # Check what mu1/mu2 were passed to predict_trajectory
            assert instance.predict_trajectory.called
            args, kwargs = instance.predict_trajectory.call_args
            
            # kwargs should contain mu1, mu2
            assert "mu1" in kwargs
            assert "mu2" in kwargs
            
            # Expected: 0.2 + 0.1 = 0.3
            assert np.isclose(kwargs["mu1"], 0.3)
            assert np.isclose(kwargs["mu2"], 0.3)
