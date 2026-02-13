
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
    
    # Needs to match what get_wt_k expects or we mock get_wt_k
    cal_dict = {"dilution": 0.1} 
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots, \
         patch("tfscreen.calibration.plot.get_wt_k") as mock_get_wt_k:
        
        mock_get_wt_k.return_value = 0.2 # k_pre
        
        fig = MagicMock()
        ax = MagicMock()
        val = np.empty((1,1), dtype=object)
        val[0,0] = ax
        mock_subplots.return_value = (fig, val)
        
        indiv_replicates(df, calibration_dict=cal_dict)
        
        # Verify plot was called. 
        # Original points: 3 (-5, 0, 5)
        # We add 2 points at t=0 (pre and post dilution) and remove existing t=0.
        # So final points should be 4: -5, 0 (pre), 0 (post), 5.
        
        # Check that plot was called with a 4-element array for x and y
        # We can't easily check the call arguments of ax.plot if they are numpy arrays without iteration
        # but we can look for the call.
        found_plot = False
        for call in ax.plot.call_args_list:
            args, _ = call
            if len(args[0]) == 4:
                found_plot = True
                # x should be [-5, 0, 0, 5]
                assert np.allclose(args[0], [-5, 0, 0, 5])
                # y calculation:
                # lnA0 = 10.0
                # k_pre = 0.2, t_pre = 5.0 => lnA_at_0_pre = 10.0 + 0.2*5 = 11.0
                # lnA_at_0_sel = 11.0 + ln(0.1) = 11.0 - 2.3025... = 8.697...
                # plot_y should be [10.0, 11.0, 8.697..., 12.0]
                assert np.allclose(args[1], [10.0, 11.0, 11.0 + np.log(0.1), 12.0])
                
        assert found_plot

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
    
    # Create duplicate calc_est columns. This causes cond_df["calc_est"] to be a 
    # DataFrame, and .to_numpy() to be (N, 2). Then slicing might yield (N, 1) etc.
    bad_df = pd.concat([df, pd.Series([10, 11, 12], name="calc_est")], axis=1)
    
    cal_dict = {"dilution": 0.1}
    
    with patch("matplotlib.pyplot.subplots") as mock_subplots, \
         patch("tfscreen.calibration.plot.get_wt_k") as mock_get_wt_k:
        
        mock_get_wt_k.return_value = 0.2
        fig = MagicMock(); ax = MagicMock()
        val = np.empty((1,1), dtype=object); val[0,0] = ax
        mock_subplots.return_value = (fig, val)
        
        # This should NO LONGER trigger the bug because of .ravel()
        indiv_replicates(bad_df, calibration_dict=cal_dict)
        
        # Check that the plot call with 4 points happened
        found_plot = False
        for call in ax.plot.call_args_list:
            args, _ = call
            if len(args[0]) == 4:
                found_plot = True
        assert found_plot
