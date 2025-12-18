
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

