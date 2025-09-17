import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass

# In your package, you would import these from your modules
# from your_module import FitSetup, FitResult, _parse_fit_results

from tfscreen.calibration.calibrate import (
    FitResult,
    _parse_fit_results
)

@dataclass
class FitSetup:
    linear_model_df: pd.DataFrame
    A0_df: pd.DataFrame
    # ... other fields are not strictly needed by the test, but a real get_args_tuple would use them
    def get_args_tuple(self, *args, **kwargs): return ()

@pytest.fixture
def fit_inputs_and_results():
    """Provides a complete set of inputs for the parsing function."""
    # Fake raw results from a fitter
    params = np.arange(0.0, 1.0, 0.1)  # [0.0, 0.1, 0.2, ..., 0.9]
    std_errors = params / 10.0         # [0.0, 0.01, 0.02, ..., 0.09]

    # Create the prerequisite FitSetup object
    lm_df = pd.DataFrame({
        "b_idx": [0, 1, 0], "m_idx": [2, 3, 0]
    }, index=pd.MultiIndex.from_tuples([
        ("wt", "iptg"), ("mut1", "iptg"), ("background", "iptg")
    ], names=['condition', 'titrant_name']))

    a0_df = pd.DataFrame({"replicate": ["r1", "r2"], "A0_idx": [4, 5]})
    
    lookup = {"iptg": np.array([6, 7])}
    
    setup = FitSetup(linear_model_df=lm_df, A0_df=a0_df)
    # Add attributes dynamically for this test, since we don't need a full FitSetup
    setup.bg_results_lookup = lookup

    # Create the fit_data dictionary
    fit_data = {
        "y": np.array([15.5, 16.5]),
        "y_var": np.array([0.25, 0.04]),
        "theta": np.array([0.1, 0.9]), "titrant_conc": np.array([1.0, 10.0]),
        "pre_time": np.array([2.0, 2.0]), "time": np.array([8.0, 8.0]),
    }
    
    return {
        "params": params, "std_errors": std_errors,
        "cov_matrix": np.zeros((10, 10)),
        "fit_setup": setup, "fit_data": fit_data
    }

def test_parse_fit_results(mocker, fit_inputs_and_results):
    """
    Tests that the raw fitter output is correctly parsed and organized.
    """
    # 1. ARRANGE
    # Mock the predict_with_error function to return a predictable value
    mock_pred_vals = np.array([15.45, 16.55])
    mock_pred_std = np.array([0.5, 0.2])
    mocker.patch('tfscreen.calibration.calibrate.predict_with_error', return_value=(mock_pred_vals, mock_pred_std))
    # Note: Change '__main__' to your module's path when using in your package
    
    # 2. ACT
    result = _parse_fit_results(**fit_inputs_and_results)

    # 3. ASSERT
    assert isinstance(result, FitResult)
    params = fit_inputs_and_results["params"]
    std_errors = fit_inputs_and_results["std_errors"]

    # --- Assert linear_model_df is correctly populated and filtered ---
    lm_df = result.linear_model_df
    assert "background" not in lm_df.index.get_level_values('condition')
    assert lm_df.shape[0] == 2 # Should have dropped the background row
    # Check values for ('wt', 'iptg') -> b_idx=0, m_idx=2
    assert lm_df.loc[("wt", "iptg"), "b_est"] == params[0]
    assert lm_df.loc[("wt", "iptg"), "m_std"] == std_errors[2]
    # Check values for ('mut1', 'iptg') -> b_idx=1, m_idx=3
    assert lm_df.loc[("mut1", "iptg"), "m_est"] == params[3]
    assert lm_df.loc[("mut1", "iptg"), "b_std"] == std_errors[1]
    
    # --- Assert A0_df is correctly populated ---
    a0_df = result.A0_df
    # Check r1 -> A0_idx=4
    assert a0_df.loc[a0_df.replicate == 'r1', "A0_est"].iloc[0] == params[4]
    # Check r2 -> A0_idx=5
    assert a0_df.loc[a0_df.replicate == 'r2', "A0_std"].iloc[0] == std_errors[5]

    # --- Assert pred_df is correctly populated ---
    pred_df = result.pred_df
    np.testing.assert_array_equal(pred_df["calc_est"], mock_pred_vals)
    np.testing.assert_array_equal(pred_df["calc_std"], mock_pred_std)
    np.testing.assert_allclose(pred_df["obs_est"], [15.5, 16.5])
    np.testing.assert_allclose(pred_df["obs_std"], [0.5, 0.2]) # sqrt of y_var

    # --- Assert bg_model_param dictionary is correctly populated ---
    bg_param = result.bg_model_param
    assert "iptg" in bg_param
    # Should have sliced params at indices [6, 7]
    np.testing.assert_array_equal(bg_param["iptg"], [params[6], params[7]])