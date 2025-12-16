
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from tfscreen.analysis.independent.model_pre_growth import _validate_group_array
from tfscreen.analysis.independent._get_indiv_growth import _run_batch_fits, _prepare_and_validate_growth_data
from tfscreen.analysis.independent.cfu_to_theta import _prep_inference_df, _run_inference

# --- model_pre_growth coverage ---
def test_validate_group_array_errors():
    with pytest.raises(ValueError, match="must be a 1D array"):
        _validate_group_array(np.array([[1]]), (1,), "test")
    
    with pytest.raises(ValueError, match="same length"):
        _validate_group_array(np.array([1, 2]), (1,), "test")

# --- _get_indiv_growth coverage ---
def test_run_batch_fits_single_index():
    # Test line 87: single index column
    df = pd.DataFrame({
        "id": [1],
        "t_sel": [0],
        "ln_cfu": [10.0]
    })
    series_selector = ["id"]
    needs_columns = ["ln_cfu"]
    
    def mock_fit(**kwargs):
        # returns param_df, pred_df
        p = pd.DataFrame({"k": [1.0]})
        pr = pd.DataFrame({"pred": [10.0]})
        return p, pr
        
    param, pred = _run_batch_fits(df, series_selector, mock_fit, needs_columns, fitter_kwargs=None)
    assert "id" in param.columns
    assert param.iloc[0]["id"] == 1

def test_run_batch_fits_kwargs():
    # Test line 78: fitter_kwargs
    df = pd.DataFrame({
        "id": [1],
        "t_sel": [0],
        "ln_cfu": [10.0]
    })
    series_selector = ["id"]
    needs_columns = ["ln_cfu"]
    
    mock_fit = MagicMock(return_value=(pd.DataFrame({"k": [1.0]}), pd.DataFrame({"pred": [10.0]})))
    
    _run_batch_fits(df, series_selector, mock_fit, needs_columns, fitter_kwargs={"extra": "val"})
    
    _, kwargs = mock_fit.call_args
    assert kwargs["extra"] == "val"

# --- cfu_to_theta coverage ---
def test_prep_inference_df_float_error():
    # Test line 137-138
    df = pd.DataFrame({
        "ln_cfu": ["not_a_number"], 
        "ln_cfu_std": [0.1],
        "t_pre": [0], "t_sel": [0],
        "genotype": ["wt"], "library": ["l"], "replicate": [1],
        "titrant_name": ["n"], "titrant_conc": [0],
        "condition_pre": ["c"], "condition_sel": ["c"]
    })
    
    # Mock read_dataframe to just return our df
    with patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", return_value=df):
        with patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", return_value=df):
            with patch("tfscreen.analysis.independent.cfu_to_theta.check_columns"):
                 with pytest.raises(ValueError, match="Could not coerce"):
                     _prep_inference_df(df, calibration_data={}, max_batch_size=1)

def test_run_inference_nan_error():
    # Test line 514, 520
    fm = MagicMock()
    fm.guesses_transformed = np.array([0.5])
    fm.lower_bounds_transformed = np.array([0.0])
    fm.upper_bounds_transformed = np.array([1.0])
    
    # Test y_obs nan
    fm.y_obs = np.array([np.nan])
    fm.y_std = np.array([0.1])
    with pytest.raises(ValueError, match="y_obs contains"):
        _run_inference(fm)
        
    # Test y_std nan
    fm.y_obs = np.array([1.0])
    fm.y_std = np.array([np.nan])
    with pytest.raises(ValueError, match="y_std contains"):
        _run_inference(fm)
