import pytest
import pandas as pd
import numpy as np

# Import all functions to be tested
from tfscreen.analysis._get_indiv_growth import (
    _prepare_and_validate_growth_data,
    _run_batch_fits,
    _apply_pre_growth_correction,
    _get_indiv_growth
)

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies."""
    
    # Mock the fitter function
    def mock_fitter(t_sel, ln_cfu, **kwargs):
        # --- FIX STARTS HERE ---
        # The arrays are passed with shape (num_series, num_timepoints).
        # Correctly get num_series from the first dimension.
        num_series = t_sel.shape[0]
        num_timepoints = t_sel.shape[1]
        # --- FIX ENDS HERE ---
        
        # Return one row of fake params per series
        param_df = pd.DataFrame({
            "k_est": np.linspace(0.1, 0.5, num_series),
            "A0_est": np.full(num_series, 1e6),
            "A0_std": np.full(num_series, 1e5),
        })
        
        # Return a prediction df with one row per observation
        total_rows = num_series * num_timepoints
        pred_df = pd.DataFrame({
            "obs": np.random.rand(total_rows),
            "pred": np.random.rand(total_rows)
        })
        return param_df, pred_df

    # Patch the MODEL_LIBRARY to use our corrected mock fitter
    mocker.patch(
        "tfscreen.analysis._get_indiv_growth.MODEL_LIBRARY",
        {
            "wls": {
                "fcn": mock_fitter,
                "args": ["t_sel", "ln_cfu","ln_cfu_var"]
            }
        }
    )

    # Make the get_scaled_cfu mock more realistic by having it add the
    # columns the downstream code expects.
    def mock_get_scaled_cfu(*args, **kwargs):
        df = args[0].copy() # Work on a copy to avoid side effects
        if "cfu" in df.columns:
            df["ln_cfu"] = np.log(df["cfu"])
        if "cfu_std" in df.columns:
            df["ln_cfu_std"] = df["cfu_std"] / df["cfu"]
        return df

    mocker.patch(
        "tfscreen.analysis._get_indiv_growth.get_scaled_cfu", 
        side_effect=mock_get_scaled_cfu
    )

    # ... (rest of the mocks are unchanged) ...
    mocker.patch("tfscreen.analysis._get_indiv_growth.check_columns", return_value=None)
    mocker.patch("tfscreen.analysis._get_indiv_growth.get_wt_k", return_value=1.0)
    
    def mock_model_pre_growth(k_est, lnA0_est, **kwargs):
        dk_geno = k_est - 1.0
        lnA0_pre = lnA0_est - 5.0
        lnA0_est_updated = lnA0_est * 1.1
        lnA0_std_updated = np.full_like(k_est, 0.5)
        return dk_geno, lnA0_pre, lnA0_est_updated, lnA0_std_updated
    
    mocker.patch("tfscreen.analysis._get_indiv_growth.model_pre_growth", side_effect=mock_model_pre_growth)


@pytest.fixture
def base_df():
    """Provides a base DataFrame with multiple series and different timepoint counts."""
    data = {
        "replicate":    [1, 1, 1,  2, 2, 2, 2,  3, 3, 3],
        "condition_sel":[1, 1, 1,  1, 1, 1, 1,  2, 2, 2],
        "titrant_name": ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"],
        "titrant_conc": [0, 0, 0,   0, 0, 0, 0,   1, 1, 1],
        "t_sel":        [0, 1, 2,   0, 1, 2, 3,   0, 1, 2],
        "t_pre":        [5, 5, 5,   5, 5, 5, 5,   5, 5, 5],
        "ln_cfu":       np.arange(10, 20),
        "ln_cfu_var":       np.arange(10, 20),
        "genotype":     ["g1", "g1", "g1", "g1", "g1", "g1", "g1", "g2", "g2", "g2"],
        "dk_geno_mask": [True, True, True, False, False, False, False, True, True, True]
    }
    return pd.DataFrame(data)

# --- Test Suite ---

def test_get_indiv_growth_full_pipeline(mock_dependencies, base_df):
    """
    An integration test for the main _get_indiv_growth function, covering all helpers.
    """
    # Define selectors for this run
    series_selector = ["replicate", "condition_sel", "titrant_name", "titrant_conc"]
    dk_geno_selector = ["genotype"]
    lnA0_selector = ["replicate"]

    param_df, pred_df = _get_indiv_growth(
        df=base_df,
        series_selector=series_selector,
        calibration_data={}, # Mocked, so can be empty
        dk_geno_selector=dk_geno_selector,
        dk_geno_mask_col="dk_geno_mask",
        lnA0_selector=lnA0_selector
    )

    # 1. Test final output shapes
    # There are 3 unique series in the base_df
    assert param_df.shape[0] == 3
    # There are 10 total rows in the base_df
    assert pred_df.shape[0] == 10

    # 2. Test parameter dataframe content
    # Check that metadata columns are present
    assert "replicate" in param_df.columns
    assert "genotype" in param_df.columns
    # Check that corrected parameters from the mock model_pre_growth are present
    assert "dk_geno" in param_df.columns
    assert "lnA0_pre" in param_df.columns
    
    # 3. Test prediction dataframe content
    assert "pred" in pred_df.columns
    assert "obs" in pred_df.columns
    # Check that temporary columns are gone
    assert "_timepoint_count" not in pred_df.columns
    assert "_t_sel_row_number" not in pred_df.columns

def test_prepare_data_invalid_fit_method(base_df):
    """
    Tests that the validation helper raises an error for a bad fit_method.
    """
    with pytest.raises(ValueError, match="fit method 'bad_method' not recognized"):
        _prepare_and_validate_growth_data(
            df=base_df,
            series_selector=["replicate"],
            fit_method="bad_method",
            dk_geno_selector=None,
            lnA0_selector=None
        )

def test_apply_correction_no_groups(mocker):
    """
    Tests that model_pre_growth is called with None for groups if selectors are not provided.
    """
    num_series = 5
    
    # --- FIX STARTS HERE ---
    # Define a valid return value for the mock
    mock_return_value = (
        np.ones(num_series), 
        np.ones(num_series), 
        np.ones(num_series), 
        np.ones(num_series)
    )
    mock_model = mocker.patch(
        "tfscreen.analysis._get_indiv_growth.model_pre_growth", 
        return_value=mock_return_value
    )
    # --- FIX ENDS HERE ---
    
    mocker.patch("tfscreen.analysis._get_indiv_growth.get_wt_k", return_value=1.0)

    param_df = pd.DataFrame({
        "k_est": np.ones(num_series),
        "lnA0_est": np.ones(num_series),
        "lnA0_std": np.ones(num_series),
    })
    series_metadata_df = pd.DataFrame({
        "condition_sel": ["A"] * num_series,
        "titrant_name": ["B"] * num_series,
        "titrant_conc": [0] * num_series,
        "t_pre": [5] * num_series
    })

    _apply_pre_growth_correction(
        param_df=param_df,
        series_metadata_df=series_metadata_df,
        calibration_data={},
        dk_geno_selector=None,
        dk_geno_mask_col=None,
        lnA0_selector=None
    )
    
    # This assertion will now work because the function didn't crash
    call_args, call_kwargs = mock_model.call_args
    assert call_kwargs["dk_geno_groups"] is None
    assert call_kwargs["dk_geno_mask"] is None
    assert call_kwargs["lnA0_groups"] is None