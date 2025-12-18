
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from tfscreen.calibration.calibrate import (
    _prep_calibration_df,
    _build_calibration_X,
    setup_calibration,
    calibrate,
    _fit_theta
)
from tfscreen.fitting.fit_manager import FitManager

@pytest.fixture
def example_calibration_df():
    # Helper to build a minimal valid dataframe
    df = pd.DataFrame({
        "ln_cfu": [10.0, 11.0, 10.0, 11.0],
        "ln_cfu_std": [0.1, 0.1, 0.1, 0.1],
        "t_pre": [5.0, 5.0, 5.0, 5.0],
        "condition_pre": ["background", "background", "background", "background"],
        "t_sel": [5.0, 5.0, 5.0, 5.0],
        "condition_sel": ["background", "cond1", "background", "cond1"],
        "replicate": [1, 1, 1, 1],
        "genotype": ["wt", "mutant", "wt", "mutant"],
        "titrant_name": ["t1", "t1", "t1", "t1"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "theta": [1.0, 0.5, 1.0, 0.5],
        "theta_std": [0.01, 0.01, 0.01, 0.01],
        "censored": [False, False, False, False]
    })
    return df

def test_prep_calibration_df_defaults(example_calibration_df):
    df = example_calibration_df.drop(columns=["genotype", "censored"])
    
    with pytest.warns(UserWarning, match="No genotype column"):
        df_out = _prep_calibration_df(df)
        
    assert "genotype" in df_out.columns
    assert (df_out["genotype"] == "wt").all()
    assert "censored" in df_out.columns
    assert not df_out["censored"].any()

def test_build_calibration_X_structure(example_calibration_df):
    # Test that it returns expected shapes/objects
    df = _prep_calibration_df(example_calibration_df)
    
    # We are testing _build_calibration_X (not _new because that's not called by default? 
    # Wait, the code imports _build_calibration_X, but inside calibrate.py there is _build_calibration_X and _build_calibration_X_new.
    # The setup_calibration calls _build_calibration_X.
    
    y_obs, y_std, X, param_df = _build_calibration_X(df)
    
    assert len(y_obs) == len(df)
    assert len(y_std) == len(df)
    # X rows should match df rows
    assert X.shape[0] == len(df)
    # X columns should match param_df rows
    assert X.shape[1] == len(param_df)
    
    # Check param classes exist
    classes = param_df["param_class"].unique()
    assert "ln_cfu_0" in classes
    assert "k_bg_b" in classes
    assert "dk_geno" in classes

def test_setup_calibration(example_calibration_df):
    fm = setup_calibration(example_calibration_df, ln_cfu_0_guess=15., k_bg_guess=0.03)
    
    assert isinstance(fm, FitManager)
    assert fm.num_obs == len(example_calibration_df)
    
    # Check guesses
    guesses = fm.guesses
    param_df = fm.param_df
    
    ln_cfu_0_mask = param_df["param_class"] == "ln_cfu_0"
    assert np.allclose(guesses[ln_cfu_0_mask], 15.0)
    
    k_bg_b_mask = param_df["param_class"] == "k_bg_b"
    assert np.allclose(guesses[k_bg_b_mask], 0.03)

def test_calibrate_mocked(example_calibration_df, tmp_path):
    # Mock the heavy lifting: run_least_squares, predict_with_error, write_calibration, read_calibration
    # Also fit_theta, but that's internal.
    
    with patch("tfscreen.calibration.calibrate.run_least_squares") as mock_rls, \
         patch("tfscreen.calibration.calibrate.predict_with_error") as mock_predict, \
         patch("tfscreen.calibration.calibrate.write_calibration") as mock_write, \
         patch("tfscreen.calibration.calibrate.read_calibration") as mock_read, \
         patch("tfscreen.calibration.calibrate._fit_theta") as mock_fit_theta:
             
        # Setup mocks
        # run_least_squares returns: params, std_errors, cov_matrix, info
        # We need to return an array of correct size. 
        # Calculate size from setup_calibration
        fm = setup_calibration(example_calibration_df)
        n_params = fm.num_params
        
        mock_rls.return_value = (np.zeros(n_params), np.zeros(n_params), np.eye(n_params), {})
        
        # predict_with_error returns: pred, pred_std
        n_obs = fm.num_obs
        mock_predict.return_value = (np.zeros(n_obs), np.zeros(n_obs))
        
        # mock_fit_theta returns dict of params
        mock_fit_theta.return_value = {"t1": [1,2,3,4]}
        
        # mock_read returns what we expect
        dummy_cal = {"dummy": "cal"}
        mock_read.return_value = dummy_cal
        
        out_file = str(tmp_path / "out.json")
        res, pred_df, param_df = calibrate(example_calibration_df, out_file)
        
        assert res == dummy_cal
        mock_write.assert_called_once()
        kwargs = mock_write.call_args.kwargs
        cal_dict = kwargs["calibration_dict"]
        assert "theta_param" in cal_dict
        assert "k_bg" in cal_dict
        
def test_fit_theta(example_calibration_df):
    # This uses run_least_squares and predefined models.
    # We can mock run_least_squares or integration test it if models are available.
    # Let's mock run_least_squares to avoid scipy dependency issues on small data or model library complexity.
    
    with patch("tfscreen.calibration.calibrate.run_least_squares") as mock_rls:
        # returns params, std, cov, info
        mock_rls.return_value = (np.array([1, 1, 1, 1]), None, None, None)
        
        res = _fit_theta(example_calibration_df)
        assert "t1" in res
        assert list(res["t1"]) == [1, 1, 1, 1]

