
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
        assert "dilution" in cal_dict
        assert cal_dict["dilution"] == 0.0196
        
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


def test_dilution_offset(example_calibration_df):
    """Verify dilution offset is subtracted from y_obs for rows with t_sel > 0."""

    dilution = 0.05
    df = _prep_calibration_df(example_calibration_df)
    raw_ln_cfu = df["ln_cfu"].to_numpy().copy()

    y_obs, _, _, _ = _build_calibration_X(df, dilution=dilution)

    # All rows in the fixture have t_sel > 0, so all should be offset
    expected = raw_ln_cfu - np.log(dilution)
    assert np.allclose(y_obs, expected)

    # Now test with some rows having t_sel == 0
    df2 = _prep_calibration_df(example_calibration_df.copy())
    df2.iloc[0, df2.columns.get_loc("t_sel")] = 0.0
    raw2 = df2["ln_cfu"].to_numpy().copy()

    y_obs2, _, _, _ = _build_calibration_X(df2, dilution=dilution)

    # First row (t_sel=0) should be unchanged; rest should be offset
    assert np.isclose(y_obs2[0], raw2[0])
    assert np.allclose(y_obs2[1:], raw2[1:] - np.log(dilution))

def test_t_pre_injection_multiple_sel(example_calibration_df, tmp_path):
    """Verify that -t_pre points are injected for EVERY condition_sel."""
    # Create two rows with same pre but different sel conditions
    df = pd.DataFrame({
        "ln_cfu": [10.0, 11.0],
        "ln_cfu_std": [0.1, 0.1],
        "t_pre": [5.0, 5.0],
        "condition_pre": ["bg", "bg"],
        "t_sel": [10.0, 10.0],
        "condition_sel": ["sel1", "sel2"], # Different sel
        "replicate": [1, 1],
        "genotype": ["wt", "wt"],
        "titrant_name": ["t1", "t1"],
        "titrant_conc": [1.0, 1.0],
        "theta": [0.5, 0.5],
        "theta_std": [0.01, 0.01],
        "censored": [False, False]
    })
    
    with patch("tfscreen.calibration.calibrate.run_least_squares") as mock_rls, \
         patch("tfscreen.calibration.calibrate.predict_with_error") as mock_predict, \
         patch("tfscreen.calibration.calibrate.write_calibration"), \
         patch("tfscreen.calibration.calibrate.read_calibration"), \
         patch("tfscreen.calibration.calibrate._fit_theta"):
        
        # setup mocks to return enough values
        fm = setup_calibration(df)
        n_params = fm.num_params
        mock_rls.return_value = (np.zeros(n_params), np.zeros(n_params), np.eye(n_params), {})
        mock_predict.return_value = (np.zeros(2), np.zeros(2))
        
        out_file = str(tmp_path / "out.json")
        _, pred_df, _ = calibrate(df, out_file)
        
        # We started with 2 rows. 
        # We expect 2 injected -t_pre rows (one for sel1, one for sel2).
        # Total rows should be 4.
        assert len(pred_df) == 4
        
        # Verify injected rows
        injected = pred_df[pred_df["t_sel"] < 0]
        assert len(injected) == 2
        assert "sel1" in injected["condition_sel"].values
        assert "sel2" in injected["condition_sel"].values

def test_ln_cfu0_grouping(example_calibration_df):
    """Verify ln_cfu_0 is now grouped by (genotype, replicate, condition_pre)."""
    
    # Create a dataframe where the same (genotype, replicate) has two condition_pre
    df = example_calibration_df.copy()
    df.loc[2, "condition_pre"] = "other_bg"
    df.loc[3, "condition_pre"] = "other_bg"
    
    df = _prep_calibration_df(df)
    _, _, _, param_df = _build_calibration_X(df)
    
    lnA0_params = param_df[param_df["param_class"] == "ln_cfu_0"]
    
    # We should have 3 unique ln_cfu_0 parameters:
    # 1. (wt, 1, background)
    # 2. (mutant, 1, background)
    # 3. (wt, 1, other_bg) -- mutant was row 3, so now it's (mutant, 1, other_bg)
    # Wait, in the original df:
    # row 0: (wt, 1, background)
    # row 1: (mutant, 1, background)
    # row 2: (wt, 1, background) -> changed to other_bg
    # row 3: (mutant, 1, background) -> changed to other_bg
    
    # So now we have:
    # (wt, 1, background)
    # (mutant, 1, background)
    # (wt, 1, other_bg)
    # (mutant, 1, other_bg)
    # Actually, row 0 and row 1 have same names, row 2 and 3 have different names.
    # Total 4.
    assert len(lnA0_params) == 4
    
    # Check names
    names = lnA0_params["patsy_name"].tolist()
    assert any("background" in n for n in names)
    assert any("other_bg" in n for n in names)
