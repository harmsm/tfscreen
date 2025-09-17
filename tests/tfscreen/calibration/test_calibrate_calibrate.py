import pytest
import pandas as pd
import numpy as np

from tfscreen.calibration.calibrate import calibrate

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def mock_dependencies(mocker):
    """A fixture to mock the internal fitting dependencies of the calibrate function."""
    # We NO LONGER mock write_calibration or read_calibration.
    # We still mock the expensive/complex internal functions.
    
    # Mock the core fitting functions
    param_df = pd.DataFrame({
        "m_est": [0.5, 0.6], 
        "b_est": [0.1, 0.2]
    }, index=pd.MultiIndex.from_tuples([("wt", "iptg"), ("mut1", "iptg")],
                                       names=["condition", "titrant_name"]))
    
    # Mock the return values for the two main fitting steps
    mock_fit_linear = mocker.patch('tfscreen.calibration.calibrate._fit_linear_model', return_value=(
        param_df, {"iptg": np.array([0.01])}, "pred_df_mock", "A0_df_mock"
    ))
    mock_fit_theta = mocker.patch('tfscreen.calibration.calibrate._fit_theta', return_value={"iptg": np.array([1, -0.9, 10, 2])})

    # Return the mocks we might want to inspect
    return {
        "fit_linear": mock_fit_linear,
        "fit_theta": mock_fit_theta,
    }

@pytest.fixture
def valid_df():
    """Provides a minimal, valid DataFrame that passes all checks."""
    return pd.DataFrame({
        "replicate": ["r1"], "pre_condition": ["wt"], "pre_time": [2],
        "condition": ["background"], "time": [8], "titrant_name": ["iptg"],
        "titrant_conc": [10], "cfu_per_mL": [1e7], "cfu_per_mL_std": [1e6],
        "theta": [0.1], "theta_std": [0.01]
    })

def test_calibrate_success_path(mock_dependencies, valid_df, tmp_path):
    """
    Tests the successful, end-to-end execution path, including the
    file write and read round trip.
    """
    # 1. ARRANGE
    # Use the tmp_path fixture to create a safe path for the output file.
    output_file = tmp_path / "test_output.json"

    # 2. ACT
    # The function now returns three values; we only need the first for this test.
    result_dict, _, _ = calibrate(valid_df, str(output_file), bg_order=1)

    # 3. ASSERT
    # Check that the internal fitting functions were called as expected.
    mock_dependencies["fit_linear"].assert_called_once()
    mock_dependencies["fit_theta"].assert_called_once()
    
    # Assert that the file was actually created.
    assert output_file.exists()
    
    # Now, verify the content of the RETURNED dictionary. This dictionary
    # has successfully gone through the write -> read cycle.
    assert "m" in result_dict
    assert "b" in result_dict
    assert "theta_param" in result_dict
    assert "bg_model_param" in result_dict
    assert "linear_df" in result_dict # Added by read_calibration

    # Check that data from the mocks is present in the final, processed dictionary.
    assert result_dict["m"][("wt", "iptg")] == 0.5
    
    # Verify that data from the theta fit is correct (read_calibration converts lists to arrays)
    expected_theta = np.array([1, -0.9, 10, 2])
    np.testing.assert_array_equal(result_dict["theta_param"]["iptg"], expected_theta)

def test_calibrate_raises_error_for_missing_columns(valid_df):
    """Verifies that a ValueError is raised if a column is missing."""
    invalid_df = valid_df.drop(columns=["theta"])
    
    with pytest.raises(ValueError, match="Not all required columns"):
        calibrate(invalid_df, "output.json")

def test_calibrate_raises_error_for_missing_background(valid_df):
    """Verifies a ValueError for titrants missing a 'background' condition."""
    # Add a new titrant that does NOT have a background condition
    invalid_df = pd.concat([
        valid_df,
        pd.DataFrame.from_records([{**valid_df.iloc[0], 
                                    "titrant_name": "lactose", 
                                    "condition": "wt"}])
    ], ignore_index=True)
    
    with pytest.raises(ValueError, match="All unique titrant_name values must have a 'background' condition"):
        calibrate(invalid_df, "output.json")

def test_calibrate_raises_error_for_invalid_bg_order(valid_df):
    """Verifies a ValueError for negative bg_order."""
    with pytest.raises(ValueError, match="bg_order must be >= 0"):
        calibrate(valid_df, "output.json", bg_order=-1)