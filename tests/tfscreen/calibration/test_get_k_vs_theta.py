import pytest
import pandas as pd
import numpy as np

from tfscreen.calibration.get_k_vs_theta import get_k_vs_theta

def test_get_k_vs_theta_success(mocker):
    """
    Tests that slopes and intercepts are correctly looked up for valid keys.
    """
    # 1. ARRANGE
    # Create a fake linear_df to be returned by the mocked reader
    mock_linear_df = pd.DataFrame({
        "m": [0.5, 0.6, 0.7],
        "b": [0.1, 0.2, 0.3],
    }, index=pd.MultiIndex.from_tuples([
        ("wt", "iptg"), ("mut1", "iptg"), ("wt", "lactose")
    ], names=['condition', 'titrant_name']))
    
    mock_cal_data = {"linear_df": mock_linear_df}
    mocker.patch('tfscreen.calibration.get_k_vs_theta.read_calibration', return_value=mock_cal_data)

    # Define inputs, including a duplicate to test proper handling
    condition = np.array(["wt", "mut1", "wt"])
    titrant_name = np.array(["iptg", "iptg", "iptg"])

    # 2. ACT
    slopes, intercepts = get_k_vs_theta(condition, titrant_name, "dummy_path.json")

    # 3. ASSERT
    assert isinstance(slopes, np.ndarray)
    assert isinstance(intercepts, np.ndarray)

    # Expected values are looked up from the mock_linear_df in order
    expected_slopes = np.array([0.5, 0.6, 0.5])
    expected_intercepts = np.array([0.1, 0.2, 0.1])

    np.testing.assert_array_equal(slopes, expected_slopes)
    np.testing.assert_array_equal(intercepts, expected_intercepts)

def test_get_k_vs_theta_raises_key_error(mocker):
    """
    Tests that a KeyError is raised if a key does not exist in the index.
    """
    # 1. ARRANGE
    # Use the same mock data as the success test
    mock_linear_df = pd.DataFrame({
        "m": [0.5], "b": [0.1]
    }, index=pd.MultiIndex.from_tuples([("wt", "iptg")]))
    
    mock_cal_data = {"linear_df": mock_linear_df}
    mocker.patch('tfscreen.calibration.get_k_vs_theta.read_calibration', return_value=mock_cal_data)

    # Define inputs where one key is invalid
    condition = np.array(["wt", "non_existent_condition"])
    titrant_name = np.array(["iptg", "iptg"])

    # 2. ACT & 3. ASSERT
    # Use pytest.raises to assert that a KeyError occurs
    with pytest.raises(KeyError):
        get_k_vs_theta(condition, titrant_name, "dummy_path.json")