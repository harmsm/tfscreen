import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from unittest.mock import patch

# Import the function to be tested
from tfscreen.process_raw.od600_to_cfu import od600_to_cfu

# ------------------------
# Pytest Fixtures
# ------------------------

@pytest.fixture
def fx_valid_constants() -> dict:
    """
    Provides a dictionary of valid calibration constants.
    The values are chosen to make the expected calculations simple and easy to verify.
    - CFU vs OD is linear: cfu = 10 + 100 * od
    - J^T C J^T term is linear: term = 1 + 1 * od
    - OD error is 10%
    """
    return {
        "OD600_MEAS_THRESHOLD": 0.1,
        "A_CFU": 10.0,
        "B_CFU": 100.0,
        "C_CFU": 0.0,         # Makes the main calculation linear
        "P_JCJT_CFU": 1.0,
        "Q_JCJT_CFU": 1.0,
        "R_JCJT_CFU": 0.0,      # Makes the variance term linear
        "OD600_PCT_STD": 0.1  # 10% standard deviation in OD measurement
    }

# ------------------------
# Test od600_to_cfu
# ------------------------

def test_od600_to_cfu_with_array_input(fx_valid_constants):
    """
    Tests the function with a NumPy array containing values above, at, and
    below the detection threshold.
    """
    od_values = np.array([0.05, 0.1, 0.2]) # Below, at, and above threshold

    cfu, cfu_std, detectable = od600_to_cfu(od_values, fx_valid_constants)

    # --- Expected values based on fx_valid_constants ---
    # For OD = 0.05 (below threshold, capped at 0.1 for calculation):
    # detectable = False
    # cfu_est = 10 + 100 * 0.1 = 20.0
    # cfu_est_std_2 = (1 + 1*0.1)**2 = 1.1**2 = 1.21
    # dy_dx_2 = (100)**2 = 10000
    # od600_std_2 = (0.1 * 0.1)**2 = 0.0001
    # cfu_std = sqrt(1.21 + 10000 * 0.0001) = sqrt(2.21)
    
    # For OD = 0.1 (at threshold):
    # detectable = True
    # cfu_est = 10 + 100 * 0.1 = 20.0
    # cfu_std = sqrt(2.21)
    
    # For OD = 0.2 (above threshold):
    # detectable = True
    # cfu_est = 10 + 100 * 0.2 = 30.0
    # cfu_est_std_2 = (1 + 1*0.2)**2 = 1.2**2 = 1.44
    # od600_std_2 = (0.1 * 0.2)**2 = 0.0004
    # cfu_std = sqrt(1.44 + 10000 * 0.0004) = sqrt(5.44)

    expected_cfu = np.array([20.0, 20.0, 30.0])
    expected_std = np.sqrt([2.21, 2.21, 5.44])
    expected_detectable = np.array([False, True, True])

    # --- Assertions ---
    assert isinstance(cfu, np.ndarray)
    assert_allclose(cfu, expected_cfu)
    assert_allclose(cfu_std, expected_std)
    assert_array_equal(detectable, expected_detectable)


def test_od600_to_cfu_with_scalar_input(fx_valid_constants):
    """
    Tests that a single scalar input produces single scalar outputs.
    """
    od_value = 0.2

    cfu, cfu_std, detectable = od600_to_cfu(od_value, fx_valid_constants)
    
    expected_cfu = 30.0
    expected_std = np.sqrt(5.44)
    expected_detectable = True

    # --- Assertions ---
    assert np.isscalar(cfu)
    assert np.isscalar(cfu_std)
    assert np.isscalar(detectable)
    assert cfu == expected_cfu
    assert pytest.approx(cfu_std) == expected_std
    assert detectable == expected_detectable


@patch('tfscreen.process_raw.od600_to_cfu.read_yaml')
def test_od600_to_cfu_with_yaml_path(mock_read_yaml, fx_valid_constants):
    """
    Tests that the function can correctly read constants from a YAML file path.
    """
    mock_read_yaml.return_value = fx_valid_constants
    od_value = 0.2
    
    od600_to_cfu(od_value, "path/to/constants.yaml")
    
    # Assert that read_yaml was called with the provided path
    mock_read_yaml.assert_called_once_with("path/to/constants.yaml")


def test_od600_to_cfu_missing_constants():
    """
    Tests that a ValueError is raised if the constants dictionary is missing
    required keys.
    """
    # Create an incomplete dictionary
    bad_constants = {"OD600_MEAS_THRESHOLD": 0.1, "A_CFU": 10.0}