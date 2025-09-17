import pytest
import pandas as pd
import numpy as np

from tfscreen.calibration.get_wt_theta import get_wt_theta

@pytest.fixture
def mock_cal_data(mocker):
    """Mocks read_calibration to return a fixed calibration dictionary."""
    cal_data = {
        "theta_param": {
            "iptg": np.array([1.0, -0.9, 10.0, 2.0]),
        }
    }
    mocker.patch('tfscreen.calibration.get_wt_theta.read_calibration', return_value=cal_data)

@pytest.mark.parametrize(
    "override_K, override_n, expected_params",
    [
        # Case 1: No overrides
        (None, None, np.array([[1.0, -0.9, 10.0, 2.0]])),
        # Case 2: Override K
        (50.0, None, np.array([[1.0, -0.9, 50.0, 2.0]])),
        # Case 3: Override n
        (None, 3.5, np.array([[1.0, -0.9, 10.0, 3.5]])),
        # Case 4: Override both
        (99.0, 4.0, np.array([[1.0, -0.9, 99.0, 4.0]])),
    ],
    ids=["no_override", "override_K", "override_n", "override_both"]
)
def test_get_wt_theta(mocker, mock_cal_data, override_K, override_n, expected_params):
    """
    Tests theta calculation with and without parameter overrides.
    """
    # 1. ARRANGE
    # Mock the hill_model to return the parameter array it was called with.
    # This lets us verify the override logic directly.
    mock_hill_model = mocker.patch(
        'tfscreen.calibration.get_wt_theta._models.hill_model',
        side_effect=lambda params, concs: params.T # Return transposed params
    )
    
    titrant_name = np.array(["iptg"])
    titrant_conc = np.array([100.0])

    # 2. ACT
    # The result will be the parameter array that was passed to the mock model
    result_params = get_wt_theta(
        titrant_name, titrant_conc, "dummy_path.json",
        override_K=override_K, override_n=override_n
    )

    # 3. ASSERT
    # Check that the parameters passed to the model match our expectations
    np.testing.assert_array_equal(result_params, expected_params)
    
    # Also verify the model was called with the correct concentrations
    call_args, _ = mock_hill_model.call_args
    np.testing.assert_array_equal(call_args[1], titrant_conc)