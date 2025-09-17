# tests/model/test_forward_model.py

import pytest
import numpy as np

# Import the function to be tested
from tfscreen.calibration.calibrate import _calculate_log_population

@pytest.fixture
def model_inputs():
    """Provides a default set of inputs for the model function."""
    # Global parameter vector
    params = np.array([
        10.0,  # lnA0, idx 0
        0.1,   # b_pre, idx 1
        -0.2,  # m_pre, idx 2
        0.5,   # b, idx 3
        -0.8,  # m, idx 4
        0.05,  # k_bg_c0, idx 5 (poly const)
        0.01   # k_bg_c1, idx 6 (poly slope)
    ])
    
    # Per-observation data arrays
    base_data = {
        "theta": np.array([0.1, 0.9, 0.5]),
        "titrant_conc": np.array([1, 10, 5]),
        "pre_time": np.array([2, 2, 2]),
        "time": np.array([8, 8, 8]),
        "not_bg": np.array([True, True, False]), # Note the third is background
        "A0_idx": np.array([0, 0, 0]),
        "b_pre_idx": np.array([1, 1, 1]),
        "m_pre_idx": np.array([2, 2, 2]),
        "b_idx": np.array([3, 3, 3]),
        "m_idx": np.array([4, 4, 4]),
        "bg_param_idx": np.array([[5, 6], [5, 6], [5, 6]]),
    }
    return {"params": params, **base_data}


def test_calculate_log_population_calculation(mocker, model_inputs):
    """
    Test the full calculation for both background and non-background points.
    """
    # 1. ARRANGE
    # Mock the external simple_poly function
    mock_simple_poly = mocker.patch('tfscreen.calibration.calibrate.simple_poly')
    
    # Based on our inputs:
    # titrant_conc = [1, 10, 5], time+pre_time = 10
    # bg_params are [0.05, 0.01]
    # We expect k_bg_t = (0.05 + 0.01*titrant_conc) * 10
    k_bg_t_expected = np.array([
        (0.05 + 0.01 * 1) * 10,   # 0.6
        (0.05 + 0.01 * 10) * 10,  # 1.5
        (0.05 + 0.01 * 5) * 10,   # 1.0
    ])
    mock_simple_poly.return_value = k_bg_t_expected / 10 # Since it's multiplied by time

    # Expected calculation for the first (non-bg) point:
    # lnA0 = 10.0
    # k_bg_t = 0.6
    # k_pre_t = (b_pre + m_pre*theta)*pre_time = (0.1 + -0.2*0.1)*2 = 0.16
    # k_t = (b + m*theta)*time = (0.5 + -0.8*0.1)*8 = 3.36
    # not_bg = 1
    # lnA[0] = 10.0 + 0.6 + 1 * (0.16 + 3.36) = 14.12
    
    # Expected calculation for the third (bg) point:
    # lnA0 = 10.0
    # k_bg_t = 1.0
    # not_bg = 0, so perturbation term is 0
    # lnA[2] = 10.0 + 1.0 + 0 = 11.0

    # 2. ACT
    result = _calculate_log_population(**model_inputs)

    # 3. ASSERT
    assert result.shape == (3,)
    # Check the first point (non-background)
    assert np.isclose(result[0], 14.12)
    # Check the third point (background)
    assert np.isclose(result[2], 11.0)


def test_calculate_log_population_background_only(mocker, model_inputs):
    """
    Test that the perturbation is zeroed out when not_bg is False.
    """
    # 1. ARRANGE
    # Mock simple_poly to return a constant background effect
    mocker.patch(
        'tfscreen.calibration.calibrate.simple_poly',
        return_value=np.array([0.1, 0.1, 0.1])
    )
    
    # Modify inputs so all points are background
    model_inputs["not_bg"] = np.array([False, False, False])
    
    # Expected result: lnA = lnA0 + k_bg_t
    # lnA0 = 10.0
    # k_bg_t = 0.1 * (pre_time + time) = 0.1 * 10 = 1.0
    expected_lnA = 10.0 + 1.0

    # 2. ACT
    result = _calculate_log_population(**model_inputs)

    # 3. ASSERT
    np.testing.assert_allclose(result, expected_lnA)