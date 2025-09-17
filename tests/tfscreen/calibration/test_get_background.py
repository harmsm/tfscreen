import pytest
import pandas as pd
import numpy as np

from tfscreen.calibration.get_background import get_background

def test_get_background(mocker):
    """
    Tests that parameters are correctly looked up and passed to the model.
    """
    # 1. ARRANGE
    # Create a fake calibration dictionary that the mock will return
    mock_cal_data = {
        "bg_model_param": {
            "iptg": np.array([0.1, 0.01]),
            "lactose": np.array([0.2, 0.02]),
        }
    }
    # Mock the file/dict reader to return our fake data
    mocker.patch('tfscreen.calibration.get_background.read_calibration', return_value=mock_cal_data)
    
    # Mock the polynomial model function
    mock_simple_poly = mocker.patch('tfscreen.calibration.get_background.simple_poly', return_value="success")

    # Define the inputs for the function
    # Note the duplicate "iptg" to ensure the lookup works for repeated values
    titrant_name = np.array(["iptg", "lactose", "iptg"])
    titrant_conc = np.array([1.0, 10.0, 100.0])

    # 2. ACT
    result = get_background(titrant_name, titrant_conc, "dummy_path.json")

    # 3. ASSERT
    # Check that the function returned the value from our mocked model
    assert result == "success"

    # The most important check: verify that simple_poly was called with the
    # correctly assembled parameter array.
    mock_simple_poly.assert_called_once()
    call_args, _ = mock_simple_poly.call_args
    
    # Expected bg_params array based on the input titrant_name array:
    # First row is params for 'iptg', second for 'lactose', third for 'iptg'
    expected_bg_params = np.array([
        [0.1, 0.01],    # for 'iptg'
        [0.2, 0.02],    # for 'lactose'
        [0.1, 0.01]     # for 'iptg'
    ])
    
    # The function transposes the params before calling, so we check arg[0]
    # against the expected transpose.
    np.testing.assert_array_equal(call_args[0], expected_bg_params.T)
    
    # Check that the concentrations were passed through correctly as arg[1]
    np.testing.assert_array_equal(call_args[1], titrant_conc)