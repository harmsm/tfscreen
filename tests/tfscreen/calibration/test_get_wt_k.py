import pytest
import numpy as np

from tfscreen.calibration.get_wt_k import get_wt_k

@pytest.mark.parametrize(
    "provided_theta, should_call_get_theta",
    [
        (np.array([0.1, 0.9]), False), # Case 1: theta is provided
        (None, True),                 # Case 2: theta is not provided
    ],
    ids=["theta_is_provided", "theta_is_none"]
)
def test_get_wt_k(mocker, provided_theta, should_call_get_theta):
    """
    Tests that the orchestrator correctly calls helpers and combines results.
    """
    # 1. ARRANGE: Mock all dependencies
    mocker.patch('tfscreen.calibration.get_wt_k.read_calibration', return_value="mock_cal_dict")
    
    mock_bg = mocker.patch('tfscreen.calibration.get_wt_k.get_background', return_value=np.array([0.1, 0.1]))
    mock_kvt = mocker.patch('tfscreen.calibration.get_wt_k.get_k_vs_theta', return_value=(
        np.array([0.5, 0.5]), np.array([0.02, 0.02])
    ))
    mock_theta = mocker.patch('tfscreen.calibration.get_wt_k.get_wt_theta', return_value=np.array([0.25, 0.75]))
    
    # Define inputs
    condition = np.array(["wt", "wt"])
    titrant_name = np.array(["iptg", "iptg"])
    titrant_conc = np.array([10.0, 100.0])

    # 2. ACT
    result = get_wt_k(
        condition, titrant_name, titrant_conc,
        calibration_data="dummy_path.json",
        theta=provided_theta
    )

    # 3. ASSERT
    # Verify that the core helpers were called correctly
    mock_bg.assert_called_once_with(titrant_name, titrant_conc, "mock_cal_dict")
    mock_kvt.assert_called_once_with(condition, titrant_name, "mock_cal_dict")
    
    # Determine which theta was used for the final calculation
    if should_call_get_theta:
        mock_theta.assert_called_once_with(titrant_name, titrant_conc, "mock_cal_dict")
        theta_used = mock_theta.return_value
    else:
        mock_theta.assert_not_called()
        theta_used = provided_theta
        
    # Verify the final mathematical combination is correct
    background = mock_bg.return_value
    slopes, intercepts = mock_kvt.return_value
    
    expected_result = background + intercepts + slopes * theta_used
    np.testing.assert_allclose(result, expected_result)