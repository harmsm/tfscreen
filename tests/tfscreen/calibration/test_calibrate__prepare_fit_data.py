import pytest
import pandas as pd
import numpy as np

from tfscreen.calibration.calibrate import _prepare_fit_data


def test_prepare_fit_data_standard_case():
    """
    Tests the data preparation with standard, positive inputs.
    """
    # 1. ARRANGE
    df = pd.DataFrame({
        "cfu_per_mL":     [100.0, 1000.0],
        "cfu_per_mL_std": [10.0, 50.0],
        "theta":          [0.1, 0.9],
        "titrant_conc":   [1.0, 10.0],
        "pre_time":       [2.0, 2.0],
        "time":           [8.0, 8.0],
    })

    # 2. ACT
    result = _prepare_fit_data(df)

    # 3. ASSERT
    assert isinstance(result, dict)
    assert all(k in result for k in ["y", "y_var", "theta", "titrant_conc"])

    # Check log-transformed y-values (ln(cfu))
    expected_y = np.log([100.0, 1000.0])
    np.testing.assert_allclose(result["y"], expected_y)

    # Check log-transformed y-variance
    # cfu_var = [10^2, 50^2] = [100, 2500]
    # cfu^2 = [100^2, 1000^2] = [10000, 1000000]
    # ln_cfu_var = [100/10000, 2500/1000000] = [0.01, 0.0025]
    expected_y_var = np.array([0.01, 0.0025])
    np.testing.assert_allclose(result["y_var"], expected_y_var)
    
    # Check that other arrays are passed through correctly
    np.testing.assert_array_equal(result["theta"], df["theta"].to_numpy())
    np.testing.assert_array_equal(result["time"], df["time"].to_numpy())

def test_prepare_fit_data_zero_and_negative_inputs():
    """
    Tests the data sanitization logic for zero or negative CFU and variance.
    """
    # 1. ARRANGE
    df = pd.DataFrame({
        "cfu_per_mL":     [100.0, 0.0, -50.0], # Includes zero and negative
        "cfu_per_mL_std": [10.0, 5.0, 0.0],   # Includes zero
        "theta":          [0.1, 0.5, 0.9],
        "titrant_conc":   [1.0, 10.0, 100.0],
        "pre_time":       [2.0, 2.0, 2.0],
        "time":           [8.0, 8.0, 8.0],
    })

    # 2. ACT
    result = _prepare_fit_data(df)

    # 3. ASSERT
    # Check that non-positive CFU values were set to 1 before np.log
    # Original cfu -> [100, 0, -50] -> Sanitized cfu -> [100, 1, 1]
    expected_y = np.log([100.0, 1.0, 1.0])
    np.testing.assert_allclose(result["y"], expected_y)

    # Check that non-positive variance was handled correctly
    # Original std -> [10, 5, 0] -> Original var -> [100, 25, 0]
    # Sanitized var -> [100, 25, 1e-9]
    # Sanitized cfu**2 -> [100**2, 1**2, 1**2] = [10000, 1, 1]
    # ln_cfu_var = [100/10000, 25/1, 1e-9/1] = [0.01, 25, 1e-9]
    expected_y_var = np.array([0.01, 25.0, 1e-9])
    np.testing.assert_allclose(result["y_var"], expected_y_var)