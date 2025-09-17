# test_fit_theta.py

import pytest

from tfscreen.calibration.calibrate import _fit_theta

import pandas as pd
import numpy as np

# Dummy function to act as our model
def hill_model(params, x):
    pass

@pytest.fixture
def mock_run_least_squares(mocker):
    """Pytest fixture to mock the run_least_squares function."""
    # Create a mock function
    mock_func = mocker.Mock()

    # Configure the mock to return a predictable result.
    # The return value mimics (params, pcov, infodict, mesg)
    mock_func.return_value = (np.array([1.0, -0.9, 5.0, 2.1]), None, None, None)
    return mock_func

def test_fit_theta_single_titrant(mocker):
    """
    Test fitting with a DataFrame containing a single titrant using mocker.
    """
    # 1. ARRANGE: Mock the dependencies.
    # We patch the functions in the 'analysis' module where _fit_theta will find them.
    mocked_params = np.array([1.0, -0.9, 5.0, 2.1])
    mock_run_least_squares = mocker.patch(
        'tfscreen.calibration.calibrate.run_least_squares',
        return_value=(mocked_params, None, None, None)
    )
    mock_hill_model = mocker.patch('tfscreen.calibration.calibrate.hill_model')

    # Create the input DataFrame, same as before.
    data = {
        "titrant_name": ["iptg"] * 5,
        "titrant_conc": np.array([0.1, 1, 10, 100, 1000]),
        "theta": np.array([0.95, 0.9, 0.5, 0.1, 0.05]),
        "theta_std": np.array([0.05] * 5),
    }
    df = pd.DataFrame(data)

    # 2. ACT: Call the real function. It will now use the mocks internally.
    result = _fit_theta(df)

    # 3. ASSERT: The assertions are nearly identical to before.
    # Check the final output dictionary
    assert list(result.keys()) == ["iptg"]
    np.testing.assert_array_equal(result["iptg"], mocked_params)

    # Verify that run_least_squares was called exactly once
    mock_run_least_squares.assert_called_once()

    # You can also perform detailed checks on how the mock was called
    call_args, call_kwargs = mock_run_least_squares.call_args
    
    # Check the positional arguments: (model, y, y_std)
    assert call_args[0] is mock_hill_model # It was passed the mocked model
    np.testing.assert_array_equal(call_args[1], df["theta"].to_numpy()) # y
    np.testing.assert_array_equal(call_args[2], df["theta_std"].to_numpy()) # y_std

    # Check the keyword arguments: guesses and args=(x,)
    expected_guesses = np.array([0.95, -0.9, 10.0, 2.0])
    np.testing.assert_allclose(call_kwargs['guesses'], expected_guesses)
    np.testing.assert_array_equal(call_kwargs['args'][0], df["titrant_conc"].to_numpy())

def test_fit_theta_multiple_titrants(mocker):
    """
    Test fitting with a DataFrame containing two different titrants.
    """

    mocked_params = np.array([1.0, -0.9, 5.0, 2.1])
    mock_run_least_squares = mocker.patch(
        'tfscreen.calibration.calibrate.run_least_squares',
        return_value=(mocked_params, None, None, None)
    )

    # 1. ARRANGE
    data = {
        "titrant_name": ["iptg", "iptg", "lactose", "lactose"],
        "titrant_conc": [1, 10, 100, 1000],
        "theta": [0.9, 0.1, 0.8, 0.2],
        "theta_std": [0.05, 0.05, 0.05, 0.05],
    }
    df = pd.DataFrame(data)

    # Configure the mock to return different values on subsequent calls
    mock_params1 = np.array([0.9, -0.8, 5.0, 2.0])
    mock_params2 = np.array([0.8, -0.6, 500, 1.5])
    mock_run_least_squares.side_effect = [
        (mock_params1, None, None, None),
        (mock_params2, None, None, None),
    ]

    # 2. ACT
    result = _fit_theta(df) #, mock_run_least_squares, hill_model)

    # 3. ASSERT
    # Check that the function was called twice
    assert mock_run_least_squares.call_count == 2

    # Check the output dictionary
    assert sorted(list(result.keys())) == ["iptg", "lactose"]
    np.testing.assert_array_equal(result["iptg"], mock_params1)
    np.testing.assert_array_equal(result["lactose"], mock_params2)

def test_fit_theta_empty_dataframe(mock_run_least_squares):
    """
    Test that an empty DataFrame results in an empty dictionary without errors.
    """
    # 1. ARRANGE
    df = pd.DataFrame({
        "titrant_name": [],
        "titrant_conc": [],
        "theta": [],
        "theta_std": [],
    })

    # 2. ACT
    result = _fit_theta(df) #, mock_run_least_squares, hill_model)

    # 3. ASSERT
    assert result == {}
    mock_run_least_squares.assert_not_called()