import pytest
import pandas as pd
from unittest.mock import Mock

from tfscreen.calibration.calibrate import _fit_linear_model


def test_fit_linear_model_orchestration(mocker):
    """
    Tests that the orchestrator function correctly calls its helpers
    and passes data between them.
    """
    # 1. ARRANGE: Mock all dependencies and their return values
    
    # Mock the return value of _build_fit_setup
    mock_fit_setup = Mock()
    mock_fit_setup.initial_guesses = "initial_guesses_array"
    mock_fit_setup.get_args_tuple.return_value = ("dummy_args_tuple",)
    mock_build_fit_setup = mocker.patch(
        'tfscreen.calibration.calibrate._build_fit_setup', return_value=mock_fit_setup
    )

    # Mock the return value of _prepare_fit_data
    mock_fit_data = {
        "y": "y_data", "y_var": "y_var_data", "theta": "theta_data",
        "titrant_conc": "titrant_conc_data", "pre_time": "pre_time_data",
        "time": "time_data"
    }
    mock_prepare_fit_data = mocker.patch(
        'tfscreen.calibration.calibrate._prepare_fit_data', return_value=mock_fit_data
    )

    # Mock the return value of the fitter
    mock_fitter_results = ("final_params", "std_errors", "cov_matrix", "fit_object")
    mock_run_least_squares = mocker.patch(
        'tfscreen.calibration.calibrate.run_least_squares', return_value=mock_fitter_results
    )

    # Mock the return value of _parse_fit_results
    mock_parsed_results = Mock()
    mock_parsed_results.linear_model_df = "final_lm_df"
    mock_parsed_results.bg_model_param = "final_bg_dict"
    mock_parsed_results.pred_df = "final_pred_df"
    mock_parsed_results.A0_df = "final_a0_df"
    mock_parse_fit_results = mocker.patch(
        'tfscreen.calibration.calibrate._parse_fit_results', return_value=mock_parsed_results
    )
    
    # Dummy inputs for the main function
    sample_df = pd.DataFrame({'a': [1]})
    bg_guesses = [0.1, 0.01]
    a0_guess = 15.0
    
    # Note: When using in your package, change 'tfscreen.calibration.calibrate' to your module's path
    # for all mocker.patch calls.
    
    # 2. ACT
    result_tuple = _fit_linear_model(sample_df, bg_guesses, a0_guess)
    
    # 3. ASSERT: Verify the calls and the data flow
    
    # Verify setup and data prep were called correctly
    mock_build_fit_setup.assert_called_once_with(sample_df, bg_guesses, a0_guess)
    mock_prepare_fit_data.assert_called_once_with(sample_df)
    
    # Verify get_args_tuple was called with the data from _prepare_fit_data
    mock_fit_setup.get_args_tuple.assert_called_once_with(
        "theta_data", "titrant_conc_data", "pre_time_data", "time_data"
    )
    
    # Verify the fitter was called with the outputs of the first two steps
    mock_run_least_squares.assert_called_once_with(
        mocker.ANY, # The _calculate_log_population function object
        "y_data",
        "y_var_data",
        guesses="initial_guesses_array",
        args=("dummy_args_tuple",)
    )
    
    # Verify the parser was called with the results from the fitter and setup
    mock_parse_fit_results.assert_called_once_with(
        "final_params", "std_errors", "cov_matrix",
        mock_fit_setup,
        mock_fit_data
    )
    
    # Verify the final returned tuple contains the results from the parser
    assert result_tuple[0] == "final_lm_df"
    assert result_tuple[1] == "final_bg_dict"
    assert result_tuple[2] == "final_pred_df"
    assert result_tuple[3] == "final_a0_df"