
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from tfscreen.analysis.cat_response.cat_fit import cat_fit

# Mock data for testing
@pytest.fixture
def mock_data():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    y_std = np.array([0.1, 0.1, 0.1, 0.1])
    return x, y, y_std

@pytest.fixture
def mock_predict_side_effect():
    def side_effect(model_func, params, cov, args=None):
        if args and len(args) > 0:
            x_pred = args[0]
            return np.zeros(len(x_pred)), np.zeros(len(x_pred))
        return np.array([]), np.array([])
    return side_effect

def test_insufficient_data():
    """Test behavior when insufficient data is provided."""
    x = np.array([1.0])
    y = np.array([2.0])
    y_std = np.array([0.1])
    
    # We pass x_pred explicit to avoid needing xfill mock just for this
    flat_output, pred_df = cat_fit(x, y, y_std, x_pred=np.array([1.0, 2.0]), models_to_run=["linear"])
    
    assert flat_output['status'] == "missing"
    assert flat_output['best_model'] == "None"
    assert np.isnan(flat_output['best_model_R2'])
    
    # Check that model entries exist and serve nans
    assert "R2|linear" in flat_output
    assert np.isnan(flat_output["R2|linear"])
    
    # Check pred_df
    assert len(pred_df) == 2 # 2 points in x_pred * 1 model
    assert np.all(np.isnan(pred_df['y']))

@patch("tfscreen.analysis.cat_response.cat_fit.MODEL_LIBRARY")
@patch("tfscreen.analysis.cat_response.cat_fit.run_least_squares")
@patch("tfscreen.analysis.cat_response.cat_fit.predict_with_error")
def test_fit_linear_success(mock_predict, mock_run_ls, mock_library, mock_data, mock_predict_side_effect):
    """Test successful fit for a linear model."""
    x, y, y_std = mock_data
    mock_predict.side_effect = mock_predict_side_effect
    
    # Setup Mock Model
    # Use imperfect fit so ss_res > 0 and AIC is finite
    mock_model_func = MagicMock(return_value=y + 0.001) 
    mock_guess_func = MagicMock(return_value=np.array([1.0, 0.0]))
    
    mock_library.__getitem__.return_value = {
        "model_func": mock_model_func,
        "guess_func": mock_guess_func,
        "param_names": ["m", "b"],
        "bounds": ([-np.inf, -np.inf], [np.inf, np.inf])
    }
    mock_library.keys.return_value = ["test_model"]
    
    # Setup run_least_squares return
    fit_obj = MagicMock()
    fit_obj.success = True
    mock_run_ls.return_value = (np.array([2.0, 0.0]), np.array([0.1, 0.1]), np.eye(2), fit_obj)
    
    x_pred = np.array([1.0, 2.0, 3.0, 4.0])
    flat_output, pred_df = cat_fit(x, y, y_std, x_pred=x_pred, models_to_run=["test_model"])
    
    assert flat_output['status'] == "success"
    assert flat_output['best_model'] == "test_model"
    # R2 logic: 1 - ss_res/ss_tot. ss_res > 0 now. R2 should be close to 1 but not 1.0 (maybe)
    # y=[2,4,6,8], mean=5. ss_tot = 9+1+1+9 = 20.
    # ss_res = sum(0.001^2) = 4*1e-6. small.
    # R2 approx 1.
    assert flat_output['R2|test_model'] > 0.99 
    assert flat_output['AIC_weight|test_model'] == 1.0 
    
    # Check pred len
    assert len(pred_df) == 4

@patch("tfscreen.analysis.cat_response.cat_fit.MODEL_LIBRARY")
@patch("tfscreen.analysis.cat_response.cat_fit.run_matrix_wls")
@patch("tfscreen.analysis.cat_response.cat_fit.predict_with_error")
def test_fit_matrix_wls_success(mock_predict, mock_run_wls, mock_library, mock_data, mock_predict_side_effect):
    """Test successful fit for a model using matrix WLS path."""
    x, y, y_std = mock_data
    mock_predict.side_effect = mock_predict_side_effect
    
    mock_model_func = MagicMock(return_value=y)
    # 2D return for guess func
    mock_guess_func = MagicMock(return_value=np.zeros((len(x), 2))) 
    
    mock_library.__getitem__.return_value = {
        "model_func": mock_model_func,
        "guess_func": mock_guess_func,
        "param_names": ["p1", "p2"],
        "bounds": ([-np.inf], [np.inf])
    }
    
    mock_run_wls.return_value = (np.array([1.0, 1.0]), np.array([0.1, 0.1]), np.eye(2), None)
    
    x_pred = np.array([1.0]) # tiny pred
    flat_output, pred_df = cat_fit(x, y, y_std, x_pred=x_pred, models_to_run=["test_matrix_model"])
    
    assert flat_output['status'] == "success"
    mock_run_wls.assert_called_once()
    assert len(pred_df) == 1

@patch("tfscreen.analysis.cat_response.cat_fit.MODEL_LIBRARY")
@patch("tfscreen.analysis.cat_response.cat_fit.run_least_squares")
def test_fit_failure(mock_run_ls, mock_library, mock_data):
    """Test handling of fit failure."""
    x, y, y_std = mock_data
    
    mock_guess_func = MagicMock(return_value=np.array([1.0]))
    mock_library.__getitem__.return_value = {
        "model_func": MagicMock(),
        "guess_func": mock_guess_func,
        "param_names": ["p1"],
        "bounds": ([-np.inf], [np.inf])
    }
    
    fit_obj = MagicMock()
    fit_obj.success = False
    fit_obj.message = "Failed"
    mock_run_ls.return_value = (None, None, None, fit_obj)
    
    flat_output, pred_df = cat_fit(x, y, y_std, x_pred=x, models_to_run=["test_fail_model"], verbose=True)
    
    assert flat_output['status'] == "failure"
    assert flat_output['best_model'] == "None" 
    assert np.isnan(flat_output['R2|test_fail_model'])

@patch("tfscreen.analysis.cat_response.cat_fit.MODEL_LIBRARY")
@patch("tfscreen.analysis.cat_response.cat_fit.run_least_squares")
@patch("tfscreen.analysis.cat_response.cat_fit.predict_with_error")
def test_multiple_models_selection(mock_predict, mock_run_ls, mock_library, mock_data, mock_predict_side_effect):
    """Test running multiple models and selecting the best one."""
    x, y, y_std = mock_data
    mock_predict.side_effect = mock_predict_side_effect
    
    # Model A: Good fit
    mock_func_A = MagicMock(return_value=y + 0.001)
    # Model B: Bad fit
    mock_func_B = MagicMock(return_value=y + 10) 
    
    mock_guess = MagicMock(return_value=np.array([1.0]))
    
    library_dict = {
        "model_A": {
            "model_func": mock_func_A,
            "guess_func": mock_guess,
            "param_names": ["pA"],
            "bounds": ([-np.inf], [np.inf])
        },
        "model_B": {
            "model_func": mock_func_B,
            "guess_func": mock_guess,
            "param_names": ["pB"],
            "bounds": ([-np.inf], [np.inf])
        }
    }
    mock_library.__getitem__.side_effect = lambda k: library_dict[k]
    
    fit_obj = MagicMock()
    fit_obj.success = True
    mock_run_ls.return_value = (np.array([1.0]), np.array([0.1]), np.eye(1), fit_obj)
    
    flat_output, pred_df = cat_fit(x, y, y_std, x_pred=x, models_to_run=["model_A", "model_B"])
    
    assert flat_output['best_model'] == "model_A"
    assert flat_output['AIC_weight|model_A'] > flat_output['AIC_weight|model_B']

def test_x_pred_none(mock_data):
    """Test x_pred default generation behavior (using xfill)."""
    x, y, y_std = mock_data
    
    with patch("tfscreen.analysis.cat_response.cat_fit.xfill") as mock_xfill:
        with patch("tfscreen.analysis.cat_response.cat_fit.MODEL_LIBRARY") as mock_lib:
            # We need to run at least one model to avoid empty summary_df issues 
            # (though strictly user shouldn't pass empty model list, test handles typical case)
            # Actually if models_to_run is [], code does not crash, just returns empty/nan stuff?
            # Let's check code: 
            # if models_to_run is None -> keys(). If keys empty -> empty list.
            # loop doesn't run. summary_results empty.
            # valid_aics empty.
            # code handles valid_aics.empty.
            # BUT: summary_df['success'] accessed below.
            # summary_df = pd.DataFrame(summary_results).
            # If summary_results [], summary_df empty. summary_df['success'] raises KeyError?
            # YES.
            # So models_to_run cannot be empty if we want success.
            # Wait, `predict_with_error` usually mocked.
            
            # Update: let's mock xfill to return something usable, and run a dummy model.
            mock_xfill.return_value = x # simple return
            
            mock_guess = MagicMock(return_value=np.array([1.0]))
            mock_lib.__getitem__.return_value = {
                "model_func": MagicMock(return_value=y),
                "guess_func": mock_guess,
                "param_names": ["p"],
                "bounds": ([-np.inf], [np.inf])
            }
            mock_lib.keys.return_value = ["test"]
            
            with patch("tfscreen.analysis.cat_response.cat_fit.run_least_squares") as mock_ls:
                 mock_ls.return_value = (np.array([1.]), np.array([1.]), np.eye(1), MagicMock(success=True))
                 with patch("tfscreen.analysis.cat_response.cat_fit.predict_with_error") as mock_pred:
                     mock_pred.return_value = (x, x)
                     
                     cat_fit(x, y, y_std, x_pred=None, models_to_run=["test"])
                     
                     mock_xfill.assert_called_once()

@patch("tfscreen.analysis.cat_response.cat_fit.MODEL_LIBRARY")
@patch("tfscreen.analysis.cat_response.cat_fit.run_least_squares")
@patch("tfscreen.analysis.cat_response.cat_fit.predict_with_error")
def test_sanitize_inputs(mock_predict, mock_ls, mock_lib, mock_predict_side_effect):
    """Test that NaNs/Infs are filtered out."""
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([1.0, 2.0, 3.0, np.inf])
    y_std = np.array([0.1, 0.1, 0.1, 0.1])
    
    mock_predict.side_effect = mock_predict_side_effect
    
    mock_guess = MagicMock(return_value=np.array([1.0]))
    mock_lib.__getitem__.return_value = {
        "model_func": MagicMock(return_value=np.array([0.,0.])), # match filtered size 2
        "guess_func": mock_guess,
        "param_names": ["p"],
        "bounds": ([-np.inf], [np.inf])
    }
    
    mock_ls.return_value = (np.array([1.]), np.array([0.1]), np.eye(1), MagicMock(success=True))
    
    # x_pred passed explicit to avoid issues
    cat_fit(x, y, y_std, x_pred=np.array([1.0]), models_to_run=["test"])
    
    # Check call args to guess_func
    args, _ = mock_guess.call_args
    x_filtered, y_filtered = args
    
    assert len(x_filtered) == 2
    assert np.allclose(x_filtered, [1.0, 2.0])
