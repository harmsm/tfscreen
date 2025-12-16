
import pytest
import numpy as np
import pandas as pd
from tfscreen.fitting.fit_manager import FitManager
from scipy.special import expit, logit

@pytest.fixture
def example_fit_manager():
    y_obs = np.array([1.0, 2.0, 3.0])
    y_std = np.array([0.1, 0.1, 0.1])
    X = np.eye(3)
    param_df = pd.DataFrame({
        "guess": [1.0, 2.0, 3.0],
        "fixed": [False, False, False],
        "transform": [None, None, None]
    })
    return FitManager(y_obs, y_std, X, param_df)

def test_init_basic(example_fit_manager):
    fm = example_fit_manager
    assert fm.num_obs == 3
    assert fm.num_params == 3
    assert np.allclose(fm.y_obs, np.array([1.0, 2.0, 3.0]))

def test_init_validation():
    y_obs = np.array([1.0])
    y_std = np.array([0.1])
    X = np.array([[1.0]])
    
    # Missing guess column
    with pytest.raises(ValueError, match="param_df must have a column named 'guess'"):
        FitManager(y_obs, y_std, X, pd.DataFrame({"foo": [1.0]}))
        
    # Invalid scale_sigma
    with pytest.raises(ValueError, match="Values in 'scale_sigma' must be positive"):
        FitManager(y_obs, y_std, X, pd.DataFrame({
            "guess": [1.0], 
            "transform": ["scale"], 
            "scale_sigma": [-1.0]
        }))

def test_transform_scale():
    y = np.zeros(1)
    X = np.zeros((1,1))
    df = pd.DataFrame({
        "guess": [10.0],
        "transform": ["scale"],
        "scale_mu": [5.0],
        "scale_sigma": [2.0]
    })
    fm = FitManager(y, y, X, df)
    
    # transform: (10 - 5) / 2 = 2.5
    v = np.array([10.0])
    v_t = fm.transform(v)
    assert np.allclose(v_t, [2.5])
    
    # back_transform: 2.5 * 2 + 5 = 10
    v_b = fm.back_transform(v_t)
    assert np.allclose(v_b, [10.0])

def test_transform_logistic():
    y = np.zeros(1)
    X = np.zeros((1,1))
    df = pd.DataFrame({
        "guess": [0.5],
        "transform": ["logistic"]
    })
    fm = FitManager(y, y, X, df)
    
    # logit(0.5) = 0.0
    v = np.array([0.5])
    v_t = fm.transform(v)
    assert np.allclose(v_t, [0.0])
    
    # expit(0.0) = 0.5
    v_b = fm.back_transform(v_t)
    assert np.allclose(v_b, [0.5])

def test_bounds_transformed():
    y = np.zeros(2)
    X = np.zeros((2,2))
    df = pd.DataFrame({
        "guess": [0.5, 10.0],
        "transform": ["logistic", "scale"],
        "scale_mu": [0, 5.0],
        "scale_sigma": [1, 2.0],
        "lower_bounds": [0.1, 8.0],
        "upper_bounds": [0.9, 12.0]
    })
    fm = FitManager(y, y, X, df)
    
    lb_t = fm.lower_bounds_transformed
    # logistic: logit(0.1)
    # scale: (8 - 5) / 2 = 1.5
    assert np.allclose(lb_t[0], logit(0.1))
    assert np.allclose(lb_t[1], 1.5)
    
    ub_t = fm.upper_bounds_transformed
    # logistic: logit(0.9)
    # scale: (12 - 5) / 2 = 3.5
    assert np.allclose(ub_t[0], logit(0.9))
    assert np.allclose(ub_t[1], 3.5)

def test_stability_bounds():
    # Test that logistic bounds are capped
    y = np.zeros(1)
    X = np.zeros((1,1))
    df = pd.DataFrame({
        "guess": [0.5],
        "transform": ["logistic"],
        "lower_bounds": [-np.inf],
        "upper_bounds": [np.inf],
    })
    fm = FitManager(y, y, X, df)
    
    assert fm.lower_bounds_transformed[0] == -fm._LOGISTIC_STABILITY_BOUND
    assert fm.upper_bounds_transformed[0] == fm._LOGISTIC_STABILITY_BOUND

def test_predict(example_fit_manager):
    fm = example_fit_manager
    v = np.array([1.1, 2.1, 3.1])
    # X is eye(3), so X @ v = v
    pred = fm.predict(v)
    assert np.allclose(pred, v)

def test_predict_from_transformed():
    y = np.zeros(1)
    X = np.array([[2.0]])
    df = pd.DataFrame({
        "guess": [10.0],
        "transform": ["scale"],
        "scale_mu": [0.0],
        "scale_sigma": [0.5]
    })
    fm = FitManager(y, y, X, df)
    
    # Transformed value of 20.0 -> (10 - 0) / 0.5 = 20
    v_t = np.array([20.0])
    
    # back_transform: 20 * 0.5 + 0 = 10.0
    # predict: 2.0 * 10.0 = 20.0
    pred = fm.predict_from_transformed(v_t)
    assert np.allclose(pred, [20.0])

def test_fixed_parameters():
    y = np.zeros(2)
    X = np.eye(2)
    df = pd.DataFrame({
        "guess": [1.0, 5.0],
        "fixed": [False, True],
    })
    fm = FitManager(y, y, X, df)
    
    v_t = np.array([2.0, 333.0]) # 333.0 should be ignored for fixed param
    
    # back_transform should reset fixed parameters to guess
    v_b = fm.back_transform(v_t)
    assert np.allclose(v_b[0], 2.0)
    assert np.allclose(v_b[1], 5.0) # Reset to guess

def test_back_transform_std_err():
    y = np.zeros(2)
    X = np.eye(2)
    df = pd.DataFrame({
        "guess": [0.5, 10.0],
        "transform": ["logistic", "scale"],
        "scale_sigma": [1, 2.0],
    })
    fm = FitManager(y, y, X, df)
    
    v_t = np.array([0.0, 1.0]) # logistic(0.5)=0, scale((12-0)/2)=? wait. guess 10 scale sigma 2.
    # We are providing v_t. 
    # v_t[0] = 0 -> p = 0.5. Derivative at 0.5: 0.5 * (1-0.5) = 0.25
    # v_t[1] = 1 -> val = 1*2 + 0 = 2? No mu defaults to 0. 
    # scale deriv is sigma = 2.
    
    std_t = np.array([0.1, 0.1])
    
    std = fm.back_transform_std_err(v_t, std_t)
    
    expected_logistic_err = 0.25 * 0.1
    expected_scale_err = 2.0 * 0.1
    
    assert np.allclose(std[0], expected_logistic_err)
    assert np.allclose(std[1], expected_scale_err)

def test_reprs(example_fit_manager):
    assert "FitManager" in str(example_fit_manager)
    assert "FitManager" in repr(example_fit_manager)
    assert "<h3>FitManager</h3>" in example_fit_manager._repr_html_()
