import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.baranyi import (
    define_model, guide, get_priors, get_hyperparameters, get_guesses
)
from unittest.mock import MagicMock, patch

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = get_hyperparameters()
    assert "tau_lag_hyper_loc_loc" in params
    assert "k_sharp_hyper_loc_loc" in params
    
    data = MagicMock()
    data.num_condition_rep = 5
    guesses = get_guesses("test", data)
    assert "test_tau_lag_hyper_loc" in guesses
    assert "test_tau_lag_offset" in guesses
    assert guesses["test_tau_lag_offset"].shape == (5,)
    
    priors = get_priors()
    assert priors is not None
    assert priors.tau_lag_hyper_loc_loc == params["tau_lag_hyper_loc_loc"]

def test_define_model():
    """Test define_model growth transition calculation."""
    data = MagicMock()
    data.num_condition_rep = 1
    data.map_condition_pre = jnp.zeros((1,), dtype=int)
    
    priors = get_priors()
    
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([2.0])
    theta = None
    
    # Mock pyro.sample to return fixed values
    # 4 hypers + 2 offsets
    sample_values = [
        1.0, # tau_lag_hyper_loc
        0.1, # tau_lag_hyper_scale
        1.0, # k_sharp_hyper_loc
        0.1, # k_sharp_hyper_scale
        jnp.zeros(1), # tau_lag_offset
        jnp.zeros(1), # k_sharp_offset
    ]
    
    with patch("numpyro.sample", side_effect=sample_values) as mock_sample:
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = jnp.arange(data.num_condition_rep)
        with patch("numpyro.plate", return_value=plate_mock):
            total_growth = define_model("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta)
            
            # tau_lag = 1.0 + 0 * 0.1 = 1.0
            # k_sharp = exp(1.0 + 0 * 0.1) = e^1 approx 2.71828
            
            k_val = jnp.exp(1.0)
            tau_val = 1.0
            t_val = 2.0
            
            term1 = jnp.logaddexp(0.0, k_val * (t_val - tau_val))
            term0 = jnp.logaddexp(0.0, -k_val * tau_val)
            integrated_sigmoid = (term1 - term0) / k_val
            
            expected_growth = 1.0 * 10.0 + 1.0 * 2.0 + (0.5 - 1.0) * integrated_sigmoid
            
            assert jnp.allclose(total_growth, jnp.array([expected_growth]))
            assert mock_sample.called

def test_guide():
    """Test guide logic follows the same structure."""
    data = MagicMock()
    data.num_condition_rep = 1
    data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()
    
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([20.0])
    
    # 4 hypers + 2 offsets
    sample_values = [1.0, 0.1, 1.0, 0.1, jnp.zeros(1), jnp.zeros(1)]
    
    with patch("numpyro.sample", side_effect=sample_values):
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = jnp.arange(data.num_condition_rep)
        with patch("numpyro.plate", return_value=plate_mock):
            total_growth = guide("test", data, priors, g_pre, g_sel, t_pre, t_sel)
            
            k_val = jnp.exp(1.0)
            tau_val = 1.0
            t_val = 20.0
            
            term1 = jnp.logaddexp(0.0, k_val * (t_val - tau_val))
            term0 = jnp.logaddexp(0.0, -k_val * tau_val)
            integrated_sigmoid = (term1 - term0) / k_val
            
            expected_growth = 1.0 * 10.0 + 1.0 * 20.0 + (0.5 - 1.0) * integrated_sigmoid
            assert jnp.allclose(total_growth, jnp.array([expected_growth]))
