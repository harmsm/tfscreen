import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.memory import (
    define_model, guide, get_priors, get_hyperparameters, get_guesses
)
from unittest.mock import MagicMock, patch

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = get_hyperparameters()
    assert "tau0_hyper_loc_loc" in params
    assert "k1_hyper_loc_loc" in params
    assert "k2_hyper_loc_loc" in params
    
    data = MagicMock()
    data.num_condition_rep = 5
    guesses = get_guesses("test", data)
    assert "test_tau0_hyper_loc" in guesses
    assert "test_tau0_offset" in guesses
    assert guesses["test_tau0_offset"].shape == (5,)
    
    priors = get_priors()
    assert priors is not None
    assert priors.tau0_hyper_loc_loc == params["tau0_hyper_loc_loc"]

def test_define_model():
    """Test define_model growth transition calculation."""
    data = MagicMock()
    data.num_condition_rep = 1
    # map_condition_pre maps everything to index 0
    data.map_condition_pre = jnp.zeros((2,), dtype=int)
    
    priors = get_priors()
    
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    theta = jnp.array([0.5, 0.5])
    
    # Mock pyro.sample to return fixed values
    # We expect 6 hyper samples + 3 offset samples in a plate
    # We'll use a side_effect to return values in order
    sample_values = [
        1.0, # tau0_hyper_loc
        0.1, # tau0_hyper_scale
        0.5, # k1_hyper_loc
        0.1, # k1_hyper_scale
        0.1, # k2_hyper_loc
        0.05, # k2_hyper_scale
        jnp.zeros(1), # tau0_offset
        jnp.zeros(1), # k1_offset
        jnp.zeros(1), # k2_offset
    ]
    
    with patch("numpyro.sample", side_effect=sample_values) as mock_sample:
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = jnp.arange(data.num_condition_rep)
        with patch("numpyro.plate", return_value=plate_mock):
            total_growth = define_model("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta)
            
            # tau0 = 1.0 + 0 * 0.1 = 1.0
            # k1 = 0.5 + 0 * 0.1 = 0.5
            # k2 = 0.1 + 0 * 0.05 = 0.1
            # tau = 1.0 + (0.5 / (0.5 + 0.1)) = 1.0 + (0.5/0.6) = 1.0 + 0.8333 = 1.8333
            
            # t_sel = 20.0 > tau = 1.8333
            # dln_cfu_pre = g_pre * t_pre = [1*10, 2*10] = [10, 20]
            # dln_cfu_sel = g_pre * tau + g_sel * (t_sel - tau)
            # 1st = 1.0 * 1.8333 + 0.5 * (20 - 1.8333) = 1.8333 + 0.5 * 18.1667 = 1.8333 + 9.0833 = 10.9166
            # 2nd = 2.0 * 1.8333 + 1.5 * (20 - 1.8333) = 3.6666 + 1.5 * 18.1667 = 3.6666 + 27.25 = 30.9166
            
            # total = [10 + 10.9166, 20 + 30.9166] = [20.9166, 50.9166]
            
            tau_val = 1.0 + (0.5 / 0.6)
            expected_1 = 1.0 * 10 + 1.0 * tau_val + 0.5 * (20 - tau_val)
            expected_2 = 2.0 * 10 + 2.0 * tau_val + 1.5 * (20 - tau_val)
            expected_growth = jnp.array([expected_1, expected_2])
            
            assert jnp.allclose(total_growth, expected_growth)
            assert mock_sample.called

def test_define_model_short_time():
    """Test define_model when t_sel < tau."""
    data = MagicMock()
    data.num_condition_rep = 1
    data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()
    
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([0.5]) # Very short time
    theta = jnp.array([0.5])
    
    # tau will be 1.8333 again
    sample_values = [1.0, 0.1, 0.5, 0.1, 0.1, 0.05, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)]
    
    with patch("numpyro.sample", side_effect=sample_values):
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = jnp.arange(data.num_condition_rep)
        with patch("numpyro.plate", return_value=plate_mock):
            total_growth = define_model("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta)
            
            # Since t_sel (0.5) < tau (1.8333)
            # total = g_pre * t_pre + g_pre * t_sel = 1.0 * 10 + 1.0 * 0.5 = 10.5
            assert jnp.allclose(total_growth, jnp.array([10.5]))

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
    theta = jnp.array([0.5])
    
    # guide has 6 hyper + 3 offset samples
    sample_values = [1.0, 0.1, 0.5, 0.1, 0.1, 0.05, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)]
    
    with patch("numpyro.sample", side_effect=sample_values):
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = jnp.arange(data.num_condition_rep)
        with patch("numpyro.plate", return_value=plate_mock):
            # Use Softplus or LogNormal for scales in reality, but mock doesn't care
            total_growth = guide("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta)
            
            # Should match define_model with same sample values
            tau_val = 1.0 + (0.5 / 0.6)
            expected = 1.0 * 10 + 1.0 * tau_val + 0.5 * (20 - tau_val)
            assert jnp.allclose(total_growth, jnp.array([expected]))
