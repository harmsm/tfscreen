import pytest
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.instant import (
    define_model, guide, get_priors, get_hyperparameters, get_guesses
)
from unittest.mock import MagicMock

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = get_hyperparameters()
    assert params == {}
    
    guesses = get_guesses("test", MagicMock())
    assert guesses == {}
    priors = get_priors()
    assert priors is not None
    assert isinstance(priors, type(get_priors()))

def test_define_model():
    """Test define_model growth transition calculation."""
    data = MagicMock()
    priors = None
    
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    
    total_growth = define_model("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta=None)
    
    # expected_growth = g_pre * t_pre + g_sel * t_sel
    # 1st = 1.0 * 10 + 0.5 * 20 = 10 + 10 = 20
    # 2nd = 2.0 * 10 + 1.5 * 20 = 20 + 30 = 50
    expected_growth = jnp.array([20.0, 50.0])
    
    assert jnp.allclose(total_growth, expected_growth)

def test_guide():
    """Test guide logic matches define_model."""
    data = MagicMock()
    priors = None
    
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    
    total_growth = guide("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta=None)
    
    expected_growth = jnp.array([20.0, 50.0])
    
    assert jnp.allclose(total_growth, expected_growth)
