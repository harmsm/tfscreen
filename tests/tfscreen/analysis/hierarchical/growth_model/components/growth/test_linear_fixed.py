import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock
import numpyro
from tfscreen.analysis.hierarchical.growth_model.components.growth import linear_fixed as growth_fixed

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = growth_fixed.get_hyperparameters()
    assert "growth_k_per_cond_rep" in params
    assert "growth_m_per_cond_rep" in params
    
    guesses = growth_fixed.get_guesses("test", MagicMock())
    assert guesses == {}
    
    priors = growth_fixed.get_priors()
    assert isinstance(priors, growth_fixed.ModelPriors)
    assert jnp.allclose(priors.growth_k_per_cond_rep, params["growth_k_per_cond_rep"])

def test_define_model():
    """Test define_model expansion logic."""
    priors = growth_fixed.get_priors()
    data = MagicMock()
    # 2 conditions, 2 replicates -> total 4 entries in priors
    # Let's map them
    data.map_condition_pre = jnp.array([0, 1])
    data.map_condition_sel = jnp.array([2, 3])
    
    params = growth_fixed.define_model("test", data, priors)
    k_pre = params.k_pre
    m_pre = params.m_pre
    k_sel = params.k_sel
    m_sel = params.m_sel
    
    assert k_pre.shape == (2,)
    assert k_sel.shape == (2,)
    assert jnp.isclose(k_pre[0], priors.growth_k_per_cond_rep[0])
    assert jnp.isclose(k_sel[0], priors.growth_k_per_cond_rep[2])

def test_guide():
    """Test guide expansion logic."""
    priors = growth_fixed.get_priors()
    data = MagicMock()
    data.map_condition_pre = jnp.array([0])
    data.map_condition_sel = jnp.array([1])
    
    params = growth_fixed.guide("test", data, priors)
    k_pre = params.k_pre
    m_pre = params.m_pre
    k_sel = params.k_sel
    m_sel = params.m_sel
    
    assert k_pre.shape == (1,)
    assert jnp.isclose(k_pre[0], priors.growth_k_per_cond_rep[0])
