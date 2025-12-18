import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components import transformation_single
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

def test_get_hyperparameters():
    """Verify get_hyperparameters returns empty dict."""
    params = transformation_single.get_hyperparameters()
    assert params == {}
    assert isinstance(params, dict)

def test_get_guesses():
    """Verify get_guesses returns empty dict."""
    guesses = transformation_single.get_guesses("test", MagicMock())
    assert guesses == {}
    assert isinstance(guesses, dict)

def test_get_priors():
    """Verify get_priors returns a ModelPriors instance."""
    priors = transformation_single.get_priors()
    assert isinstance(priors, transformation_single.ModelPriors)

def test_define_model():
    """Verify define_model returns 1.0, 1.0, 1.0."""
    res = transformation_single.define_model("test", MagicMock(), MagicMock())
    assert res == (1.0, 1.0, 1.0)

def test_guide():
    """Verify guide returns 1.0, 1.0, 1.0."""
    res = transformation_single.guide("test", MagicMock(), MagicMock())
    assert res == (1.0, 1.0, 1.0)

def test_update_thetas():
    """Verify update_thetas acts as a pass-through."""
    theta = jnp.array([0.1, 0.5, 0.9])
    
    # Standard call
    res = transformation_single.update_thetas(theta, params=(1.0, 1.0, 1.0))
    assert jnp.array_equal(res, theta)
    
    # With different params
    res2 = transformation_single.update_thetas(theta, params=(100.0, 5.0, 0.1))
    assert jnp.array_equal(res2, theta)
