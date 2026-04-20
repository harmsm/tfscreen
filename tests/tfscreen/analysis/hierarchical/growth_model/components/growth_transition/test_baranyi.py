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

# ---------------------------------------------------------------------------
# Pinning tests
# ---------------------------------------------------------------------------

from numpyro.handlers import trace, seed
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.baranyi import (
    _PINNABLE_SUFFIXES,
    ModelPriors,
)


def _trace_args():
    data = MagicMock()
    data.num_condition_rep = 3
    data.map_condition_pre = jnp.array([0, 1, 2], dtype=int)
    g_pre = jnp.array([1.0, 1.0, 1.0])
    g_sel = jnp.array([0.5, 0.5, 0.5])
    t_pre = jnp.array([10.0, 10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0, 20.0])
    return data, g_pre, g_sel, t_pre, t_sel


def test_pinnable_suffixes():
    assert set(_PINNABLE_SUFFIXES) == {
        "tau_lag_hyper_loc", "tau_lag_hyper_scale",
        "k_sharp_hyper_loc", "k_sharp_hyper_scale",
    }


def test_model_priors_default_pinned_is_empty():
    assert get_priors().pinned == {}


def test_model_priors_accepts_pinned_dict():
    pinned = {"tau_lag_hyper_loc": 1.0}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


def test_define_model_unpinned_uses_sample_sites():
    name = "g"
    priors = get_priors()
    data, g_pre, g_sel, t_pre, t_sel = _trace_args()
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    for suffix in _PINNABLE_SUFFIXES:
        assert tr[f"{name}_{suffix}"]["type"] == "sample"


def test_define_model_pinned_replaces_with_deterministic():
    name = "g"
    pinned = {"tau_lag_hyper_loc": 2.0, "k_sharp_hyper_scale": 0.3}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    data, g_pre, g_sel, t_pre, t_sel = _trace_args()
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    assert tr[f"{name}_tau_lag_hyper_loc"]["type"] == "deterministic"
    assert float(tr[f"{name}_tau_lag_hyper_loc"]["value"]) == pytest.approx(2.0)
    assert tr[f"{name}_k_sharp_hyper_scale"]["type"] == "deterministic"
    assert tr[f"{name}_tau_lag_hyper_scale"]["type"] == "sample"


def test_guide_pinned_drops_variational_params():
    name = "g"
    pinned = {"tau_lag_hyper_loc": 2.0, "k_sharp_hyper_scale": 0.3}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    data, g_pre, g_sel, t_pre, t_sel = _trace_args()
    with seed(rng_seed=0):
        tr = trace(guide).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    assert f"{name}_tau_lag_hyper_loc_loc" not in tr
    assert f"{name}_tau_lag_hyper_loc" not in tr
    assert f"{name}_k_sharp_hyper_scale_loc" not in tr
    assert f"{name}_k_sharp_hyper_scale" not in tr
    assert f"{name}_tau_lag_hyper_scale" in tr
    assert f"{name}_k_sharp_hyper_loc" in tr


def test_model_and_guide_pinned_compatible():
    name = "g"
    pinned = {"tau_lag_hyper_loc": 2.0, "k_sharp_hyper_scale": 0.3}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    data, g_pre, g_sel, t_pre, t_sel = _trace_args()
    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples
