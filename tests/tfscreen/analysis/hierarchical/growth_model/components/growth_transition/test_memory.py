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

# ---------------------------------------------------------------------------
# Pinning tests
# ---------------------------------------------------------------------------

from numpyro.handlers import trace, seed
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.memory import (
    _PINNABLE_SUFFIXES,
    ModelPriors,
)


def _trace_args():
    """Build the args needed to trace memory.define_model / guide."""
    data = MagicMock()
    data.num_condition_rep = 3
    data.map_condition_pre = jnp.array([0, 1, 2], dtype=int)
    g_pre = jnp.array([1.0, 1.0, 1.0])
    g_sel = jnp.array([0.5, 0.5, 0.5])
    t_pre = jnp.array([10.0, 10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0, 20.0])
    theta = jnp.array([0.5, 0.5, 0.5])
    return data, g_pre, g_sel, t_pre, t_sel, theta


def test_pinnable_suffixes_includes_all_six_hypers():
    assert set(_PINNABLE_SUFFIXES) == {
        "tau0_hyper_loc", "tau0_hyper_scale",
        "k1_hyper_loc", "k1_hyper_scale",
        "k2_hyper_loc", "k2_hyper_scale",
    }


def test_model_priors_default_pinned_is_empty_dict():
    assert get_priors().pinned == {}


def test_model_priors_accepts_pinned_dict():
    pinned = {"tau0_hyper_loc": 1.0}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


def test_define_model_unpinned_uses_sample_sites():
    name = "g"
    priors = get_priors()
    data, g_pre, g_sel, t_pre, t_sel, theta = _trace_args()
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )
    for suffix in _PINNABLE_SUFFIXES:
        assert tr[f"{name}_{suffix}"]["type"] == "sample"


def test_define_model_pinned_replaces_with_deterministic():
    name = "g"
    pinned = {"tau0_hyper_loc": 1.0, "k2_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    data, g_pre, g_sel, t_pre, t_sel, theta = _trace_args()
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )
    assert tr[f"{name}_tau0_hyper_loc"]["type"] == "deterministic"
    assert float(tr[f"{name}_tau0_hyper_loc"]["value"]) == pytest.approx(1.0)
    assert tr[f"{name}_k2_hyper_scale"]["type"] == "deterministic"
    assert tr[f"{name}_tau0_hyper_scale"]["type"] == "sample"
    assert tr[f"{name}_k1_hyper_loc"]["type"] == "sample"


def test_guide_pinned_drops_variational_params():
    name = "g"
    pinned = {"tau0_hyper_loc": 1.0, "k2_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    data, g_pre, g_sel, t_pre, t_sel, theta = _trace_args()
    with seed(rng_seed=0):
        tr = trace(guide).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )
    assert f"{name}_tau0_hyper_loc_loc" not in tr
    assert f"{name}_tau0_hyper_loc_scale" not in tr
    assert f"{name}_tau0_hyper_loc" not in tr
    assert f"{name}_k2_hyper_scale_loc" not in tr
    assert f"{name}_k2_hyper_scale_scale" not in tr
    assert f"{name}_k2_hyper_scale" not in tr
    # Unpinned still present
    assert f"{name}_k1_hyper_loc" in tr


def test_model_and_guide_pinned_have_compatible_sample_sites():
    name = "g"
    pinned = {"tau0_hyper_loc": 1.0, "k2_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    data, g_pre, g_sel, t_pre, t_sel, theta = _trace_args()
    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(
            name=name, data=data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )
    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples
