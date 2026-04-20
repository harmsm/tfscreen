"""Unit tests for the shared pinning helpers."""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from tfscreen.analysis.hierarchical.growth_model.components._pinning import (
    _hyper,
    _pinned_value,
)


# ---------------------------------------------------------------------------
# _pinned_value
# ---------------------------------------------------------------------------


def test_pinned_value_returns_none_when_suffix_missing():
    assert _pinned_value("k_hyper_loc", {}) is None
    assert _pinned_value("k_hyper_loc", {"m_hyper_loc": 1.0}) is None


def test_pinned_value_returns_jax_scalar():
    val = _pinned_value("k_hyper_loc", {"k_hyper_loc": 0.04})
    assert val is not None
    assert isinstance(val, jnp.ndarray)
    assert val.shape == ()
    assert float(val) == pytest.approx(0.04)


def test_pinned_value_casts_int_to_float():
    val = _pinned_value("foo", {"foo": 3})
    assert val.dtype == jnp.float32 or val.dtype == jnp.float64


# ---------------------------------------------------------------------------
# _hyper - unpinned path
# ---------------------------------------------------------------------------


def test_hyper_unpinned_creates_sample_site():
    def model():
        return _hyper("g", "k_hyper_loc", dist.Normal(0.0, 1.0), {})

    with seed(rng_seed=0):
        tr = trace(model).get_trace()

    site = tr["g_k_hyper_loc"]
    assert site["type"] == "sample"
    assert isinstance(site["fn"], dist.Normal)


def test_hyper_unpinned_returns_sample_value():
    """The returned value must equal the value recorded in the trace."""
    captured = {}

    def model():
        captured["val"] = _hyper("g", "k_hyper_loc",
                                 dist.Normal(0.0, 1.0), {})

    with seed(rng_seed=0):
        tr = trace(model).get_trace()

    assert float(captured["val"]) == pytest.approx(
        float(tr["g_k_hyper_loc"]["value"])
    )


# ---------------------------------------------------------------------------
# _hyper - pinned path
# ---------------------------------------------------------------------------


def test_hyper_pinned_creates_deterministic_site():
    def model():
        return _hyper("g", "k_hyper_loc",
                      dist.Normal(0.0, 1.0),
                      {"k_hyper_loc": 0.04})

    with seed(rng_seed=0):
        tr = trace(model).get_trace()

    site = tr["g_k_hyper_loc"]
    assert site["type"] == "deterministic"
    assert float(site["value"]) == pytest.approx(0.04)


def test_hyper_pinned_returns_pinned_constant():
    captured = {}

    def model():
        captured["val"] = _hyper("g", "k_hyper_loc",
                                 dist.Normal(0.0, 1.0),
                                 {"k_hyper_loc": 0.04})

    with seed(rng_seed=0):
        trace(model).get_trace()

    assert float(captured["val"]) == pytest.approx(0.04)


def test_hyper_pinned_ignores_other_suffixes_in_pinned_dict():
    """Only the matching suffix should pin; others should sample."""
    pinned = {"m_hyper_loc": 99.0}

    def model():
        return _hyper("g", "k_hyper_loc",
                      dist.Normal(0.0, 1.0),
                      pinned)

    with seed(rng_seed=0):
        tr = trace(model).get_trace()

    # Should sample, not deterministic
    assert tr["g_k_hyper_loc"]["type"] == "sample"


def test_hyper_name_prefix_in_site_name():
    def model():
        return _hyper("widget", "k_hyper_loc",
                      dist.Normal(0.0, 1.0), {})

    with seed(rng_seed=0):
        tr = trace(model).get_trace()

    assert "widget_k_hyper_loc" in tr
