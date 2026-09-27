"""Tests for the per-tube level offset (sample_offset: level)."""
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from tfscreen.tfmodel.generative.components.sample_offset.level import (
    ModelPriors,
    define_model,
    get_guesses,
    get_priors,
    guide,
)

SHAPE = (2, 3, 1, 2, 1, 4)


def _data():
    data = MagicMock()
    (data.num_replicate, data.num_time, data.num_condition_pre,
     data.num_condition_sel, data.num_titrant_name,
     data.num_titrant_conc) = SHAPE
    return data


def _trace(fn, **subs):
    f = handlers.substitute(fn, data=subs) if subs else fn
    return handlers.trace(handlers.seed(f, 0)).get_trace(
        "sample_offset", _data(), get_priors())


def test_priors():
    assert isinstance(get_priors(), ModelPriors)
    assert get_priors().sigma_prior_scale == 0.2


@pytest.mark.parametrize("fn", [define_model, guide])
def test_shape_broadcasts_over_genotypes(fn):
    with handlers.seed(rng_seed=0):
        out = fn("sample_offset", _data(), get_priors())
    assert out.shape == (*SHAPE, 1)


def test_one_offset_per_tube_constant_in_time():
    num_tubes = int(np.prod(SHAPE))
    offsets = jnp.arange(num_tubes, dtype=float) / 100.0
    tr = _trace(define_model, sample_offset_sigma=0.1,
                sample_offset_offset=offsets)
    assert tr["sample_offset_offset"]["value"].shape == (num_tubes,)
    with handlers.substitute(data={"sample_offset_sigma": 0.1,
                                   "sample_offset_offset": offsets}):
        out = define_model("sample_offset", _data(), get_priors())
    # The offset is the level itself: no scaling by elapsed time.
    assert np.allclose(np.asarray(out).ravel(), np.asarray(offsets))


def test_guide_follows_loc_scale_convention():
    tr = _trace(guide)
    params = {k for k, v in tr.items() if v["type"] == "param"}
    assert params == {"sample_offset_sigma_loc", "sample_offset_sigma_scale",
                      "sample_offset_offset_loc", "sample_offset_offset_scale"}


def test_guesses():
    assert set(get_guesses("sample_offset", _data())) == {"sample_offset_sigma"}
