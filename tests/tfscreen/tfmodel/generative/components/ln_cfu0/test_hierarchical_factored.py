import pytest
import inspect
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.tfmodel.generative.components.ln_cfu0.hierarchical_factored import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    _FALLBACK_TUBE_SCALE,
    _PINNABLE_SUFFIXES,
)

# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

MockGrowthData = namedtuple("MockGrowthData", [
    "num_replicate",
    "num_condition_pre",
    "num_genotype",
    "batch_size",
    "batch_idx",
    "scale_vector",
    "ln_cfu0_spiked_mask",
    "ln_cfu0_wt_mask",
    "ln_cfu0_library_masks",
    "num_ln_cfu0_library_classes",
    "ln_cfu",
    "good_mask",
    "map_ln_cfu0",
])


def _make_ln_cfu(num_replicate, num_condition_pre, num_genotype, per_geno_values):
    """Build a (rep, time=2, cond_pre, cond_sel=2, tname=1, tconc=3, geno) array."""
    shape = (num_replicate, 2, num_condition_pre, 2, 1, 3, num_genotype)
    arr = np.zeros(shape)
    for g, v in enumerate(per_geno_values):
        arr[..., g] = v
    return arr


def _all_library_masks(num_genotype):
    return jnp.ones((1, num_genotype), dtype=bool)


MockPreSplitData = namedtuple("MockPreSplitData", [
    "ln_cfu_t0", "ln_cfu_t0_std", "good_mask",
])


def _make_presplit(num_replicate, num_condition_pre, num_genotype,
                   per_geno_values, valid_mask=None):
    """Build a MockPreSplitData with shape (rep, cond_pre, geno), constant
    per genotype.  ``valid_mask`` (shape (geno,)) marks covered genotypes."""
    shape = (num_replicate, num_condition_pre, num_genotype)
    ln_cfu_t0 = np.zeros(shape)
    for g, v in enumerate(per_geno_values):
        ln_cfu_t0[..., g] = v

    if valid_mask is None:
        valid_mask = np.ones(num_genotype, dtype=bool)
    good_mask = np.zeros(shape, dtype=bool)
    good_mask[..., valid_mask] = True

    return MockPreSplitData(
        ln_cfu_t0=ln_cfu_t0, ln_cfu_t0_std=np.ones(shape), good_mask=good_mask,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data():
    """4 library genotypes, 2 replicates, 2 conditions, all ln_cfu=8."""
    R, C, G = 2, 2, 4
    ln_cfu = _make_ln_cfu(R, C, G, [8.0, 8.0, 8.0, 8.0])
    return MockGrowthData(
        num_replicate=R, num_condition_pre=C, num_genotype=G, batch_size=G,
        batch_idx=jnp.arange(G, dtype=jnp.int32),
        scale_vector=jnp.ones(G, dtype=float),
        ln_cfu0_spiked_mask=jnp.zeros(G, dtype=bool),
        ln_cfu0_wt_mask=jnp.zeros(G, dtype=bool),
        ln_cfu0_library_masks=_all_library_masks(G),
        num_ln_cfu0_library_classes=1,
        ln_cfu=ln_cfu,
        good_mask=np.ones_like(ln_cfu, dtype=bool),
        map_ln_cfu0=jnp.arange(G, dtype=jnp.int32),
    )


@pytest.fixture
def mock_data_varied():
    """
    6 genotypes: wt(12), spiked(10, 11), library(8, 7, 9). 2 replicates, 2 conditions.

    Condition 0: genotype values as above.
    Condition 1: each value shifted by +0.5 (tube_offset = 0.5 for all replicates).
    """
    R, C, G = 2, 2, 6
    per_geno = [12.0, 10.0, 11.0, 8.0, 7.0, 9.0]
    shape = (R, 2, C, 2, 1, 3, G)  # (rep, time, cond_pre, cond_sel, tname, tconc, geno)
    ln_cfu = np.zeros(shape)
    for g, v in enumerate(per_geno):
        ln_cfu[:, :, 0, :, :, :, g] = v          # condition_pre 0
        ln_cfu[:, :, 1, :, :, :, g] = v + 0.5    # condition_pre 1: tube_offset = +0.5

    return MockGrowthData(
        num_replicate=R, num_condition_pre=C, num_genotype=G, batch_size=G,
        batch_idx=jnp.arange(G, dtype=jnp.int32),
        scale_vector=jnp.ones(G, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, True, False, False, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False, False, False, False]),
        ln_cfu0_library_masks=_all_library_masks(G),
        num_ln_cfu0_library_classes=1,
        ln_cfu=ln_cfu,
        good_mask=np.ones(shape, dtype=bool),
        map_ln_cfu0=jnp.arange(G, dtype=jnp.int32),
    )


@pytest.fixture
def mock_data_two_classes():
    """7 genotypes: wt, spiked, 3 singles (class 0), 2 doubles (class 1). 2R, 2C."""
    R, C, G = 2, 2, 7
    ln_cfu_vals = [12.0, 10.0, 8.0, 7.0, 9.0, 11.0, 13.0]
    ln_cfu = _make_ln_cfu(R, C, G, ln_cfu_vals)
    singles_mask = jnp.array([False, False, True,  True,  True,  False, False])
    doubles_mask = jnp.array([False, False, False, False, False, True,  True ])
    return MockGrowthData(
        num_replicate=R, num_condition_pre=C, num_genotype=G, batch_size=G,
        batch_idx=jnp.arange(G, dtype=jnp.int32),
        scale_vector=jnp.ones(G, dtype=float),
        ln_cfu0_spiked_mask=jnp.array([False, True, False, False, False, False, False]),
        ln_cfu0_wt_mask=jnp.array([True, False, False, False, False, False, False]),
        ln_cfu0_library_masks=jnp.stack([singles_mask, doubles_mask]),
        num_ln_cfu0_library_classes=2,
        ln_cfu=ln_cfu,
        good_mask=np.ones_like(ln_cfu, dtype=bool),
        map_ln_cfu0=jnp.arange(G, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------

def _default_subs(name, data, priors, *, offset_geno=None, tube_offset=None):
    """Build a substitution dict with deterministic values for all sample sites."""
    R, C, B = data.num_replicate, data.num_condition_pre, data.batch_size
    num_classes = getattr(data, "num_ln_cfu0_library_classes", 1)
    guesses = get_guesses(name, data)
    subs = {
        f"{name}_spiked_loc": jnp.array(float(guesses[f"{name}_spiked_loc"])),
        f"{name}_wt_loc":     jnp.array(float(guesses[f"{name}_wt_loc"])),
        f"{name}_tube_scale": jnp.array(0.5),
        f"{name}_offset_geno": offset_geno if offset_geno is not None
                               else jnp.zeros((R, B), dtype=float),
        f"{name}_tube_offset": tube_offset if tube_offset is not None
                               else jnp.zeros((R, C), dtype=float),
    }
    for i in range(num_classes):
        subs[f"{name}_hyper_loc_{i}"]   = jnp.array(float(guesses[f"{name}_hyper_loc_{i}"]))
        subs[f"{name}_hyper_scale_{i}"] = jnp.array(float(guesses[f"{name}_hyper_scale_{i}"]))
    return subs


# ---------------------------------------------------------------------------
# Tests: get_hyperparameters
# ---------------------------------------------------------------------------

def test_get_hyperparameters_has_tube_scale_loc():
    params = get_hyperparameters()
    assert "ln_cfu0_tube_scale_loc" in params
    assert params["ln_cfu0_tube_scale_loc"] == pytest.approx(_FALLBACK_TUBE_SCALE)


def test_get_hyperparameters_multi_class():
    params = get_hyperparameters(num_classes=3)
    assert params["ln_cfu0_hyper_loc_locs"].shape == (3,)
    assert params["ln_cfu0_hyper_scale_locs"].shape == (3,)


# ---------------------------------------------------------------------------
# Tests: get_priors
# ---------------------------------------------------------------------------

def test_get_priors_no_data_returns_defaults():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.ln_cfu0_tube_scale_loc == pytest.approx(_FALLBACK_TUBE_SCALE)


def test_get_priors_accepts_data_keyword():
    sig = inspect.signature(get_priors)
    assert "data" in sig.parameters
    assert sig.parameters["data"].default is None


def test_get_priors_accepts_presplit_keyword():
    """get_priors must accept an optional `presplit` keyword and forward it
    to the shared empirical estimator (ModelOrchestrator detects this)."""
    sig = inspect.signature(get_priors)
    assert "presplit" in sig.parameters
    assert sig.parameters["presplit"].default is None


def test_get_priors_with_data_overrides_subgroup_scales(mock_data_varied):
    priors = get_priors(data=mock_data_varied)
    # wt: single value → floored; spiked: two values ±0.5 from median
    from tfscreen.tfmodel.generative.components.ln_cfu0.hierarchical import _SCALE_FLOOR
    assert priors.ln_cfu0_wt_scale == pytest.approx(_SCALE_FLOOR)
    assert priors.ln_cfu0_spiked_scale == pytest.approx(1.4826 * 0.5, rel=1e-4)


def test_get_priors_with_presplit_overrides_wt_scale(mock_data_varied):
    """
    presplit is correctly forwarded through to the shared empirical
    estimator: a wt presplit spread (rather than the degenerate single-value
    ln_cfu spread) drives ln_cfu0_wt_scale.
    """
    R, C, G = mock_data_varied.num_replicate, mock_data_varied.num_condition_pre, \
        mock_data_varied.num_genotype
    presplit_vals = np.zeros((R, C, G))
    presplit_vals[0, :, 0] = 10.0
    presplit_vals[1, :, 0] = 14.0
    presplit_good = np.zeros((R, C, G), dtype=bool)
    presplit_good[:, :, 0] = True
    presplit = MockPreSplitData(
        ln_cfu_t0=presplit_vals, ln_cfu_t0_std=np.ones((R, C, G)),
        good_mask=presplit_good,
    )

    priors = get_priors(data=mock_data_varied, presplit=presplit)
    # loc = median(10,10,14,14) = 12; MAD=2; scale = 1.4826*2
    from tfscreen.tfmodel.generative.components.ln_cfu0.hierarchical import _SCALE_FLOOR
    assert priors.ln_cfu0_wt_scale == pytest.approx(1.4826 * 2.0, rel=1e-4)
    assert priors.ln_cfu0_wt_scale != pytest.approx(_SCALE_FLOOR)


# ---------------------------------------------------------------------------
# Tests: get_guesses – shape and structure
# ---------------------------------------------------------------------------

def test_get_guesses_offset_geno_shape(mock_data):
    R, G = mock_data.num_replicate, mock_data.num_genotype
    guesses = get_guesses("x", mock_data)
    assert f"x_offset_geno" in guesses
    assert guesses["x_offset_geno"].shape == (R, G)


def test_get_guesses_tube_offset_shape(mock_data):
    R, C = mock_data.num_replicate, mock_data.num_condition_pre
    guesses = get_guesses("x", mock_data)
    assert f"x_tube_offset" in guesses
    assert guesses["x_tube_offset"].shape == (R, C)


def test_get_guesses_tube_scale_present(mock_data):
    guesses = get_guesses("x", mock_data)
    assert "x_tube_scale" in guesses
    assert float(guesses["x_tube_scale"]) > 0


def test_get_guesses_uniform_data_offsets_zero(mock_data):
    """All genotypes at same ln_cfu → zero offsets and zero tube offsets."""
    guesses = get_guesses("x", mock_data)
    assert jnp.allclose(guesses["x_offset_geno"], 0.0)
    assert jnp.allclose(guesses["x_tube_offset"], 0.0)


def test_get_guesses_tube_offset_from_condition_shift(mock_data_varied):
    """
    Condition 1 is uniformly +0.5 above condition 0.  The estimated tube offsets
    should reflect this systematic shift: tube_offset[:, 1] ≈ +0.25 and
    tube_offset[:, 0] ≈ −0.25 (median-centred within each replicate).
    """
    guesses = get_guesses("x", mock_data_varied)
    tube_offset = np.array(guesses["x_tube_offset"])  # (R, C)

    # The two conditions differ by exactly 0.5; after centering on the
    # per-genotype median, their residuals are ±0.25.
    diff = tube_offset[:, 1] - tube_offset[:, 0]
    assert np.allclose(diff, 0.5, atol=0.02)


def test_get_guesses_offset_geno_not_condition_pre_indexed(mock_data):
    """offset_geno has shape (R, G), NOT (R, C, G) — the key structural check."""
    guesses = get_guesses("x", mock_data)
    R, G = mock_data.num_replicate, mock_data.num_genotype
    assert guesses["x_offset_geno"].shape == (R, G)
    assert guesses["x_offset_geno"].ndim == 2


def test_get_guesses_accepts_presplit_keyword():
    """get_guesses must accept an optional `presplit` keyword and forward it
    to the shared empirical estimator (ModelOrchestrator detects this)."""
    sig = inspect.signature(get_guesses)
    assert "presplit" in sig.parameters
    assert sig.parameters["presplit"].default is None


def test_get_guesses_wt_loc_prefers_presplit(mock_data_varied):
    """wt_loc guess uses the presplit-derived value instead of the ln_cfu
    median when presplit covers the wt genotype."""
    presplit = _make_presplit(
        num_replicate=2, num_condition_pre=2, num_genotype=6,
        per_geno_values=[20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        valid_mask=np.array([True, False, False, False, False, False]),
    )
    guesses = get_guesses("x", mock_data_varied, presplit=presplit)
    assert guesses["x_wt_loc"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests: define_model – return shape
# ---------------------------------------------------------------------------

def test_define_model_return_shape(mock_data):
    """Return tensor has shape (R, 1, C, 1, 1, 1, batch)."""
    name   = "lnc0"
    priors = get_priors()
    subs   = _default_subs(name, mock_data, priors)

    model  = substitute(define_model, data=subs)
    result = model(name=name, data=mock_data, priors=priors)

    R, C, B = mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size
    assert result.shape == (R, 1, C, 1, 1, 1, B)


def test_define_model_deterministic_site_shape(mock_data):
    """Deterministic site ``name`` has shape (R, C, batch)."""
    name   = "lnc0"
    priors = get_priors()
    subs   = _default_subs(name, mock_data, priors)

    model_tr = trace(substitute(define_model, data=subs)).get_trace(
        name=name, data=mock_data, priors=priors
    )

    R, C, B = mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size
    assert model_tr[name]["value"].shape == (R, C, B)


# ---------------------------------------------------------------------------
# Tests: define_model – factoring correctness
# ---------------------------------------------------------------------------

def test_define_model_tube_offset_zero_gives_geno_baseline(mock_data):
    """
    When tube_offset = 0, the output equals geno_baseline broadcast across
    conditions — every condition has the same ln_cfu0 for each (rep, geno).
    """
    name   = "lnc0"
    priors = get_priors()
    subs   = _default_subs(name, mock_data, priors)  # tube_offset already 0

    model = substitute(define_model, data=subs)
    tr    = trace(model).get_trace(name=name, data=mock_data, priors=priors)
    site  = tr[name]["value"]  # (R, C, batch)

    # All conditions should be identical
    for c in range(mock_data.num_condition_pre):
        assert jnp.allclose(site[:, 0, :], site[:, c, :])


def test_define_model_offset_geno_zero_gives_hyper_loc_plus_tube(mock_data):
    """
    When offset_geno = 0, geno_baseline = hyper_loc, so
    ln_cfu0[r, c, g] = hyper_loc + tube_offset[r, c] for all library g.
    """
    name   = "lnc0"
    R, C, B = mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size

    known_tube = jnp.array([[0.1, 0.3], [0.2, 0.4]])  # (R, C)
    hyper_loc  = 8.0
    priors     = get_priors()
    subs       = {
        f"{name}_hyper_loc_0":   jnp.array(hyper_loc),
        f"{name}_hyper_scale_0": jnp.array(1.0),
        f"{name}_spiked_loc":    jnp.array(10.0),
        f"{name}_wt_loc":        jnp.array(12.0),
        f"{name}_tube_scale":    jnp.array(0.5),
        f"{name}_offset_geno":   jnp.zeros((R, B)),
        f"{name}_tube_offset":   known_tube,
    }

    model = substitute(define_model, data=subs)
    tr    = trace(model).get_trace(name=name, data=mock_data, priors=priors)
    site  = tr[name]["value"]  # (R, C, batch) — all library genotypes

    for r in range(R):
        for c in range(C):
            expected = hyper_loc + float(known_tube[r, c])
            assert jnp.allclose(site[r, c, :], expected), \
                f"r={r}, c={c}: expected {expected}, got {site[r, c, :]}"


def test_define_model_tube_offset_shifts_all_genotypes_equally(mock_data):
    """
    Adding a tube_offset shifts every genotype in that (rep, cond) by the same
    amount regardless of the genotype-specific offset.
    """
    name  = "lnc0"
    R, C, B = mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size
    priors = get_priors()

    offset_geno = jnp.arange(R * B, dtype=float).reshape(R, B)
    tube_delta  = 2.5

    subs_zero = _default_subs(name, mock_data, priors, offset_geno=offset_geno)
    subs_tube = _default_subs(name, mock_data, priors, offset_geno=offset_geno,
                              tube_offset=jnp.full((R, C), tube_delta))

    tr_zero = trace(substitute(define_model, data=subs_zero)).get_trace(
        name=name, data=mock_data, priors=priors)
    tr_tube = trace(substitute(define_model, data=subs_tube)).get_trace(
        name=name, data=mock_data, priors=priors)

    diff = tr_tube[name]["value"] - tr_zero[name]["value"]
    assert jnp.allclose(diff, tube_delta)


def test_define_model_geno_baseline_shared_across_conditions(mock_data):
    """
    Non-zero offset_geno changes the value the same way for every condition —
    the geno_baseline component does not depend on condition_pre.
    """
    name  = "lnc0"
    R, C, B = mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size
    priors = get_priors()

    offset_geno = jnp.ones((R, B)) * 1.5
    subs = _default_subs(name, mock_data, priors, offset_geno=offset_geno)

    tr   = trace(substitute(define_model, data=subs)).get_trace(
        name=name, data=mock_data, priors=priors)
    site = tr[name]["value"]  # (R, C, batch)

    # Across conditions, values should differ only by tube_offset (=0 here)
    for c in range(C):
        assert jnp.allclose(site[:, 0, :], site[:, c, :])


# ---------------------------------------------------------------------------
# Tests: define_model – spiked / wt genotypes
# ---------------------------------------------------------------------------

def test_define_model_spiked_uses_spiked_loc(mock_data_varied):
    """Spiked genotypes (indices 1, 2) receive spiked_loc, not hyper_loc."""
    name   = "lnc0"
    priors = get_priors()
    spiked_loc = 10.5
    subs = _default_subs(name, mock_data_varied, priors)
    subs[f"{name}_spiked_loc"] = jnp.array(spiked_loc)
    subs[f"{name}_offset_geno"] = jnp.zeros(
        (mock_data_varied.num_replicate, mock_data_varied.batch_size))

    tr   = trace(substitute(define_model, data=subs)).get_trace(
        name=name, data=mock_data_varied, priors=priors)
    site = tr[name]["value"]  # (R, C, batch)

    # Spiked indices 1, 2: value = spiked_loc + tube_offset (=0)
    assert jnp.allclose(site[:, :, 1], spiked_loc)
    assert jnp.allclose(site[:, :, 2], spiked_loc)


def test_define_model_wt_uses_wt_loc(mock_data_varied):
    """Wt genotype (index 0) receives wt_loc."""
    name   = "lnc0"
    priors = get_priors()
    wt_loc = 13.0
    subs = _default_subs(name, mock_data_varied, priors)
    subs[f"{name}_wt_loc"] = jnp.array(wt_loc)
    subs[f"{name}_offset_geno"] = jnp.zeros(
        (mock_data_varied.num_replicate, mock_data_varied.batch_size))

    tr   = trace(substitute(define_model, data=subs)).get_trace(
        name=name, data=mock_data_varied, priors=priors)
    site = tr[name]["value"]

    assert jnp.allclose(site[:, :, 0], wt_loc)


# ---------------------------------------------------------------------------
# Tests: define_model – two library classes
# ---------------------------------------------------------------------------

def test_define_model_two_classes_separate_locs(mock_data_two_classes):
    """Class-1 doubles receive hyper_loc_1, not hyper_loc_0."""
    name   = "lnc0"
    data   = mock_data_two_classes
    priors = get_priors(data=data)
    R, C, B = data.num_replicate, data.num_condition_pre, data.batch_size

    subs = {
        f"{name}_hyper_loc_0":   jnp.array(8.0),
        f"{name}_hyper_scale_0": jnp.array(1.0),
        f"{name}_hyper_loc_1":   jnp.array(12.0),
        f"{name}_hyper_scale_1": jnp.array(1.0),
        f"{name}_spiked_loc":    jnp.array(10.0),
        f"{name}_wt_loc":        jnp.array(13.0),
        f"{name}_tube_scale":    jnp.array(0.5),
        f"{name}_offset_geno":   jnp.zeros((R, B)),
        f"{name}_tube_offset":   jnp.zeros((R, C)),
    }

    tr   = trace(substitute(define_model, data=subs)).get_trace(
        name=name, data=data, priors=priors)
    site = tr[name]["value"]  # (R, C, batch)

    assert jnp.allclose(site[:, :, 0], 13.0)  # wt
    assert jnp.allclose(site[:, :, 1], 10.0)  # spiked
    # singles (class 0): indices 2,3,4
    for g in [2, 3, 4]:
        assert jnp.allclose(site[:, :, g], 8.0)
    # doubles (class 1): indices 5,6
    for g in [5, 6]:
        assert jnp.allclose(site[:, :, g], 12.0)


# ---------------------------------------------------------------------------
# Tests: guide – shapes and structure
# ---------------------------------------------------------------------------

def test_guide_return_shape(mock_data):
    """Guide returns the same shape as define_model."""
    name   = "lnc0"
    priors = get_priors()
    R, C, B = mock_data.num_replicate, mock_data.num_condition_pre, mock_data.batch_size

    with seed(rng_seed=0):
        result = guide(name=name, data=mock_data, priors=priors)

    assert result.shape == (R, 1, C, 1, 1, 1, B)


def test_guide_offset_geno_param_shape(mock_data):
    """Variational params for offset_geno have shape (R, G), not (R, C, G)."""
    name   = "lnc0"
    priors = get_priors()
    R, G   = mock_data.num_replicate, mock_data.num_genotype

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    assert f"{name}_offset_geno_locs" in tr
    assert tr[f"{name}_offset_geno_locs"]["value"].shape == (R, G)
    assert tr[f"{name}_offset_geno_scales"]["value"].shape == (R, G)


def test_guide_tube_offset_param_shape(mock_data):
    """Variational params for tube_offset have shape (R, C)."""
    name   = "lnc0"
    priors = get_priors()
    R, C   = mock_data.num_replicate, mock_data.num_condition_pre

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    assert f"{name}_tube_offset_locs" in tr
    assert tr[f"{name}_tube_offset_locs"]["value"].shape == (R, C)
    assert tr[f"{name}_tube_offset_scales"]["value"].shape == (R, C)


def test_guide_has_tube_scale_sample_site(mock_data):
    """Guide registers a sample site for tube_scale."""
    name   = "lnc0"
    priors = get_priors()

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    assert f"{name}_tube_scale" in tr
    assert tr[f"{name}_tube_scale"]["type"] == "sample"


def test_guide_offset_geno_sample_shape(mock_data):
    """Sampled offset_geno in guide has shape (R, batch)."""
    name   = "lnc0"
    priors = get_priors()
    R, B   = mock_data.num_replicate, mock_data.batch_size

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    assert tr[f"{name}_offset_geno"]["value"].shape == (R, B)


def test_guide_tube_offset_sample_shape(mock_data):
    """Sampled tube_offset in guide has shape (R, C)."""
    name   = "lnc0"
    priors = get_priors()
    R, C   = mock_data.num_replicate, mock_data.num_condition_pre

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    assert tr[f"{name}_tube_offset"]["value"].shape == (R, C)


def test_guide_two_classes_has_per_class_params(mock_data_two_classes):
    """Two library classes produce per-class hyper_loc/scale variational params."""
    name   = "lnc0"
    priors = get_priors(data=mock_data_two_classes)

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(
            name=name, data=mock_data_two_classes, priors=priors)

    for i in range(2):
        assert f"{name}_hyper_loc_{i}" in tr
        assert f"{name}_hyper_scale_{i}" in tr


# ---------------------------------------------------------------------------
# Tests: model/guide sample-site compatibility (SVI sanity)
# ---------------------------------------------------------------------------

def test_model_and_guide_have_compatible_sample_sites(mock_data):
    """Every model sample site (non-obs) has a matching guide sample site."""
    name   = "compat"
    priors = get_priors()

    with seed(rng_seed=0):
        model_tr = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors)
        guide_tr = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors)

    model_samples = {
        n for n, s in model_tr.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {
        n for n, s in guide_tr.items() if s["type"] == "sample"
    }

    assert model_samples == guide_samples, (
        f"model/guide sample sites differ:\n"
        f"  model only: {model_samples - guide_samples}\n"
        f"  guide only: {guide_samples - model_samples}"
    )


def test_model_and_guide_compatible_two_classes(mock_data_two_classes):
    """Model/guide symmetry holds for two library classes."""
    name   = "compat2"
    priors = get_priors(data=mock_data_two_classes)

    with seed(rng_seed=0):
        model_tr = trace(define_model).get_trace(
            name=name, data=mock_data_two_classes, priors=priors)
        guide_tr = trace(guide).get_trace(
            name=name, data=mock_data_two_classes, priors=priors)

    model_samples = {
        n for n, s in model_tr.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {
        n for n, s in guide_tr.items() if s["type"] == "sample"
    }

    assert model_samples == guide_samples, (
        f"model/guide sample sites differ:\n"
        f"  model only: {model_samples - guide_samples}\n"
        f"  guide only: {guide_samples - model_samples}"
    )


# ---------------------------------------------------------------------------
# Tests: pinning
# ---------------------------------------------------------------------------

def test_pinnable_suffixes():
    assert set(_PINNABLE_SUFFIXES) == {"hyper_loc", "hyper_scale"}


def test_define_model_pinned_becomes_deterministic(mock_data):
    """Pinned hyper_loc_0 is registered as a deterministic site."""
    name   = "lnc0"
    pinned = {"hyper_loc_0": 8.0, "hyper_scale_0": 1.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors)

    assert tr[f"{name}_hyper_loc_0"]["type"] == "deterministic"
    assert jnp.allclose(tr[f"{name}_hyper_loc_0"]["value"], 8.0)
    assert tr[f"{name}_hyper_scale_0"]["type"] == "deterministic"
    # spiked/wt and tube sites remain sample sites
    assert tr[f"{name}_spiked_loc"]["type"] == "sample"
    assert tr[f"{name}_tube_scale"]["type"] == "sample"


def test_guide_pinned_drops_variational_params(mock_data):
    """Pinned hyper has no variational params in the guide."""
    name   = "lnc0"
    pinned = {"hyper_loc_0": 8.0}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    assert f"{name}_hyper_loc_0" not in tr
    assert f"{name}_hyper_loc_0_loc" not in tr
    # Unpinned hyper_scale_0 still present
    assert f"{name}_hyper_scale_0" in tr
    # Other sites intact
    assert f"{name}_tube_scale" in tr
    assert f"{name}_offset_geno" in tr
    assert f"{name}_tube_offset" in tr


def test_model_and_guide_compatible_under_pinning(mock_data):
    """Model/guide sample-site symmetry holds when hypers are pinned."""
    name   = "compat_pin"
    pinned = {"hyper_loc_0": 8.0, "hyper_scale_0": 1.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        model_tr = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors)
        guide_tr = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors)

    model_samples = {
        n for n, s in model_tr.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {
        n for n, s in guide_tr.items() if s["type"] == "sample"
    }

    assert model_samples == guide_samples, (
        f"model only: {model_samples - guide_samples}\n"
        f"guide only: {guide_samples - model_samples}"
    )
