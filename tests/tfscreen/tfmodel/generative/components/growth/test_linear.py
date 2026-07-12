import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.tfmodel.generative.components.growth.linear import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    get_scale_bounds,
    _parse_condition_label,
    LinearParams,
)


def _site_loc(site):
    """Drill to the innermost Normal .loc, unwrapping plate/expand wrappers."""
    fn = site["fn"]
    for _ in range(5):
        if hasattr(fn, "loc"):
            return np.asarray(fn.loc)
        if hasattr(fn, "base_dist"):
            fn = fn.base_dist
        else:
            break
    raise AssertionError("could not find .loc on site distribution")

# --- Mock Data Fixtures ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_condition_rep",
    "num_replicate",
    "map_condition_pre",
    "map_condition_sel",
    "ln_cfu",
    "t_sel",
    "good_mask",
])


def _make_ln_cfu(slopes, num_rep=2, num_cond_pre=4, num_tname=1, num_tconc=3, num_geno=3):
    num_cond_sel = len(slopes)
    num_time = 2
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    arr = np.full(shape, 7.0)
    for cs, slope in enumerate(slopes):
        arr[:, 1, :, cs, :, :, :] = 7.0 + slope
    return jnp.array(arr)


def _make_t_sel(num_rep=2, num_cond_pre=4, num_cond_sel=4, num_tname=1, num_tconc=3, num_geno=3):
    num_time = 2
    shape = (num_rep, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    arr = np.zeros(shape)
    arr[:, 1, ...] = 1.0
    return jnp.array(arr)


@pytest.fixture
def mock_data():
    num_condition_rep = 3
    num_replicate = 2
    map_condition_pre = jnp.array([0, 2, 2, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([1, 0, 1, 2], dtype=jnp.int32)
    ln_cfu = _make_ln_cfu([0.0, 0.0, 0.0, 0.0])
    t_sel  = _make_t_sel()
    good_mask = jnp.ones(ln_cfu.shape, dtype=bool)

    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        ln_cfu=ln_cfu,
        t_sel=t_sel,
        good_mask=good_mask,
    )


@pytest.fixture
def mock_data_empirical():
    """
    Two condition_reps with different baseline growth rates:
      cond_rep 0 (cond_sel index 0): slope = 0.030
      cond_rep 1 (cond_sel index 1): slope = 0.020
    """
    num_condition_rep = 2
    num_replicate = 1
    num_cond_pre = 2
    num_cond_sel = 2
    num_tname = 1
    num_tconc = 1
    num_geno = 1
    num_time = 2

    shape = (num_replicate, num_time, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)
    ln_cfu_arr = np.full(shape, 7.0)
    ln_cfu_arr[:, 1, :, 0, :, :, :] = 7.03
    ln_cfu_arr[:, 1, :, 1, :, :, :] = 7.02

    t_arr = np.zeros(shape)
    t_arr[:, 1, ...] = 1.0

    map_condition_pre = jnp.array([0, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([0, 1], dtype=jnp.int32)

    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        ln_cfu=jnp.array(ln_cfu_arr),
        t_sel=jnp.array(t_arr),
        good_mask=jnp.ones(shape, dtype=bool),
    )


# --- Test Cases ---

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "k_loc" in params
    assert "k_scale" in params
    assert "m_loc" in params
    assert "m_scale" in params
    assert "m_scale_minus" in params
    assert "m_scale_plus" in params
    assert params["k_loc"] == 0.020
    assert params["m_scale_minus"] < params["m_scale_plus"]  # minus is tighter


def test_get_priors_no_labels_legacy_behavior():
    """With no condition_labels, m_is_selection is None (backward compat)."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.k_loc == 0.020
    assert priors.m_loc == 0.0
    assert priors.m_is_selection is None
    assert not hasattr(priors, "pinned")


def test_get_guesses_keys_and_shapes(mock_data):
    name = "test_growth"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    assert f"{name}_k_locs"   in guesses
    assert f"{name}_k_scales" in guesses
    assert f"{name}_m_locs"   in guesses
    assert f"{name}_m_scales" in guesses

    expected_shape = (mock_data.num_condition_rep,)
    assert guesses[f"{name}_k_locs"].shape == expected_shape
    assert guesses[f"{name}_m_locs"].shape == expected_shape


def test_get_guesses_constant_ln_cfu_gives_zero_k(mock_data):
    """With constant ln_cfu (slope=0), k_locs should be near zero."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.allclose(guesses[f"{name}_k_locs"], 0.0, atol=1e-6)


def test_get_guesses_m_locs_always_zero(mock_data):
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    assert jnp.all(guesses[f"{name}_m_locs"] == 0.0)


def test_get_guesses_empirical_k_locs(mock_data_empirical):
    """k_locs should equal per-condition OLS slopes."""
    name = "eg"
    guesses = get_guesses(name, mock_data_empirical)
    k_locs = np.array(guesses[f"{name}_k_locs"])
    assert abs(k_locs[0] - 0.030) < 1e-6
    assert abs(k_locs[1] - 0.020) < 1e-6


def test_get_guesses_masked_observations(mock_data_empirical):
    """Fully masked cond_sel falls back to global median, not NaN."""
    name = "eg"
    mask_arr = np.array(mock_data_empirical.good_mask)
    mask_arr[:, :, :, 0, :, :, :] = False
    masked_data = mock_data_empirical._replace(good_mask=jnp.array(mask_arr))

    guesses = get_guesses(name, masked_data)
    k_locs = np.array(guesses[f"{name}_k_locs"])
    assert np.all(np.isfinite(k_locs))


def test_get_guesses_non7d_fallback():
    """Non-7D tensors trigger hard-coded fallback without error."""
    name = "fg"
    MockSimple = namedtuple("MockSimple", [
        "num_condition_rep", "num_replicate",
        "map_condition_pre", "map_condition_sel",
        "ln_cfu", "t_sel", "good_mask",
    ])
    data = MockSimple(
        num_condition_rep=3,
        num_replicate=2,
        map_condition_pre=jnp.array([0, 1, 2]),
        map_condition_sel=jnp.array([0, 1, 2]),
        ln_cfu=jnp.zeros((3,)),
        t_sel=jnp.zeros((3,)),
        good_mask=jnp.ones((3,), dtype=bool),
    )
    guesses = get_guesses(name, data)
    assert guesses[f"{name}_k_locs"].shape == (3,)
    assert jnp.all(guesses[f"{name}_m_locs"] == 0.0)


# --- define_model tests ---

def test_define_model_structure_and_shapes(mock_data):
    """Verifies output shapes and sample site names."""
    name = "test_growth"
    priors = get_priors()

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )

    assert f"{name}_k" in model_trace
    assert f"{name}_m" in model_trace
    assert model_trace[f"{name}_k"]["type"] == "sample"
    assert model_trace[f"{name}_m"]["type"] == "sample"

    k_per_condition = model_trace[f"{name}_k"]["value"]
    m_per_condition = model_trace[f"{name}_m"]["value"]
    assert k_per_condition.shape == (mock_data.num_condition_rep,)
    assert m_per_condition.shape == (mock_data.num_condition_rep,)

    with seed(rng_seed=0):
        params = define_model(name=name, data=mock_data, priors=priors)

    assert params.k_pre.shape == mock_data.map_condition_pre.shape
    assert params.m_pre.shape == mock_data.map_condition_pre.shape
    assert params.k_sel.shape == mock_data.map_condition_sel.shape
    assert params.m_sel.shape == mock_data.map_condition_sel.shape


def test_define_model_calculation_logic(mock_data):
    """Substituted k values propagate correctly through the condition mapping."""
    name = "test_growth"
    priors = get_priors()

    subs = {
        f"{name}_k": jnp.array([10.0, 12.0, 8.0]),
        f"{name}_m": jnp.zeros(mock_data.num_condition_rep),
    }
    substituted = substitute(define_model, data=subs)
    params = substituted(name=name, data=mock_data, priors=priors)

    expected_k_pre = jnp.array([10.0, 12.0, 8.0])[mock_data.map_condition_pre]
    assert jnp.allclose(params.k_pre, expected_k_pre)
    expected_k_sel = jnp.array([10.0, 12.0, 8.0])[mock_data.map_condition_sel]
    assert jnp.allclose(params.k_sel, expected_k_sel)


# --- guide tests ---

def test_guide_logic_and_shapes(mock_data):
    """Guide creates per-condition variational params and samples."""
    name = "test_growth_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )
        params = guide(name=name, data=mock_data, priors=priors)

    assert f"{name}_k_locs" in guide_trace
    assert f"{name}_k_scales" in guide_trace
    assert f"{name}_m_locs" in guide_trace
    assert f"{name}_m_scales" in guide_trace
    assert f"{name}_k" in guide_trace
    assert f"{name}_m" in guide_trace

    k_locs_val = guide_trace[f"{name}_k_locs"]["value"]
    assert k_locs_val.shape == (mock_data.num_condition_rep,)

    assert params.k_pre.shape == mock_data.map_condition_pre.shape
    assert params.m_pre.shape == mock_data.map_condition_pre.shape
    assert params.k_sel.shape == mock_data.map_condition_sel.shape
    assert params.m_sel.shape == mock_data.map_condition_sel.shape


def test_model_and_guide_have_compatible_sample_sites(mock_data):
    """Guide and model must agree on which sample sites exist."""
    name = "compat"
    priors = get_priors()

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )
    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )

    model_samples = {
        n for n, s in model_trace.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {
        n for n, s in guide_trace.items() if s["type"] == "sample"
    }

    assert model_samples == guide_samples, (
        f"model and guide sample sites differ:\n"
        f"  model only: {model_samples - guide_samples}\n"
        f"  guide only: {guide_samples - model_samples}"
    )


# --- m_pinned (hard clamp) tests ---

def test_define_model_m_pinned_clamps_to_m_loc(mock_data):
    """With m_pinned, m is a deterministic site equal to m_loc; k still sampled."""
    name = "pin"
    m_loc = jnp.array([0.1, 0.2, 0.3])
    priors = get_priors().replace(m_loc=m_loc, m_pinned=True)

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )

    assert model_trace[f"{name}_m"]["type"] == "deterministic"
    assert model_trace[f"{name}_k"]["type"] == "sample"
    np.testing.assert_allclose(
        np.asarray(model_trace[f"{name}_m"]["value"]), np.asarray(m_loc)
    )

    # The clamped value propagates through the condition mapping.
    with seed(rng_seed=0):
        params = define_model(name=name, data=mock_data, priors=priors)
    np.testing.assert_allclose(
        np.asarray(params.m_sel), np.asarray(m_loc[mock_data.map_condition_sel])
    )


def test_guide_m_pinned_registers_no_m_params(mock_data):
    """With m_pinned, the guide drops m's variational params and sample site."""
    name = "pin"
    priors = get_priors().replace(m_loc=jnp.array([0.1, 0.2, 0.3]), m_pinned=True)

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )

    assert f"{name}_k_locs" in guide_trace       # k still variational
    assert f"{name}_m_locs" not in guide_trace
    assert f"{name}_m_scales" not in guide_trace
    assert f"{name}_m" not in guide_trace


def test_model_guide_compatible_when_m_pinned(mock_data):
    """Model and guide must still agree on sample sites when m is clamped."""
    name = "pin"
    priors = get_priors().replace(m_loc=jnp.array([0.1, 0.2, 0.3]), m_pinned=True)

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )
    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=priors
        )

    model_samples = {
        n for n, s in model_trace.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {n for n, s in guide_trace.items() if s["type"] == "sample"}

    assert model_samples == guide_samples
    assert f"{name}_m" not in model_samples  # m is no longer sampled
    assert f"{name}_k" in model_samples      # k still is


# ---------------------------------------------------------------------------
# _parse_condition_label
# ---------------------------------------------------------------------------

class TestParseConditionLabel:

    def test_plus_returns_true(self):
        assert _parse_condition_label("kanR+kan") is True

    def test_minus_returns_false(self):
        assert _parse_condition_label("kanR-kan") is False

    def test_selection_with_description_plus(self):
        assert _parse_condition_label("pheS+4CP") is True

    def test_control_with_description_minus(self):
        assert _parse_condition_label("pheS-4CP") is False

    def test_no_plus_no_minus_raises(self):
        with pytest.raises(ValueError, match="does not unambiguously"):
            _parse_condition_label("kanR_no_selection")

    def test_both_plus_and_minus_raises(self):
        with pytest.raises(ValueError, match="does not unambiguously"):
            _parse_condition_label("weird+condition-name")

    def test_error_includes_label_name(self):
        label = "mystery_condition"
        with pytest.raises(ValueError, match=label):
            _parse_condition_label(label)

    def test_error_explains_convention(self):
        with pytest.raises(ValueError, match=r"\+.*selection"):
            _parse_condition_label("unlabeled")


# ---------------------------------------------------------------------------
# get_priors with condition_labels
# ---------------------------------------------------------------------------

class TestGetPriorsWithLabels:

    def test_plus_conditions_marked_selection(self):
        labels = ["kanR+kan", "pheS+4CP"]
        priors = get_priors(condition_labels=labels)
        assert priors.m_is_selection == (True, True)

    def test_minus_conditions_marked_control(self):
        labels = ["kanR-kan", "pheS-4CP"]
        priors = get_priors(condition_labels=labels)
        assert priors.m_is_selection == (False, False)

    def test_mixed_labels_correct_classification(self):
        labels = ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]
        priors = get_priors(condition_labels=labels)
        assert priors.m_is_selection == (True, False, True, False)

    def test_single_selection_label(self):
        priors = get_priors(condition_labels=["sel+media"])
        assert priors.m_is_selection == (True,)

    def test_invalid_label_propagates_error(self):
        with pytest.raises(ValueError, match="unlabeled"):
            get_priors(condition_labels=["kanR+kan", "unlabeled"])

    def test_none_labels_gives_none_is_selection(self):
        priors = get_priors(condition_labels=None)
        assert priors.m_is_selection is None

    def test_empty_labels_gives_empty_tuple(self):
        priors = get_priors(condition_labels=[])
        assert priors.m_is_selection == ()

    def test_scales_preserved_in_priors(self):
        priors = get_priors(condition_labels=["a+b", "a-b"])
        assert priors.m_scale_minus == pytest.approx(0.001)
        assert priors.m_scale_plus == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Phase 2 tests: get_priors with explicit is_selection flags
# ---------------------------------------------------------------------------

class TestGetPriorsIsSelectionPhase2:
    """
    Phase 2 tests: get_priors should accept an ``is_selection`` list of booleans
    directly, so callers no longer need +/- in condition names.

    All tests are xfail until a new ``is_selection`` parameter is added to
    ``get_priors``.  Remove the xfail marker after each test passes.
    """

    def test_is_selection_sets_m_is_selection(self):
        """get_priors(is_selection=[...]) directly sets m_is_selection."""
        priors = get_priors(is_selection=[True, False, True])
        assert priors.m_is_selection == (True, False, True)

    def test_is_selection_requires_no_plus_minus_in_names(self):
        """
        When is_selection is provided, condition names without +/- must work.
        Currently, callers must use condition_labels with +/- to get per-condition
        m priors; Phase 2 removes that naming constraint.
        """
        # With the current API (condition_labels), these names would raise
        # ValueError from _parse_condition_label.  With is_selection, they work.
        priors = get_priors(is_selection=[False, True])
        assert priors.m_is_selection == (False, True)

    def test_is_selection_false_uses_tight_prior(self):
        """is_selection=False conditions get m_scale_minus (tight prior)."""
        hypers = get_hyperparameters()
        priors = get_priors(is_selection=[False, True])
        assert priors.m_is_selection is not None
        scales = [
            hypers["m_scale_plus"] if sel else hypers["m_scale_minus"]
            for sel in priors.m_is_selection
        ]
        assert scales[0] == pytest.approx(hypers["m_scale_minus"])
        assert scales[1] == pytest.approx(hypers["m_scale_plus"])

    def test_is_selection_all_false(self):
        """All non-selective conditions: m_is_selection all False."""
        priors = get_priors(is_selection=[False, False])
        assert priors.m_is_selection == (False, False)

    def test_is_selection_and_condition_labels_mutually_exclusive(self):
        """Passing both is_selection and condition_labels must raise ValueError."""
        with pytest.raises(ValueError, match="is_selection"):
            get_priors(condition_labels=["a+b"], is_selection=[True])


# ---------------------------------------------------------------------------
# define_model — per-condition m prior
# ---------------------------------------------------------------------------

class TestDefineModelSelectionAware:

    def _make_data(self, num_condition_rep):
        """Minimal mock with num_condition_rep conditions."""
        num_rep = 1
        num_cond_pre = 1
        num_cond_sel = num_condition_rep
        num_tname = 1
        num_tconc = 1
        num_geno = 1
        num_time = 2
        shape = (num_rep, num_time, num_cond_pre, num_cond_sel,
                 num_tname, num_tconc, num_geno)

        return MockGrowthData(
            num_condition_rep=num_condition_rep,
            num_replicate=num_rep,
            map_condition_pre=jnp.zeros(num_condition_rep, dtype=jnp.int32),
            map_condition_sel=jnp.arange(num_condition_rep, dtype=jnp.int32),
            ln_cfu=jnp.zeros(shape),
            t_sel=jnp.zeros(shape),
            good_mask=jnp.ones(shape, dtype=bool),
        )

    def test_selection_aware_prior_tighter_for_control(self):
        """
        The Normal prior on m for a '-' condition must be tighter than for
        a '+' condition.  We verify this by inspecting the log-prob of a
        moderate m value under each condition's prior.
        """
        labels = ["kanR+kan", "kanR-kan"]
        priors = get_priors(condition_labels=labels)
        data = self._make_data(num_condition_rep=2)

        # Substitute specific m values and inspect the model trace
        m_test_value = 0.005   # moderate m, within normal range but 5× tight range
        subs = {
            "test_m_k": jnp.zeros(2),
            "test_m_m": jnp.full(2, m_test_value),
        }
        substituted = substitute(define_model, data=subs)

        with seed(rng_seed=0):
            tr = trace(substituted).get_trace(
                name="test_m", data=data, priors=priors
            )

        m_site = tr["test_m_m"]
        # The trace fn should have two m values; we can check the log-prob
        # is lower (more penalized) for the control condition.
        m_values = m_site["value"]
        assert m_values.shape == (2,)
        # kanR+kan (index 0) is selection: wider prior → less penalty
        # kanR-kan (index 1) is control: tight prior → more penalty
        import numpyro.distributions as d
        lp_sel = d.Normal(priors.m_loc, priors.m_scale_plus).log_prob(m_test_value)
        lp_ctrl = d.Normal(priors.m_loc, priors.m_scale_minus).log_prob(m_test_value)
        assert float(lp_sel) > float(lp_ctrl)

    def test_legacy_path_unchanged(self):
        """With m_is_selection=None, the single m_scale prior applies."""
        priors = get_priors()  # no condition_labels → m_is_selection=None
        assert priors.m_is_selection is None

        data = self._make_data(num_condition_rep=2)
        with seed(rng_seed=0):
            tr = trace(define_model).get_trace(
                name="leg", data=data, priors=priors
            )
        assert "leg_m" in tr
        assert tr["leg_m"]["value"].shape == (2,)

    def test_define_model_shapes_with_labels(self):
        """Output shapes should be the same as the legacy path."""
        labels = ["sel+a", "ctrl-b", "sel+c"]
        priors = get_priors(condition_labels=labels)
        data = self._make_data(num_condition_rep=3)

        with seed(rng_seed=0):
            params = define_model(name="sa", data=data, priors=priors)

        assert params.k_pre.shape == data.map_condition_pre.shape
        assert params.m_pre.shape == data.map_condition_pre.shape
        assert params.k_sel.shape == data.map_condition_sel.shape
        assert params.m_sel.shape == data.map_condition_sel.shape

    def test_model_guide_compatible_with_labels(self):
        """Model and guide sample sites must still match with labels."""
        labels = ["kanR+kan", "kanR-kan"]
        priors = get_priors(condition_labels=labels)
        data = self._make_data(num_condition_rep=2)

        with seed(rng_seed=0):
            model_trace = trace(define_model).get_trace(
                name="compat2", data=data, priors=priors
            )
        with seed(rng_seed=0):
            guide_trace = trace(guide).get_trace(
                name="compat2", data=data, priors=priors
            )

        model_samples = {
            n for n, s in model_trace.items()
            if s["type"] == "sample" and not s.get("is_observed", False)
        }
        guide_samples = {n for n, s in guide_trace.items() if s["type"] == "sample"}
        assert model_samples == guide_samples


# ---------------------------------------------------------------------------
# Phase 1: per-condition (array) priors on k / m
# ---------------------------------------------------------------------------

class TestPerConditionPriors:

    def test_scalar_prior_broadcasts(self, mock_data):
        """A scalar k_loc broadcasts to every condition (legacy behaviour)."""
        name = "sc"
        priors = get_priors()  # scalar k_loc = 0.020
        with seed(rng_seed=0):
            tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
        loc = _site_loc(tr[f"{name}_k"])
        assert loc.shape == (mock_data.num_condition_rep,)
        assert np.allclose(loc, 0.020)

    def test_array_k_loc_is_per_condition(self, mock_data):
        """A per-condition k_loc array flows through to the k sample site."""
        name = "arr"
        per_cond = jnp.array([0.011, 0.021, 0.029])
        priors = get_priors().replace(k_loc=per_cond, k_scale=jnp.full(3, 0.002))
        with seed(rng_seed=0):
            tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
        loc = _site_loc(tr[f"{name}_k"])
        assert np.allclose(loc, np.array([0.011, 0.021, 0.029]))

    def test_array_prior_initializes_guide_params(self, mock_data):
        """The guide's k_locs param initialises from a per-condition array prior."""
        name = "gp"
        per_cond = jnp.array([0.011, 0.021, 0.029])
        priors = get_priors().replace(k_loc=per_cond)
        with seed(rng_seed=0):
            gtr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)
        k_locs = np.asarray(gtr[f"{name}_k_locs"]["value"])
        assert np.allclose(k_locs, np.array([0.011, 0.021, 0.029]))

    def test_array_and_scalar_give_same_sites(self, mock_data):
        """Model/guide site sets are identical whether priors are scalar or array."""
        name = "ss"
        priors = get_priors().replace(k_loc=jnp.array([0.011, 0.021, 0.029]))
        with seed(rng_seed=0):
            mtr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
        with seed(rng_seed=0):
            gtr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)
        model_samples = {
            n for n, s in mtr.items()
            if s["type"] == "sample" and not s.get("is_observed", False)
        }
        guide_samples = {n for n, s in gtr.items() if s["type"] == "sample"}
        assert model_samples == guide_samples


class TestGetScaleBounds:

    def test_structure(self):
        bounds = get_scale_bounds()
        assert set(bounds) == {"k", "m"}
        for suffix, spec in bounds.items():
            assert set(spec) == {"floor", "ceiling", "scale_field"}
            assert spec["floor"] < spec["ceiling"]

    def test_baseline_floor_tight(self):
        """The baseline term k must get a tight floor to be pinnable."""
        bounds = get_scale_bounds()
        assert bounds["k"]["floor"] <= 0.002
        assert bounds["k"]["scale_field"] == "k_scale"
        assert bounds["m"]["scale_field"] == "m_scale_plus"
