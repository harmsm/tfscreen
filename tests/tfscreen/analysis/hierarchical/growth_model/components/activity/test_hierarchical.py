import pytest
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.activity.hierarchical import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    _PINNABLE_SUFFIXES,
)

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_genotype",
    "batch_size",
    "batch_idx", 
    "wt_indexes",
    "scale_vector",
    "map_genotype",
    "num_not_wt", # kept for consistency with old tests if needed
    "not_wt_mask" 
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 4 total genotypes (0 is WT, 1-3 are mutants)
    - 8 observations (batch size)
    """
    num_genotype = 4
    batch_size = 8
    
    # Batch indices mapping observations to genotypes
    # [WT, Mut1, Mut2, Mut3, Mut1, WT, Mut2, Mut3]
    batch_idx = jnp.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=jnp.int32)
    
    # WT is genotype 0
    wt_indexes = jnp.array([0], dtype=jnp.int32)
    
    # Scale vector for the scale handler
    scale_vector = jnp.ones(batch_size, dtype=float)
    
    # Legacy fields (if needed by other logic, though not strictly by define_model now)
    num_not_wt = 3
    not_wt_mask = jnp.array([False, True, True, True])
    map_genotype = batch_idx # In a batch context, map matches batch_idx
    
    return MockGrowthData(
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        wt_indexes=wt_indexes,
        scale_vector=scale_vector,
        map_genotype=map_genotype,
        num_not_wt=num_not_wt,
        not_wt_mask=not_wt_mask
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "hyper_loc_loc" in params
    assert params["hyper_loc_loc"] == 0.0

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.hyper_loc_loc == 0.0

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_activity"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)
    assert f"{name}_hyper_loc" in guesses
    
    # Check offset guess
    # The code initializes zeros for ALL genotypes (num_genotype)
    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_genotype,)
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert jnp.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for hierarchical activity.
    """
    name = "test_activity"
    priors = get_priors()
    
    # Get base guesses (genotype-level)
    base_guesses = get_guesses(name, mock_data)
    
    # Construct batch-level guesses for substitute
    # define_model samples 'offset' with shape (batch_size,), not (num_genotype,)
    # We must map the genotype guesses to the batch
    batch_guesses = base_guesses.copy()
    
    genotype_offsets = base_guesses[f"{name}_offset"]
    batch_offsets = genotype_offsets[mock_data.batch_idx]
    batch_guesses[f"{name}_offset"] = batch_offsets

    # Substitute
    substituted_model = substitute(define_model, data=batch_guesses)
    
    # --- 1. Execute Model ---
    final_activity = substituted_model(name=name, 
                                       data=mock_data, 
                                       priors=priors)

    # --- 2. Trace execution ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check the Deterministic Site ---
    assert name in model_trace
    activity_site = model_trace[name]["value"]
    
    # Check shape: Should match batch_size
    assert activity_site.shape == (mock_data.batch_size,)
    
    # --- 4. Check WT Logic ---
    # WT indices in batch_idx are 0 and 5
    wt_indices = jnp.where(jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    assert jnp.all(activity_site[wt_indices] == 1.0)
    
    # --- 5. Check Mutant Logic ---
    # Mutants are guesses as 0.0 offset -> log(activity) = 0 -> activity = 1.0
    mutant_indices = jnp.where(~jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    assert jnp.allclose(activity_site[mutant_indices], 1.0)
    
    # --- 6. Check Final Expanded Shape ---
    # Expect: (1, 1, 1, 1, 1, 1, batch_size)
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_activity.shape == expected_shape

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes and execution.
    """
    name = "test_activity_guide"
    priors = get_priors()

    # Seed the guide execution because it samples
    with seed(rng_seed=0):
        final_activity = guide(name=name,
                               data=mock_data,
                               priors=priors)

    # Expect: (1, 1, 1, 1, 1, 1, batch_size)
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_activity.shape == expected_shape

    # Basic sanity check on values (should be positive)
    assert jnp.all(final_activity >= 0.0)


# ---------------------------------------------------------------------------
# Tests: pinning – sanity / ModelPriors
# ---------------------------------------------------------------------------

def test_pinnable_suffixes():
    """The module exposes the expected pinnable suffix tuple."""
    assert set(_PINNABLE_SUFFIXES) == {"hyper_loc", "hyper_scale"}


def test_model_priors_default_pinned_is_empty_dict():
    """ModelPriors() with no `pinned` argument exposes an empty dict."""
    priors = ModelPriors(**get_hyperparameters())
    assert hasattr(priors, "pinned")
    assert priors.pinned == {}


def test_model_priors_accepts_pinned_dict():
    """ModelPriors accepts a `pinned` dict and exposes it on the instance."""
    pinned = {"hyper_loc": 0.0, "hyper_scale": 0.5}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


# ---------------------------------------------------------------------------
# Tests: define_model under pinning
# ---------------------------------------------------------------------------

def test_define_model_unpinned_uses_sample_sites(mock_data):
    """Without any pinning, every pinnable suffix is a sample site."""
    name = "act"
    priors = get_priors()  # pinned = {}

    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )

    for suffix in _PINNABLE_SUFFIXES:
        site_name = f"{name}_{suffix}"
        assert site_name in tr
        assert tr[site_name]["type"] == "sample"


def test_define_model_pinned_replaces_with_deterministic(mock_data):
    """Pinned suffixes become deterministic sites with the pinned value."""
    name = "act"
    pinned = {"hyper_loc": 0.123}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    base_guesses = get_guesses(name, mock_data)
    # Drop pinned keys so substitute doesn't override the deterministic value.
    sub_data = {
        k: v for k, v in base_guesses.items()
        if not any(k == f"{name}_{s}" for s in pinned)
    }
    # Map per-genotype offset to per-batch shape
    sub_data[f"{name}_offset"] = base_guesses[f"{name}_offset"][mock_data.batch_idx]

    substituted = substitute(define_model, data=sub_data)
    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(
            name=name, data=mock_data, priors=priors
        )

    site_name = f"{name}_hyper_loc"
    assert tr[site_name]["type"] == "deterministic"
    assert jnp.allclose(tr[site_name]["value"], 0.123)

    # Unpinned suffix still a sample
    assert tr[f"{name}_hyper_scale"]["type"] == "sample"


def test_define_model_all_pinned_has_only_offset_sample_site(mock_data):
    """When all hypers are pinned, only the per-genotype offset remains as a sample."""
    name = "act"
    pinned = {"hyper_loc": 0.0, "hyper_scale": 0.1}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    base_guesses = get_guesses(name, mock_data)
    sub_data = {
        f"{name}_offset": base_guesses[f"{name}_offset"][mock_data.batch_idx]
    }
    substituted = substitute(define_model, data=sub_data)
    with seed(rng_seed=0):
        tr = trace(substituted).get_trace(
            name=name, data=mock_data, priors=priors
        )

    sample_sites = {n for n, s in tr.items() if s["type"] == "sample"}
    assert sample_sites == {f"{name}_offset"}


# ---------------------------------------------------------------------------
# Tests: guide under pinning
# ---------------------------------------------------------------------------

def test_guide_pinned_drops_variational_params(mock_data):
    """
    A pinned hyper must not register any variational params or sample sites
    in the guide.  Unpinned hypers retain their full param + sample machinery.
    """
    name = "act"
    pinned = {"hyper_loc": 0.0}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    # Pinned hyper: no sample, no params
    assert f"{name}_hyper_loc" not in tr
    assert f"{name}_hyper_loc_loc" not in tr
    assert f"{name}_hyper_loc_scale" not in tr

    # Unpinned hyper still present
    assert f"{name}_hyper_scale" in tr
    assert f"{name}_hyper_scale_loc" in tr
    assert f"{name}_hyper_scale_scale" in tr

    # Per-genotype offset machinery untouched
    assert f"{name}_offset" in tr
    assert f"{name}_offset_locs" in tr
    assert f"{name}_offset_scales" in tr


def test_guide_all_pinned_keeps_only_offset_machinery(mock_data):
    """When all hypers are pinned, only offset params/samples remain in the guide."""
    name = "act"
    pinned = {"hyper_loc": 0.0, "hyper_scale": 0.1}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    for suffix in _PINNABLE_SUFFIXES:
        assert f"{name}_{suffix}" not in tr

    # Variational params for the hypers should be gone
    for guide_param in (
        "hyper_loc_loc", "hyper_loc_scale",
        "hyper_scale_loc", "hyper_scale_scale",
    ):
        assert f"{name}_{guide_param}" not in tr

    assert f"{name}_offset" in tr
    assert f"{name}_offset_locs" in tr
    assert f"{name}_offset_scales" in tr


# ---------------------------------------------------------------------------
# Model+guide compatibility under pinning (SVI sanity)
# ---------------------------------------------------------------------------

def test_model_and_guide_pinned_have_compatible_sample_sites(mock_data):
    """
    Numpyro requires the guide to have a `sample` site for every model
    `sample` site (excluding `obs`) — and no extras.  After pinning, both
    sides drop the same sites; this test confirms the symmetry.
    """
    name = "compat"
    pinned = {"hyper_loc": 0.0}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=priors
        )
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