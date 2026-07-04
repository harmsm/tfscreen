import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.tfmodel.generative.components.dk_geno.pinned import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    get_extract_specs,
    read_dk_geno_pins,
    build_dk_geno_values,
)

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "batch_size",
    "batch_idx",
    "wt_indexes",
])


@pytest.fixture
def mock_data():
    """
    4 genotypes (0 = wt, 1-3 = mutants). batch_idx repeats some entries to
    exercise the batching/indexing path.
    """
    batch_idx = jnp.array([0, 1, 2, 3, 1, 0], dtype=jnp.int32)
    return MockGrowthData(
        batch_size=6,
        batch_idx=batch_idx,
        wt_indexes=jnp.array([0], dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "dk_geno_values" in params
    assert np.asarray(params["dk_geno_values"]).shape == (1,)
    assert np.all(np.asarray(params["dk_geno_values"]) == 0.0)


def test_get_guesses(mock_data):
    guesses = get_guesses("test_dk", mock_data)
    assert isinstance(guesses, dict)
    assert len(guesses) == 0


def test_get_priors_default_is_placeholder():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert np.asarray(priors.dk_geno_values).shape == (1,)


def test_get_priors_with_values():
    values = jnp.array([0.0, -0.02, 0.01, 0.0])
    priors = get_priors(dk_geno_values=values)
    assert isinstance(priors, ModelPriors)
    assert jnp.allclose(priors.dk_geno_values, values)


def test_get_extract_specs():
    MockCtx = namedtuple("MockCtx", ["growth_tm"])
    MockTm = namedtuple("MockTm", ["df"])
    ctx = MockCtx(growth_tm=MockTm(df="the_df"))

    specs = get_extract_specs(ctx)
    assert isinstance(specs, list)
    assert len(specs) == 1
    assert specs[0]["input_df"] == "the_df"
    assert specs[0]["params_to_get"] == ["dk_geno"]
    assert specs[0]["map_column"] == "map_genotype"


# ---------------------------------------------------------------------------
# define_model / guide — core logic and shapes
# ---------------------------------------------------------------------------

def test_define_model_logic_and_shapes(mock_data):
    """
    Per-genotype pinned values are indexed via batch_idx; wt is forced to 0
    even though its own pin (index 0) is already 0 here.
    """
    name = "test_dk_pinned"
    dk_values = jnp.array([0.0, -0.02, 0.01, 0.005])
    priors = get_priors(dk_geno_values=dk_values)

    final_dk_geno = define_model(name=name, data=mock_data, priors=priors)

    model_trace = trace(define_model).get_trace(
        name=name, data=mock_data, priors=priors
    )

    assert name in model_trace
    dk_geno_site = model_trace[name]["value"]
    assert dk_geno_site.shape == (mock_data.batch_size,)

    expected = dk_values[mock_data.batch_idx]
    assert jnp.allclose(dk_geno_site, expected)

    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_dk_geno.shape == expected_shape
    assert jnp.allclose(final_dk_geno[0, 0, 0, 0, 0, 0, :], expected)


def test_define_model_forces_wt_to_zero_even_if_pinned_nonzero(mock_data):
    """
    Defense in depth: even if dk_geno_values has a nonzero entry at a wt
    index, define_model must still force wt entries to exactly 0.
    """
    name = "test_dk_pinned_wt"
    # wt (index 0) mistakenly pinned nonzero.
    dk_values = jnp.array([0.5, -0.02, 0.01, 0.005])
    priors = get_priors(dk_geno_values=dk_values)

    model_trace = trace(define_model).get_trace(
        name=name, data=mock_data, priors=priors
    )
    dk_geno_site = model_trace[name]["value"]

    wt_in_batch = jnp.where(jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    assert jnp.all(dk_geno_site[wt_in_batch] == 0.0)


def test_guide_logic_and_shapes(mock_data):
    """guide mirrors define_model but registers no sites."""
    name = "test_dk_pinned_guide"
    dk_values = jnp.array([0.0, -0.02, 0.01, 0.005])
    priors = get_priors(dk_geno_values=dk_values)

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)
        final_dk_geno = guide(name=name, data=mock_data, priors=priors)

    # No sites registered at all (pure function, no pyro.sample/deterministic).
    assert len(tr) == 0

    expected = dk_values[mock_data.batch_idx]
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_dk_geno.shape == expected_shape
    assert jnp.allclose(final_dk_geno[0, 0, 0, 0, 0, 0, :], expected)


def test_model_and_guide_have_no_sample_sites(mock_data):
    """Neither define_model nor guide register any pyro.sample sites."""
    name = "compat"
    dk_values = jnp.array([0.0, -0.02, 0.01, 0.005])
    priors = get_priors(dk_geno_values=dk_values)

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
    guide_samples = {n for n, s in guide_trace.items() if s["type"] == "sample"}

    assert model_samples == set()
    assert guide_samples == set()


# ---------------------------------------------------------------------------
# read_dk_geno_pins — CSV I/O
# ---------------------------------------------------------------------------

def test_read_dk_geno_pins_valid(tmp_path):
    csv_path = tmp_path / "pins.csv"
    csv_path.write_text("genotype,dk_geno\nm1,-0.02\nm2,0.01\n")

    pins = read_dk_geno_pins(str(csv_path))
    assert pins == {"m1": -0.02, "m2": 0.01}


def test_read_dk_geno_pins_missing_column(tmp_path):
    csv_path = tmp_path / "pins.csv"
    csv_path.write_text("genotype\nm1\n")

    with pytest.raises(ValueError, match="missing required column"):
        read_dk_geno_pins(str(csv_path))


def test_read_dk_geno_pins_unrecognised_column(tmp_path):
    csv_path = tmp_path / "pins.csv"
    csv_path.write_text("genotype,dk_geno,extra\nm1,-0.02,foo\n")

    with pytest.raises(ValueError, match="unrecognised column"):
        read_dk_geno_pins(str(csv_path))


def test_read_dk_geno_pins_duplicate_genotype(tmp_path):
    csv_path = tmp_path / "pins.csv"
    csv_path.write_text("genotype,dk_geno\nm1,-0.02\nm1,0.03\n")

    with pytest.raises(ValueError, match="more than once"):
        read_dk_geno_pins(str(csv_path))


# ---------------------------------------------------------------------------
# build_dk_geno_values — array assembly and validation
# ---------------------------------------------------------------------------

def test_build_dk_geno_values_defaults_unlisted_to_zero():
    pins = {"m1": -0.02}
    genotype_labels = ["wt", "m1", "m2"]

    values = build_dk_geno_values(pins, genotype_labels)
    assert values.shape == (3,)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(-0.02)
    assert values[2] == pytest.approx(0.0)


def test_build_dk_geno_values_unknown_genotype_raises():
    pins = {"m1": -0.02, "not_in_data": 0.01}
    genotype_labels = ["wt", "m1", "m2"]

    with pytest.raises(ValueError, match="not_in_data"):
        build_dk_geno_values(pins, genotype_labels)


def test_build_dk_geno_values_nonzero_wt_raises():
    pins = {"wt": 0.01, "m1": -0.02}
    genotype_labels = ["wt", "m1", "m2"]

    with pytest.raises(ValueError, match="wildtype dk_geno must be exactly 0.0"):
        build_dk_geno_values(pins, genotype_labels)


def test_build_dk_geno_values_zero_wt_allowed():
    pins = {"wt": 0.0, "m1": -0.02}
    genotype_labels = ["wt", "m1", "m2"]

    values = build_dk_geno_values(pins, genotype_labels)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(-0.02)


def test_build_dk_geno_values_custom_wt_label():
    pins = {"WT": 0.0, "m1": -0.02}
    genotype_labels = ["WT", "m1"]

    values = build_dk_geno_values(pins, genotype_labels, wt_label="WT")
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(-0.02)
