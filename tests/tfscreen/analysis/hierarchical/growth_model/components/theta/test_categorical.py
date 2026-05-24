import pytest
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.theta.categorical import (
    ModelPriors,
    ThetaParam,
    define_model,
    guide,
    run_model,
    get_hyperparameters,
    get_guesses,
    get_priors,
    build_calc_df,
    compute_theta_samples,
)

# --- Mock Data Fixture ---

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "batch_size",
    "batch_idx",
    "scale_vector",
    "map_theta",
    "scatter_theta",
    "geno_theta_idx",
    "titrant_conc"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    """
    num_titrant_name = 2
    num_titrant_conc = 3
    num_genotype = 4
    
    batch_size = 2
    batch_idx = jnp.array([1, 3], dtype=jnp.int32)
    scale_vector = jnp.ones(batch_size, dtype=float)
    map_theta = jnp.array([0, 5, 10, 23, 1], dtype=jnp.int32)
    
    # New required fields
    geno_theta_idx = jnp.array([1, 3], dtype=jnp.int32)
    titrant_conc = jnp.array([0.0, 1.0, 10.0])
    
    return MockData(
        num_titrant_name=num_titrant_name,
        num_titrant_conc=num_titrant_conc,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        scale_vector=scale_vector,
        map_theta=map_theta,
        scatter_theta=1,
        geno_theta_idx=geno_theta_idx,
        titrant_conc=titrant_conc
    )

@pytest.fixture
def model_setup(mock_data):
    """
    Provides a deterministic ThetaParam object (BATCHED) for testing run_model.
    """
    name = "test_theta_cat"
    priors = get_priors()
    base_guesses = get_guesses(name, mock_data)
    
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_logit_theta_offset"]
    batch_guesses[f"{name}_logit_theta_offset"] = full_offsets[..., mock_data.batch_idx]

    substituted_model = substitute(define_model, data=batch_guesses)
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)
    
    # Ensure mu/sigma are present (since define_model returns them)
    assert theta_param.mu is not None
    assert theta_param.sigma is not None
    assert theta_param.concentrations is not None
    
    return theta_param

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "logit_theta_hyper_loc_loc" in params

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_theta_cat"
    guesses = get_guesses(name, mock_data)
    assert isinstance(guesses, dict)
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype
    )
    assert guesses[f"{name}_logit_theta_offset"].shape == expected_shape

def test_define_model_shapes_and_values(mock_data):
    """
    Tests the core logic of define_model.
    """
    name = "test_theta_cat"
    priors = get_priors()
    
    base_guesses = get_guesses(name, mock_data)
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_logit_theta_offset"]
    batch_guesses[f"{name}_logit_theta_offset"] = full_offsets[..., mock_data.batch_idx]
    
    substituted_model = substitute(define_model, data=batch_guesses)
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)

    expected_batch_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_param.theta.shape == expected_batch_shape
    assert theta_param.mu.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
    assert jnp.allclose(theta_param.theta, 0.5)
    assert jnp.allclose(theta_param.concentrations, mock_data.titrant_conc)

def test_run_model_no_scatter(model_setup, mock_data):
    """
    Tests 'run_model' with scatter_theta=0.
    """
    theta_param = model_setup
    data = mock_data._replace(scatter_theta=0)
    theta_calc = run_model(theta_param, data)
    
    # We no longer assert 'is' because run_model slices genotypes
    # it should be equal if indices match full theta_param.theta
    # In model_setup, theta_param.theta already has batch_size genotypes.
    # mock_data.geno_theta_idx is [1, 3] which matches mock_data.batch_idx.
    
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_calc.shape == expected_shape
    assert jnp.allclose(theta_calc, theta_param.theta)

def test_run_model_with_scatter(model_setup, mock_data):
    """
    Tests 'run_model' with scatter_theta=1.
    """
    theta_param = model_setup
    data = mock_data 
    theta_calc = run_model(theta_param, data)
    expected_shape = (1, 1, 1, 1, 
                      mock_data.num_titrant_name, 
                      mock_data.num_titrant_conc, 
                      mock_data.batch_size)
    assert theta_calc.shape == expected_shape

def test_run_model_concentration_mapping(model_setup, mock_data):
    """
    Tests 'run_model' concentration mapping logic.
    """
    theta_param = model_setup
    # Data has different concentrations
    new_conc = jnp.array([1.0, 0.0]) # Swapped and subset
    data = mock_data._replace(titrant_conc=new_conc, scatter_theta=0)
    
    theta_calc = run_model(theta_param, data)
    
    # Shape should follow new_conc
    assert theta_calc.shape == (mock_data.num_titrant_name, 2, mock_data.batch_size)
    
    # Check values mapping
    # Orig concentrations: [0.0, 1.0, 10.0] -> indices [0, 1, 2]
    # new_conc [1.0, 0.0] should map to indices [1, 0]
    expected = theta_param.theta[:, [1, 0], :]
    assert jnp.allclose(theta_calc, expected)

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_theta_cat_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        theta_param = guide(name=name,
                            data=mock_data,
                            priors=priors)

    expected_sample_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_param.theta.shape == expected_sample_shape
    assert theta_param.mu.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)

def test_population_moments_logic(mock_data):
    """
    Verify that get_population_moments returns the expected tensors.
    """
    name = "test_moments"
    priors = get_priors()

    with seed(rng_seed=0):
        theta_param = define_model(name, mock_data, priors)

    from tfscreen.analysis.hierarchical.growth_model.components.theta.categorical import get_population_moments
    mu, sigma = get_population_moments(theta_param, mock_data)

    assert mu.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
    assert sigma.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
    # Check sigma is positive
    assert jnp.all(sigma > 0)


# ---------------------------------------------------------------------------
# Fixtures for build_calc_df / compute_theta_samples
# ---------------------------------------------------------------------------

@pytest.fixture
def training_df():
    """
    Minimal growth_tm.df with all columns needed by build_calc_df.
    Layout: 2 titrant_names x 2 titrant_concs x 3 genotypes = 12 unique rows.
    """
    rows = []
    for ni, tname in enumerate(["IPTG", "aTc"]):
        for ci, tconc in enumerate([0.0, 1.0]):
            for gi, geno in enumerate(["wt", "M1A", "K2R"]):
                rows.append({
                    "genotype": geno,
                    "titrant_name": tname,
                    "titrant_conc": tconc,
                    "titrant_name_idx": ni,
                    "titrant_conc_idx": ci,
                    "genotype_idx": gi,
                })
    return pd.DataFrame(rows)


class MockGrowthTM:
    def __init__(self, df):
        self.df = df


class MockModel:
    def __init__(self, df):
        self.growth_tm = MockGrowthTM(df)


@pytest.fixture
def mock_model(training_df):
    return MockModel(training_df)


@pytest.fixture
def posterior_theta(training_df):
    """
    Fake posteriors matching the three parameters used by compute_theta_samples.

    Layout: S=5, N_name=2, N_conc=2, N_geno=3.

    hyper_loc  = 0  (all zeros)
    hyper_scale = 1 (all ones)
    offset[s, ni, ci, gi] = s * 100 + ni * 10 + ci * 3 + gi

    => logit_theta = 0 + offset * 1 = offset
    => theta = sigmoid(offset)
    """
    S, N_name, N_conc, N_geno = 5, 2, 2, 3
    offset = np.zeros((S, N_name, N_conc, N_geno))
    for s in range(S):
        for n in range(N_name):
            for c in range(N_conc):
                for g in range(N_geno):
                    offset[s, n, c, g] = s * 100 + n * 10 + c * 3 + g
    return {
        "theta_logit_theta_hyper_loc":   np.zeros((S, N_name, N_conc)),
        "theta_logit_theta_hyper_scale": np.ones((S, N_name, N_conc)),
        "theta_logit_theta_offset":      offset,
    }


# ---------------------------------------------------------------------------
# Tests for build_calc_df
# ---------------------------------------------------------------------------

class TestBuildCalcDf:

    def test_default_returns_all_training_rows(self, mock_model, training_df):
        calc_df, internal_cols, extra = build_calc_df(mock_model, None)
        assert set(["genotype", "titrant_name", "titrant_conc"]).issubset(calc_df.columns)
        assert len(calc_df) == len(training_df)
        assert extra == {}

    def test_internal_cols_stripped_key(self, mock_model):
        _, internal_cols, _ = build_calc_df(mock_model, None)
        assert set(internal_cols) == {"titrant_name_idx", "titrant_conc_idx", "genotype_idx"}

    def test_manual_titrant_df_filters_to_known_concs(self, mock_model):
        manual = pd.DataFrame({"titrant_name": ["IPTG", "aTc"], "titrant_conc": [0.0, 1.0]})
        calc_df, _, _ = build_calc_df(mock_model, manual)
        # 2 titrant pairs × 3 genotypes = 6 rows
        assert len(calc_df) == 6

    def test_manual_titrant_df_with_genotype_column(self, mock_model):
        manual = pd.DataFrame({
            "genotype": ["wt", "M1A"],
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 0.0],
        })
        calc_df, _, _ = build_calc_df(mock_model, manual)
        assert set(calc_df["genotype"]) == {"wt", "M1A"}
        assert len(calc_df) == 2

    def test_manual_titrant_df_unknown_conc_raises(self, mock_model):
        manual = pd.DataFrame({"titrant_name": ["IPTG"], "titrant_conc": [999.0]})
        with pytest.raises(ValueError, match="not seen during training"):
            build_calc_df(mock_model, manual)

    def test_manual_titrant_df_missing_columns_raises(self, mock_model):
        manual = pd.DataFrame({"titrant_name": ["IPTG"]})
        with pytest.raises(ValueError, match="missing columns"):
            build_calc_df(mock_model, manual)


# ---------------------------------------------------------------------------
# Tests for compute_theta_samples
# ---------------------------------------------------------------------------

class TestComputeThetaSamples:

    def test_output_shape(self, mock_model, posterior_theta):
        calc_df, _, _ = build_calc_df(mock_model, None)
        samples = compute_theta_samples(calc_df, posterior_theta)
        assert samples.shape == (5, len(calc_df))

    def test_correct_values_indexed(self, mock_model, posterior_theta):
        calc_df, _, _ = build_calc_df(mock_model, None)
        samples = compute_theta_samples(calc_df, posterior_theta)
        # Spot-check: for sample s=2 and the first row.
        # offset[s, ni, ci, gi] = s*100 + ni*10 + ci*3 + gi; hyper_loc=0, hyper_scale=1
        # => theta = sigmoid(offset)
        row = calc_df.iloc[0]
        logit = (2 * 100
                 + int(row["titrant_name_idx"]) * 10
                 + int(row["titrant_conc_idx"]) * 3
                 + int(row["genotype_idx"]))
        expected = 1.0 / (1.0 + np.exp(-logit))
        assert samples[2, 0] == pytest.approx(expected)

    def test_all_rows_finite(self, mock_model, posterior_theta):
        calc_df, _, _ = build_calc_df(mock_model, None)
        samples = compute_theta_samples(calc_df, posterior_theta)
        assert np.all(np.isfinite(samples))