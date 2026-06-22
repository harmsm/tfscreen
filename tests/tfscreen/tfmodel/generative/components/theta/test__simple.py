import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, seed
from collections import namedtuple

from tfscreen.tfmodel.generative.components.theta._simple import (
    ModelPriors,
    ThetaParam,
    define_model,
    guide,
    run_model,
    get_hyperparameters,
    get_guesses,
    get_priors,
    get_population_moments,
    _logit,
    _THETA_EPS,
)


# ---------------------------------------------------------------------------
# Mock data fixture
# ---------------------------------------------------------------------------

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "scatter_theta",
    "geno_theta_idx",
    "titrant_conc",
])


@pytest.fixture
def mock_data():
    """
    Two titrants, three concentrations, four genotypes.

    The simple theta component does not use batch_idx/scale_vector, so
    those fields are intentionally omitted.
    """
    return MockData(
        num_titrant_name=2,
        num_titrant_conc=3,
        num_genotype=4,
        scatter_theta=1,
        geno_theta_idx=jnp.array([1, 3], dtype=jnp.int32),
        titrant_conc=jnp.array([0.0, 1.0, 10.0]),
    )


@pytest.fixture
def mock_priors():
    """
    A non-trivial theta_values tensor of shape (num_titrant_name=2,
    num_titrant_conc=3) with two saturating Hill-like rows.  2-D path.
    """
    theta_values = jnp.array([
        [0.10, 0.50, 0.90],
        [0.05, 0.20, 0.80],
    ])
    return ModelPriors(theta_values=theta_values, sigma_floor=0.05)


@pytest.fixture
def mock_priors_3d(mock_data):
    """
    Per-genotype theta_values of shape (T=2, C=3, G=4).

    Each genotype has a distinct sigmoid-like curve so we can verify that
    values are NOT broadcast across the genotype axis.
    Axis order: (titrant_name, titrant_conc, genotype).
    """
    # shape = (T=2, C=3, G=4): outer=titrant, middle=conc, inner=genotype.
    theta_values = jnp.array([
        # titrant 0 — shape (C=3, G=4)
        [[0.90, 0.80, 0.70, 0.60],   # conc 0, genotypes 0-3
         [0.50, 0.40, 0.30, 0.20],   # conc 1
         [0.10, 0.05, 0.08, 0.03]],  # conc 2
        # titrant 1
        [[0.85, 0.75, 0.65, 0.55],
         [0.45, 0.35, 0.25, 0.15],
         [0.12, 0.07, 0.09, 0.04]],
    ])  # shape (2, 3, 4) = (T, C, G)
    return ModelPriors(theta_values=theta_values, sigma_floor=0.05)


# ---------------------------------------------------------------------------
# Helper / sanity tests
# ---------------------------------------------------------------------------

def test_logit_handles_extremes():
    """_logit must be finite at 0 and 1 thanks to eps clipping."""
    out = _logit(jnp.array([0.0, 0.5, 1.0]))
    assert jnp.all(jnp.isfinite(out))
    # logit(0.5) == 0 exactly
    assert jnp.isclose(out[1], 0.0)
    # Sign and rough magnitude (float32 makes 1.0-eps and eps round
    # asymmetrically, so we don't insist on exact antisymmetry)
    assert out[0] < -10.0
    assert out[2] > 10.0


def test_get_hyperparameters_shape_and_keys():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "theta_values" in params
    assert "sigma_floor" in params
    # Placeholder is a 2D array (so .ndim == 2 holds at construction)
    assert params["theta_values"].ndim == 2
    assert params["sigma_floor"] > 0


def test_get_priors_returns_model_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert hasattr(priors, "theta_values")
    assert hasattr(priors, "sigma_floor")


def test_get_guesses_is_empty(mock_data):
    """Simple theta has no latent params → no guesses."""
    guesses = get_guesses("any_name", mock_data)
    assert guesses == {}


def test_model_priors_construction(mock_priors):
    """ModelPriors accepts theta_values + sigma_floor and exposes them."""
    assert mock_priors.theta_values.shape == (2, 3)
    assert mock_priors.sigma_floor == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# define_model: shapes, values, no sample sites
# ---------------------------------------------------------------------------

def test_define_model_shapes_and_values(mock_data, mock_priors):
    """define_model must broadcast theta to (Name, Conc, Geno)."""
    name = "simple_theta"
    theta_param = define_model(name=name, data=mock_data, priors=mock_priors)

    assert isinstance(theta_param, ThetaParam)

    # theta tensor: (Name, Conc, Geno) and constant along genotype
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype,
    )
    assert theta_param.theta.shape == expected_shape
    for g in range(mock_data.num_genotype):
        assert jnp.allclose(theta_param.theta[..., g], mock_priors.theta_values)

    # Population moments: (Name, Conc, 1)
    expected_moment_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        1,
    )
    assert theta_param.mu.shape == expected_moment_shape
    assert theta_param.sigma.shape == expected_moment_shape

    # mu == logit(theta_values)
    expected_mu = _logit(mock_priors.theta_values)[..., None]
    assert jnp.allclose(theta_param.mu, expected_mu)

    # sigma is the constant floor
    assert jnp.allclose(theta_param.sigma, mock_priors.sigma_floor)

    # concentrations passed through
    assert jnp.allclose(theta_param.concentrations, mock_data.titrant_conc)


def test_define_model_registers_no_sample_sites(mock_data, mock_priors):
    """
    The simple theta component must register zero sample sites.  Only a
    deterministic site for the broadcast theta tensor is allowed.
    """
    name = "simple"
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(
            name=name, data=mock_data, priors=mock_priors,
        )

    sample_sites = [n for n, s in tr.items() if s["type"] == "sample"]
    assert sample_sites == [], f"unexpected sample sites in model: {sample_sites}"

    # The deterministic theta site is registered
    assert f"{name}_theta" in tr
    assert tr[f"{name}_theta"]["type"] == "deterministic"


# ---------------------------------------------------------------------------
# guide: same outputs as define_model, no params, no sites
# ---------------------------------------------------------------------------

def test_guide_returns_same_values_as_model(mock_data, mock_priors):
    name = "simple"

    model_param = define_model(name=name, data=mock_data, priors=mock_priors)
    guide_param = guide(name=name, data=mock_data, priors=mock_priors)

    assert jnp.allclose(model_param.theta, guide_param.theta)
    assert jnp.allclose(model_param.mu, guide_param.mu)
    assert jnp.allclose(model_param.sigma, guide_param.sigma)
    assert jnp.allclose(model_param.concentrations, guide_param.concentrations)


def test_guide_registers_no_sites_at_all(mock_data, mock_priors):
    """
    Guide must be a true no-op: no sample sites, no param sites, and no
    deterministic sites either (those belong on the model side).
    """
    name = "simple"
    with seed(rng_seed=0):
        tr = trace(guide).get_trace(
            name=name, data=mock_data, priors=mock_priors,
        )

    sample_sites = [n for n, s in tr.items() if s["type"] == "sample"]
    param_sites = [n for n, s in tr.items() if s["type"] == "param"]
    det_sites = [n for n, s in tr.items() if s["type"] == "deterministic"]

    assert sample_sites == [], f"unexpected sample sites in guide: {sample_sites}"
    assert param_sites == [], f"unexpected param sites in guide: {param_sites}"
    assert det_sites == [], f"unexpected deterministic sites in guide: {det_sites}"


def test_model_and_guide_have_compatible_sample_sites(mock_data, mock_priors):
    """
    SVI requires the model and guide to register the same set of sample
    sites (excluding observed sites).  Both should be empty here.
    """
    name = "simple"
    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name=name, data=mock_data, priors=mock_priors,
        )
        guide_trace = trace(guide).get_trace(
            name=name, data=mock_data, priors=mock_priors,
        )

    model_samples = {
        n for n, s in model_trace.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_samples = {
        n for n, s in guide_trace.items() if s["type"] == "sample"
    }

    assert model_samples == set()
    assert guide_samples == set()
    assert model_samples == guide_samples


# ---------------------------------------------------------------------------
# run_model: genotype slicing, concentration mapping, scatter
# ---------------------------------------------------------------------------

def test_run_model_no_scatter(mock_data, mock_priors):
    """run_model with scatter_theta=0 returns (Name, Conc, GenoSubset)."""
    name = "simple"
    theta_param = define_model(name=name, data=mock_data, priors=mock_priors)

    data = mock_data._replace(scatter_theta=0)
    out = run_model(theta_param, data)

    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.geno_theta_idx.shape[0],
    )
    assert out.shape == expected_shape

    # Theta is constant along genotype, so the subset equals the original
    # theta_values broadcast to the subset width.
    for g in range(out.shape[-1]):
        assert jnp.allclose(out[..., g], mock_priors.theta_values)


def test_run_model_with_scatter(mock_data, mock_priors):
    """run_model with scatter_theta=1 inserts four leading singleton dims."""
    name = "simple"
    theta_param = define_model(name=name, data=mock_data, priors=mock_priors)

    out = run_model(theta_param, mock_data)
    expected_shape = (
        1, 1, 1, 1,
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.geno_theta_idx.shape[0],
    )
    assert out.shape == expected_shape


def test_run_model_concentration_mapping(mock_data, mock_priors):
    """
    Re-ordering / sub-setting data.titrant_conc must re-order columns
    of the output via searchsorted into theta_param.concentrations.
    """
    name = "simple"
    theta_param = define_model(name=name, data=mock_data, priors=mock_priors)

    new_conc = jnp.array([1.0, 0.0])  # swapped + subset of [0.0, 1.0, 10.0]
    data = mock_data._replace(titrant_conc=new_conc, scatter_theta=0)

    out = run_model(theta_param, data)
    assert out.shape == (
        mock_data.num_titrant_name,
        new_conc.shape[0],
        mock_data.geno_theta_idx.shape[0],
    )

    # Expected: pull columns [1, 0] from theta_param.theta after slicing genotypes
    expected = theta_param.theta[..., mock_data.geno_theta_idx][:, [1, 0], :]
    assert jnp.allclose(out, expected)


# ---------------------------------------------------------------------------
# get_population_moments
# ---------------------------------------------------------------------------

def test_get_population_moments_returns_mu_sigma(mock_data, mock_priors):
    name = "simple"
    theta_param = define_model(name=name, data=mock_data, priors=mock_priors)
    mu, sigma = get_population_moments(theta_param, mock_data)

    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        1,
    )
    assert mu.shape == expected_shape
    assert sigma.shape == expected_shape
    assert jnp.all(sigma > 0)
    assert jnp.allclose(sigma, mock_priors.sigma_floor)


def test_extreme_theta_values_finite_mu(mock_data):
    """
    theta_values containing 0.0 and 1.0 must produce finite mu thanks to
    the eps-clipping inside _logit.
    """
    theta_values = jnp.array([
        [0.0, 0.5, 1.0],
        [1.0, 0.5, 0.0],
    ])
    priors = ModelPriors(theta_values=theta_values, sigma_floor=0.01)
    data = MockData(
        num_titrant_name=2,
        num_titrant_conc=3,
        num_genotype=2,
        scatter_theta=0,
        geno_theta_idx=jnp.array([0, 1], dtype=jnp.int32),
        titrant_conc=jnp.array([0.0, 1.0, 10.0]),
    )

    theta_param = define_model(name="x", data=data, priors=priors)
    assert jnp.all(jnp.isfinite(theta_param.mu))

    # logit(0.5) == 0; logit(0.0) is large negative; logit(1.0) is large positive.
    # Float32 makes the +/- bounds asymmetric, so we assert sign + rough
    # magnitude rather than exact equality with log((1-eps)/eps).
    assert jnp.isclose(theta_param.mu[0, 1, 0], 0.0)
    assert theta_param.mu[0, 0, 0] < -10.0
    assert theta_param.mu[0, 2, 0] > 10.0


# ---------------------------------------------------------------------------
# 3-D theta_values: per-genotype path
# ---------------------------------------------------------------------------

class TestPerGenotype:
    """Tests for the (T, C, G) theta_values path introduced for calibration."""

    def test_define_model_returns_per_genotype_theta(self, mock_data, mock_priors_3d):
        """Each genotype should get its own theta column, not a broadcast copy."""
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        assert theta_param.theta.shape == (
            mock_data.num_titrant_name,
            mock_data.num_titrant_conc,
            mock_data.num_genotype,
        )
        # The per-genotype theta must match the input exactly.
        assert jnp.allclose(theta_param.theta, mock_priors_3d.theta_values)
        # Genotypes must differ from each other — no broadcast collapse.
        assert not jnp.allclose(theta_param.theta[..., 0], theta_param.theta[..., 1])

    def test_define_model_3d_population_moment_shapes(self, mock_data, mock_priors_3d):
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        expected = (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
        assert theta_param.mu.shape == expected
        assert theta_param.sigma.shape == expected

    def test_define_model_3d_mu_is_mean_over_genotypes(self, mock_data, mock_priors_3d):
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        expected_mu = jnp.mean(
            _logit(mock_priors_3d.theta_values), axis=-1, keepdims=True
        )
        assert jnp.allclose(theta_param.mu, expected_mu, atol=1e-5)

    def test_define_model_3d_sigma_at_least_floor(self, mock_data, mock_priors_3d):
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        assert jnp.all(theta_param.sigma >= mock_priors_3d.sigma_floor)

    def test_define_model_3d_sigma_is_floor_when_genotypes_identical(self, mock_data):
        """When all genotypes share the same theta, sigma collapses to the floor."""
        tv = jnp.broadcast_to(
            jnp.array([[0.3, 0.5, 0.7], [0.2, 0.4, 0.6]])[..., None],
            (2, 3, mock_data.num_genotype),
        )
        floor = 0.05
        priors = ModelPriors(theta_values=tv, sigma_floor=floor)
        theta_param = define_model(name="pg", data=mock_data, priors=priors)
        assert jnp.allclose(theta_param.sigma, floor, atol=1e-5)

    def test_define_model_3d_registers_no_sample_sites(self, mock_data, mock_priors_3d):
        with seed(rng_seed=0):
            tr = trace(define_model).get_trace(
                name="pg", data=mock_data, priors=mock_priors_3d,
            )
        sample_sites = [n for n, s in tr.items() if s["type"] == "sample"]
        assert sample_sites == []
        assert "pg_theta" in tr

    def test_guide_3d_matches_model(self, mock_data, mock_priors_3d):
        model_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        guide_param = guide(name="pg", data=mock_data, priors=mock_priors_3d)
        assert jnp.allclose(model_param.theta, guide_param.theta)
        assert jnp.allclose(model_param.mu, guide_param.mu)

    def test_run_model_3d_slices_correct_genotypes(self, mock_data, mock_priors_3d):
        """run_model must pick the right per-genotype curve using geno_theta_idx."""
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        out = run_model(theta_param, mock_data._replace(scatter_theta=0))

        expected_shape = (
            mock_data.num_titrant_name,
            mock_data.num_titrant_conc,
            mock_data.geno_theta_idx.shape[0],
        )
        assert out.shape == expected_shape
        # Each output column must match the theta_values column for that genotype.
        for col, geno_idx in enumerate(mock_data.geno_theta_idx.tolist()):
            assert jnp.allclose(out[..., col],
                                 mock_priors_3d.theta_values[..., geno_idx])

    def test_run_model_3d_concentration_mapping(self, mock_data, mock_priors_3d):
        """Concentration remapping must work correctly with per-genotype theta."""
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        # Request concentrations [1.0, 0.0] — subset + reorder of [0, 1, 10].
        new_conc = jnp.array([1.0, 0.0])
        out = run_model(
            theta_param,
            mock_data._replace(titrant_conc=new_conc, scatter_theta=0),
        )
        # Expected: columns [1, 0] from theta_values, then slice by geno_theta_idx.
        expected = mock_priors_3d.theta_values[..., mock_data.geno_theta_idx][:, [1, 0], :]
        assert jnp.allclose(out, expected)

    def test_run_model_3d_scatter(self, mock_data, mock_priors_3d):
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        out = run_model(theta_param, mock_data)
        expected_shape = (
            1, 1, 1, 1,
            mock_data.num_titrant_name,
            mock_data.num_titrant_conc,
            mock_data.geno_theta_idx.shape[0],
        )
        assert out.shape == expected_shape

    def test_get_population_moments_3d(self, mock_data, mock_priors_3d):
        theta_param = define_model(name="pg", data=mock_data, priors=mock_priors_3d)
        mu, sigma = get_population_moments(theta_param, mock_data)
        expected_shape = (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
        assert mu.shape == expected_shape
        assert sigma.shape == expected_shape
        assert jnp.all(sigma > 0)
