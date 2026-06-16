import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed

from tfscreen.tfmodel.generative.components.noise.logit_normal import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "sigma_logit_scale" in params
    assert params["sigma_logit_scale"] == 0.5


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.sigma_logit_scale == 0.5


def test_get_guesses():
    name = "theta_growth_noise"
    guesses = get_guesses(name, None)
    assert isinstance(guesses, dict)
    expected_key = f"{name}_sigma_logit"
    assert expected_key in guesses
    assert guesses[expected_key] > 0.0


# ---------------------------------------------------------------------------
# define_model
# ---------------------------------------------------------------------------

@pytest.fixture
def name():
    return "test_logit_normal"


@pytest.fixture
def priors():
    return get_priors()


@pytest.fixture
def fx_calc():
    # 2-D: (titrant_conc=2, genotype=3) matching the minimal model tensor shape
    return jnp.array([[0.1, 0.5, 0.9],
                      [0.2, 0.4, 0.8]])


@pytest.fixture
def model_setup(name, priors):
    guesses = get_guesses(name, None)
    guesses[f"{name}_epsilon"] = jnp.zeros((2, 3))
    substituted = substitute(define_model, data=guesses)
    return {"name": name, "priors": priors, "substituted": substituted}


def test_define_model_output_shape(model_setup, fx_calc):
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted = model_setup["substituted"]

    rng_key = jax.random.PRNGKey(0)
    seeded = seed(substituted, rng_key)
    out = seeded(name=name, fx_calc=fx_calc, priors=priors)
    assert out.shape == fx_calc.shape


def test_define_model_output_in_unit_interval(model_setup, fx_calc):
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted = model_setup["substituted"]

    rng_key = jax.random.PRNGKey(0)
    seeded = seed(substituted, rng_key)
    out = seeded(name=name, fx_calc=fx_calc, priors=priors)
    assert jnp.all(out > 0.0)
    assert jnp.all(out < 1.0)


def test_define_model_zero_epsilon_recovers_input(name, priors, fx_calc):
    """With epsilon fixed to zero, output should match input (up to clipping)."""
    guesses = {
        f"{name}_sigma_logit": jnp.array(1e-3),
        f"{name}_epsilon": jnp.zeros(fx_calc.shape),
    }
    substituted = substitute(define_model, data=guesses)
    rng_key = jax.random.PRNGKey(0)
    out = seed(substituted, rng_key)(name=name, fx_calc=fx_calc, priors=priors)
    assert jnp.allclose(out, fx_calc, atol=1e-5)


def test_define_model_sample_sites(model_setup, fx_calc):
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted = model_setup["substituted"]

    rng_key = jax.random.PRNGKey(0)
    model_trace = trace(seed(substituted, rng_key)).get_trace(
        name=name, fx_calc=fx_calc, priors=priors
    )
    assert f"{name}_sigma_logit" in model_trace
    assert f"{name}_epsilon" in model_trace
    assert name in model_trace  # deterministic site

    sigma_site = model_trace[f"{name}_sigma_logit"]
    assert isinstance(sigma_site["fn"], dist.HalfNormal)

    eps_site = model_trace[f"{name}_epsilon"]
    assert isinstance(eps_site["fn"], dist.Normal)
    assert eps_site["value"].shape == fx_calc.shape


def test_define_model_large_sigma_spreads_output(name, priors, fx_calc):
    """Large sigma_logit should produce outputs spread away from the input."""
    large_sigma = 10.0
    rng_key = jax.random.PRNGKey(42)
    results = []
    for seed_val in range(20):
        guesses = {
            f"{name}_sigma_logit": jnp.array(large_sigma),
            f"{name}_epsilon": jax.random.normal(
                jax.random.PRNGKey(seed_val), fx_calc.shape
            ) * large_sigma,
        }
        substituted = substitute(define_model, data=guesses)
        out = seed(substituted, rng_key)(name=name, fx_calc=fx_calc, priors=priors)
        results.append(out)
    results = jnp.stack(results)
    # With large sigma the std across samples should be substantial
    # Check the (0, 1) element: titrant_conc=0, genotype=1 => theta=0.5, most sensitive
    assert jnp.std(results[:, 0, 1]) > 0.1


def test_define_model_small_sigma_stays_near_input(name, priors, fx_calc):
    """Small sigma_logit should produce outputs close to the input."""
    tiny_sigma = 1e-4
    guesses = {
        f"{name}_sigma_logit": jnp.array(tiny_sigma),
        f"{name}_epsilon": jnp.zeros(fx_calc.shape),
    }
    substituted = substitute(define_model, data=guesses)
    rng_key = jax.random.PRNGKey(0)
    out = seed(substituted, rng_key)(name=name, fx_calc=fx_calc, priors=priors)
    assert jnp.allclose(out, fx_calc, atol=1e-3)


def test_define_model_boundary_safety(name, priors):
    """Boundary values (0, 1) should not produce NaN or inf."""
    fx_boundary = jnp.array([[0.0, 1.0, 0.5]])  # shape (1, 3): 1 titrant_conc, 3 genotypes
    guesses = {
        f"{name}_sigma_logit": jnp.array(1e-3),
        f"{name}_epsilon": jnp.zeros((1, 3)),
    }
    substituted = substitute(define_model, data=guesses)
    rng_key = jax.random.PRNGKey(0)
    out = seed(substituted, rng_key)(name=name, fx_calc=fx_boundary, priors=priors)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out > 0.0)
    assert jnp.all(out < 1.0)


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

def test_guide_creates_sigma_logit_params(name, priors, fx_calc):
    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, fx_calc=fx_calc, priors=priors
        )

    assert f"{name}_sigma_logit_loc" in guide_trace
    assert f"{name}_sigma_logit_scale" in guide_trace
    assert f"{name}_sigma_logit" in guide_trace
    assert isinstance(guide_trace[f"{name}_sigma_logit"]["fn"], dist.LogNormal)


def test_guide_epsilon_shape(name, priors, fx_calc):
    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, fx_calc=fx_calc, priors=priors
        )
    assert f"{name}_epsilon" in guide_trace
    assert guide_trace[f"{name}_epsilon"]["value"].shape == fx_calc.shape


def test_guide_output_in_unit_interval(name, priors, fx_calc):
    with seed(rng_seed=1):
        out = guide(name=name, fx_calc=fx_calc, priors=priors)
    assert out.shape == fx_calc.shape
    assert jnp.all(out > 0.0)
    assert jnp.all(out < 1.0)


def test_guide_sigma_scale_constraint(name, priors, fx_calc):
    """sigma_logit_scale variational parameter must be positive."""
    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name, fx_calc=fx_calc, priors=priors
        )
    scale_val = guide_trace[f"{name}_sigma_logit_scale"]["value"]
    assert float(scale_val) > 0.0
