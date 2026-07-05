import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, seed, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.tfmodel.generative.observe.base_growth import observe, guide

# --- Mock Data Fixtures ---

MockBaseGrowth = namedtuple("MockBaseGrowth", [
    "rate_obs",
    "rate_std",
    "good_mask",
    "num_genotype",
])

MockGrowth = namedtuple("MockGrowth", [
    "batch_idx",
    "batch_size",
    "scale_vector",
])

MockPriors = namedtuple("MockPriors", ["k_ref_loc", "k_ref_scale"])


@pytest.fixture
def mock_data():
    """
    Base-growth data shaped (num_genotype,) = (4,). rate_obs[g] == g so
    genotype slicing is detectable. One entry masked bad.
    """
    num_genotype = 4
    rate_obs = jnp.arange(num_genotype, dtype=float)
    rate_std = jnp.ones(num_genotype) * 0.1

    good_mask = jnp.ones(num_genotype, dtype=bool)
    good_mask = good_mask.at[0].set(False)

    return MockBaseGrowth(
        rate_obs=rate_obs,
        rate_std=rate_std,
        good_mask=good_mask,
        num_genotype=num_genotype,
    )


@pytest.fixture
def full_growth():
    """Full (no-subsample) genotype batch: idx 0..3, unit scale."""
    batch_size = 4
    return MockGrowth(
        batch_idx=jnp.arange(batch_size),
        batch_size=batch_size,
        scale_vector=jnp.ones(batch_size),
    )


@pytest.fixture
def priors():
    return MockPriors(k_ref_loc=0.5, k_ref_scale=0.2)


def _dk_geno(batch_size, value=0.1):
    """Growth-model dk_geno tensor (1,1,1,1,1,1,batch_size)."""
    return jnp.ones((1, 1, 1, 1, 1, 1, batch_size)) * value


def test_observe_k_ref_site(mock_data, full_growth, priors):
    """observe samples k_ref from a Normal(prior loc, prior scale)."""
    name = "base_growth"
    dk_geno = _dk_geno(full_growth.batch_size)

    tr = trace(seed(observe, jax.random.PRNGKey(0))).get_trace(
        name=name, data=mock_data, dk_geno=dk_geno,
        growth=full_growth, priors=priors,
    )

    k_ref_name = f"{name}_k_ref"
    assert k_ref_name in tr
    assert not tr[k_ref_name]["is_observed"]
    fn = tr[k_ref_name]["fn"]
    assert isinstance(fn, dist.Normal)
    assert jnp.allclose(fn.loc, priors.k_ref_loc)
    assert jnp.allclose(fn.scale, priors.k_ref_scale)


def test_observe_obs_site_structure(mock_data, full_growth, priors):
    """observe creates a masked Normal at '{name}_obs' with
    loc == k_ref + dk_geno_flat and scale/obs sliced by batch_idx."""
    name = "base_growth"
    fixed_k_ref = 0.7
    dk_geno = _dk_geno(full_growth.batch_size, value=0.1)

    conditioned = substitute(observe, data={f"{name}_k_ref": fixed_k_ref})
    tr = trace(seed(conditioned, jax.random.PRNGKey(1))).get_trace(
        name=name, data=mock_data, dk_geno=dk_geno,
        growth=full_growth, priors=priors,
    )

    obs_name = f"{name}_obs"
    assert obs_name in tr
    site = tr[obs_name]
    assert site["is_observed"]
    assert isinstance(site["fn"], dist.MaskedDistribution)

    base = site["fn"].base_dist
    assert isinstance(base, dist.Normal)

    dk_flat = dk_geno[0, 0, 0, 0, 0, 0, :]
    assert jnp.allclose(base.loc, fixed_k_ref + dk_flat)

    bi = full_growth.batch_idx
    assert jnp.allclose(base.scale, mock_data.rate_std[bi])
    assert jnp.allclose(site["value"], mock_data.rate_obs[bi])


def test_observe_uses_shared_genotype_plate(mock_data, full_growth, priors):
    """The likelihood plate must be the shared, unprefixed
    'shared_genotype_plate', not a name-prefixed one."""
    name = "base_growth"
    dk_geno = _dk_geno(full_growth.batch_size)

    tr = trace(seed(observe, jax.random.PRNGKey(2))).get_trace(
        name=name, data=mock_data, dk_geno=dk_geno,
        growth=full_growth, priors=priors,
    )
    plate_sites = [k for k, v in tr.items() if v["type"] == "plate"]
    assert "shared_genotype_plate" in plate_sites
    assert f"{name}_genotype_plate" not in plate_sites


def test_observe_batch_idx_slices_genotypes(mock_data, priors):
    """A subsampled batch_idx selects the corresponding genotype entries."""
    name = "base_growth"
    growth = MockGrowth(
        batch_idx=jnp.array([1, 3]),
        batch_size=2,
        scale_vector=jnp.ones(2) * 2.0,
    )
    dk_geno = _dk_geno(growth.batch_size, value=0.0)

    conditioned = substitute(observe, data={f"{name}_k_ref": 0.0})
    tr = trace(seed(conditioned, jax.random.PRNGKey(3))).get_trace(
        name=name, data=mock_data, dk_geno=dk_geno,
        growth=growth, priors=priors,
    )
    site = tr[f"{name}_obs"]
    # rate_obs[g] == g, so batch [1, 3] -> observed values [1, 3].
    assert jnp.allclose(site["value"], jnp.array([1.0, 3.0]))
    assert jnp.all(site["scale"] == 2.0)


def test_observe_masking(mock_data, full_growth, priors):
    """Masked genotype (idx 0) contributes zero log-prob even with wrong pred."""
    name = "base_growth"
    # Huge k_ref makes every prediction wildly wrong; only masking saves idx 0.
    dk_geno = _dk_geno(full_growth.batch_size, value=0.0)
    conditioned = substitute(observe, data={f"{name}_k_ref": 1000.0})
    tr = trace(seed(conditioned, jax.random.PRNGKey(4))).get_trace(
        name=name, data=mock_data, dk_geno=dk_geno,
        growth=full_growth, priors=priors,
    )
    log_probs = tr[f"{name}_obs"]["fn"].log_prob(tr[f"{name}_obs"]["value"])
    assert log_probs[0] == 0.0        # masked
    assert log_probs[1] != 0.0        # unmasked


def test_guide_registers_k_ref(mock_data, full_growth, priors):
    """The guide registers the k_ref variational site backed by pyro.params,
    with the scale constrained > 1e-4 (guards against scale collapse)."""
    name = "base_growth"
    dk_geno = _dk_geno(full_growth.batch_size)

    tr = trace(seed(guide, jax.random.PRNGKey(5))).get_trace(
        name=name, data=mock_data, dk_geno=dk_geno,
        growth=full_growth, priors=priors,
    )

    assert f"{name}_k_ref_loc" in tr
    assert f"{name}_k_ref_scale" in tr
    assert f"{name}_k_ref" in tr
    assert isinstance(tr[f"{name}_k_ref"]["fn"], dist.Normal)

    # loc param initialized from the prior loc.
    assert jnp.allclose(tr[f"{name}_k_ref_loc"]["value"], priors.k_ref_loc)

    # scale param carries the greater_than(1e-4) constraint.
    scale_site = tr[f"{name}_k_ref_scale"]
    constraint = scale_site["kwargs"]["constraint"]
    assert isinstance(constraint, dist.constraints._GreaterThan)
    assert constraint.lower_bound == pytest.approx(1e-4)
