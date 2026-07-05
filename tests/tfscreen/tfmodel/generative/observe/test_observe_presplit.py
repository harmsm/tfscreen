import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.tfmodel.generative.observe.presplit import observe, guide

# --- Mock Data Fixtures ---

MockPreSplit = namedtuple("MockPreSplit", [
    "ln_cfu_t0",
    "ln_cfu_t0_std",
    "good_mask",
    "num_replicate",
    "num_condition_pre",
    "num_genotype",
])

MockGrowth = namedtuple("MockGrowth", [
    "batch_idx",
    "batch_size",
    "scale_vector",
])


@pytest.fixture
def mock_data():
    """
    Pre-split data shaped (num_replicate, num_condition_pre, num_genotype)
    = (2, 1, 4). ln_cfu_t0[r, c, g] == g so genotype slicing is detectable.
    One entry masked bad.
    """
    num_replicate = 2
    num_condition_pre = 1
    num_genotype = 4
    shape = (num_replicate, num_condition_pre, num_genotype)

    ln_cfu_t0 = jnp.broadcast_to(jnp.arange(num_genotype, dtype=float), shape)
    ln_cfu_t0_std = jnp.ones(shape) * 0.2

    good_mask = jnp.ones(shape, dtype=bool)
    good_mask = good_mask.at[0, 0, 0].set(False)

    return MockPreSplit(
        ln_cfu_t0=ln_cfu_t0,
        ln_cfu_t0_std=ln_cfu_t0_std,
        good_mask=good_mask,
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
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


def _ln_cfu0(batch_size, value=5.0):
    """Growth-model ln_cfu0 tensor (num_rep, 1, num_cp, 1, 1, 1, batch_size)."""
    return jnp.ones((2, 1, 1, 1, 1, 1, batch_size)) * value


def test_observe_site_structure(mock_data, full_growth):
    """observe creates a masked Normal at '{name}_obs' with the right loc,
    scale, and observed value (ln_cfu0 squeezed, std/obs sliced by batch)."""
    name = "test"
    ln_cfu0 = _ln_cfu0(full_growth.batch_size, value=5.0)

    tr = trace(observe).get_trace(
        name=name, data=mock_data, ln_cfu0=ln_cfu0, growth=full_growth,
    )

    obs_name = f"{name}_obs"
    assert obs_name in tr
    site = tr[obs_name]
    assert site["is_observed"]
    assert isinstance(site["fn"], dist.MaskedDistribution)

    base = site["fn"].base_dist
    assert isinstance(base, dist.Normal)

    # ln_cfu0 squeezed to (num_rep, num_cp, batch_size)
    expected_loc = ln_cfu0[:, 0, :, 0, 0, 0, :]
    assert jnp.allclose(base.loc, expected_loc)

    bi = full_growth.batch_idx
    assert jnp.allclose(base.scale, mock_data.ln_cfu_t0_std[:, :, bi])
    assert jnp.allclose(site["value"], mock_data.ln_cfu_t0[:, :, bi])


def test_observe_uses_shared_genotype_plate(mock_data, full_growth):
    """The innermost plate must be the shared, unprefixed 'shared_genotype_plate'
    (so Numpyro shares the genotype subsample with the growth model), never a
    name-prefixed plate."""
    name = "test"
    ln_cfu0 = _ln_cfu0(full_growth.batch_size)

    tr = trace(observe).get_trace(
        name=name, data=mock_data, ln_cfu0=ln_cfu0, growth=full_growth,
    )

    plate_sites = [k for k, v in tr.items() if v["type"] == "plate"]
    assert "shared_genotype_plate" in plate_sites
    assert f"{name}_genotype_plate" not in plate_sites
    # The two outer plates ARE name-prefixed.
    assert f"{name}_replicate" in plate_sites
    assert f"{name}_condition_pre" in plate_sites


def test_observe_batch_idx_slices_genotypes(mock_data):
    """A subsampled batch_idx must select the corresponding genotype columns of
    the pre-split tensors."""
    name = "test"
    batch_idx = jnp.array([1, 3])
    growth = MockGrowth(
        batch_idx=batch_idx,
        batch_size=2,
        scale_vector=jnp.ones(2) * 2.0,
    )
    ln_cfu0 = _ln_cfu0(growth.batch_size)

    tr = trace(observe).get_trace(
        name=name, data=mock_data, ln_cfu0=ln_cfu0, growth=growth,
    )
    site = tr[f"{name}_obs"]

    # ln_cfu_t0[r, c, g] == g, so the observed values should be [1, 3].
    assert jnp.allclose(site["value"][:, :, 0], 1.0)
    assert jnp.allclose(site["value"][:, :, 1], 3.0)


def test_observe_scaling(mock_data):
    """The sub-sampling scale vector from growth is applied to the site."""
    name = "test"
    growth = MockGrowth(
        batch_idx=jnp.array([1, 3]),
        batch_size=2,
        scale_vector=jnp.ones(2) * 2.0,
    )
    ln_cfu0 = _ln_cfu0(growth.batch_size)

    tr = trace(observe).get_trace(
        name=name, data=mock_data, ln_cfu0=ln_cfu0, growth=growth,
    )
    assert jnp.all(tr[f"{name}_obs"]["scale"] == 2.0)


def test_observe_masking(mock_data, full_growth):
    """Masked entries contribute zero log-prob even with a wildly wrong pred."""
    name = "test"
    # ln_cfu0 way off everywhere; masked entry is (rep=0, cp=0, geno=0).
    ln_cfu0 = _ln_cfu0(full_growth.batch_size, value=1000.0)

    tr = trace(observe).get_trace(
        name=name, data=mock_data, ln_cfu0=ln_cfu0, growth=full_growth,
    )
    log_probs = tr[f"{name}_obs"]["fn"].log_prob(tr[f"{name}_obs"]["value"])

    assert log_probs[0, 0, 0] == 0.0        # masked
    assert log_probs[1, 0, 0] != 0.0        # unmasked


def test_guide_is_noop(mock_data, full_growth):
    """The guide introduces no sample/param sites (presplit has no latents)."""
    name = "test"
    ln_cfu0 = _ln_cfu0(full_growth.batch_size)

    tr = trace(seed(guide, jax.random.PRNGKey(0))).get_trace(
        name=name, data=mock_data, ln_cfu0=ln_cfu0, growth=full_growth,
    )
    assert len(tr) == 0
