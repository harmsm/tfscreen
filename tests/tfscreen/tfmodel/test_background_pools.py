"""
Per-background congression pools (planning/dimer-thermo-model.md step 3d):
each genetic background is its own transformed pool, so each library's
co-resident sets and bulk fractions come from its background's table.
"""
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro.handlers import seed, trace

from tfscreen.genetics.library_design import (
    library_composition_table,
    write_library_composition,
)
from tfscreen.tfmodel.generative.components.transformation import mixture
from tfscreen.tfmodel.inference.batch_safety import (
    find_orchestrator_batch_order_mismatches,
)
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "smoke-tests", "test_data")
_GROWTH_CSV = os.path.join(_SMOKE, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE, "binding-smoke.csv")
_LIBRARY_YAML = os.path.join(_SMOKE, "library-smoke.yaml")


def _tables(tmp_path):
    ref = library_composition_table(_LIBRARY_YAML)
    plain = tmp_path / "plain.csv"
    write_library_composition(ref, plain)
    # A second pool over the same names: no spikes, a different mix.
    bg = ref.copy()
    bg["bulk_fraction"] = 1.0
    bg["in_spiked_origin"] = False
    w = np.linspace(1.0, 3.0, len(bg))
    bg["pool_fraction"] = w / w.sum()
    both = pd.concat([ref.assign(background="reference"),
                      bg.assign(background="bg2")], ignore_index=True)
    pools = tmp_path / "pools.csv"
    write_library_composition(both, pools)
    return str(plain), str(pools)


def _growth():
    g = pd.read_csv(_GROWTH_CSV)
    g["background"] = np.where(g["library"] == "pheS", "bg2", "reference")
    return g


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    plain, pools = _tables(tmp_path_factory.mktemp("pools"))
    kw = dict(binding_df=_BINDING_CSV, transformation="mixture",
              transformation_lambda=(0.36, 0.05), batch_size=None)
    legacy = ModelOrchestrator(growth_df=_GROWTH_CSV, library_file=plain, **kw)
    pooled = ModelOrchestrator(growth_df=_growth(), library_file=pools, **kw)
    return legacy, pooled


def _slot_library(o):
    tm = o.growth_tm
    slots = list(tm.tensor_dim_labels[tm.tensor_dim_names.index("condition_pre")])
    lib = (o.growth_df[["_condition_pre_key", "library"]].drop_duplicates()
           .set_index("_condition_pre_key")["library"])
    return [str(lib[s]) for s in slots]


def test_pool_fields(models):
    legacy, pooled = models
    libs = _slot_library(pooled)
    order = list(dict.fromkeys(libs))
    for o in (legacy, pooled):
        g = o.data.growth
        assert g.num_pools == 2                      # one per library
        assert list(g.condition_pre_pool) == [order.index(lib) for lib in libs]
    lg, pg = legacy.data.growth, pooled.data.growth
    # without backgrounds both libraries draw from the one table ...
    lb = np.asarray(lg.pool_bulk_fraction)
    np.testing.assert_allclose(lb[0], lb[1])
    # ... but each draws its own co-resident sets
    assert not np.array_equal(np.asarray(lg.pool_coresident_idx[0]),
                              np.asarray(lg.pool_coresident_idx[1]))
    # the first library draws as the single pool always did
    np.testing.assert_array_equal(np.asarray(lg.pool_coresident_idx[0]),
                                  np.asarray(lg.coresident_idx))
    # with backgrounds, pheS (bg2) takes bg2's table
    q = order.index("pheS")
    assert np.all(np.asarray(pg.pool_bulk_fraction)[q] == 1.0)
    np.testing.assert_allclose(np.asarray(pg.pool_bulk_fraction)[1 - q], lb[0])
    np.testing.assert_array_equal(np.asarray(pg.pool_coresident_idx[0]),
                                  np.asarray(lg.pool_coresident_idx[0]))


def test_one_library_has_one_pool(tmp_path):
    plain, _ = _tables(tmp_path)
    g = pd.read_csv(_GROWTH_CSV)
    g = g[g["library"] == "kanR"]
    o = ModelOrchestrator(growth_df=g, binding_df=_BINDING_CSV,
                          library_file=plain, batch_size=None,
                          transformation="mixture",
                          transformation_lambda=(0.36, 0.05))
    assert o.data.growth.num_pools == 1
    assert o.data.growth.pool_coresident_idx is None


def test_each_slot_uses_its_pool(models):
    _, o = models
    g = o.get_batch(o.data, jnp.arange(o.data.num_genotype)).growth
    G = g.num_genotype
    T, C = g.num_titrant_name, g.num_titrant_conc
    rng = np.random.default_rng(0)
    theta = jnp.asarray(rng.uniform(0.05, 0.95, (1, 1, 1, 1, T, C, G)))
    act = jnp.ones((1, 1, 1, 1, 1, 1, G))
    dk = jnp.asarray(rng.normal(0, 0.01, (1, 1, 1, 1, 1, 1, G)))
    focal = (theta, act, dk)
    population = (theta, jnp.ones(G), dk.reshape(G))
    lam = jnp.asarray(0.4)
    picked = mixture.cell_classes(focal, population, lam, g)
    for slot, q in enumerate(g.condition_pre_pool):
        one = mixture._pool_classes(focal, population, lam, g,
                                    g.pool_coresident_idx[q],
                                    g.pool_bulk_fraction[q])
        for field in ("theta", "dk_geno", "log_weight"):
            got = np.asarray(getattr(picked, field))[:, :, :, slot]
            want = np.asarray(getattr(one, field))[:, :, :, 0]
            np.testing.assert_allclose(got, np.broadcast_to(want, got.shape))


def test_pooled_model_traces_and_is_batch_safe(models):
    _, o = models
    data = o.get_batch(o.data, jnp.arange(o.data.num_genotype))
    tr = trace(seed(o.jax_model, 0)).get_trace(data=data, priors=o.priors)
    assert np.all(np.isfinite(np.asarray(tr["growth_pred"]["value"])))
    assert find_orchestrator_batch_order_mismatches(o) == {}


def test_library_with_two_backgrounds_is_refused(tmp_path):
    _, pools = _tables(tmp_path)
    g = _growth()
    g.loc[g.index[:5], "background"] = "bg2"      # some kanR tubes too
    with pytest.raises(ValueError, match="one background"):
        ModelOrchestrator(growth_df=g, binding_df=_BINDING_CSV,
                          library_file=pools, batch_size=None)


def test_background_column_without_declared_backgrounds(tmp_path):
    plain, _ = _tables(tmp_path)
    with pytest.raises(ValueError, match="library table has none"):
        ModelOrchestrator(growth_df=_growth(), binding_df=_BINDING_CSV,
                          library_file=plain, batch_size=None)
