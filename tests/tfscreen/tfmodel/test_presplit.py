"""
Tests for the optional pre-split (t = -t_pre) data path.

Coverage:
  - _read_presplit_df: column checks, silent genotype dropping
  - _build_presplit_tm: tensor shape/alignment with growth_tm
  - ModelOrchestrator: presplit=None (no change), presplit passed in
  - jax_model: presplit observation site fires when data.presplit is set
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, seed

from tfscreen.tfmodel.model_orchestrator import (
    _read_presplit_df,
    _build_presplit_tm,
    _read_growth_df,
    _build_growth_tm,
)
from tfscreen.tfmodel.data_class import PreSplitData, DataClass
from tfscreen.tfmodel.tensors.populate_dataclass import populate_dataclass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_growth_df():
    """Minimal growth DataFrame with two genotypes, two replicates, two
    condition_pre values — enough to exercise the presplit alignment logic."""
    rows = []
    for rep in [1, 2]:
        for cp in ["kanR-cond", "pheS-cond"]:
            for geno in ["wt", "A1V", "A2V"]:
                for t_sel in [60.0, 90.0]:
                    rows.append({
                        "library": "lib",
                        "replicate": rep,
                        "condition_pre": cp,
                        "condition_sel": cp,
                        "titrant_name": "iptg",
                        "titrant_conc": 0.0,
                        "t_pre": 30.0,
                        "t_sel": t_sel,
                        "genotype": geno,
                        "ln_cfu": 10.0,
                        "ln_cfu_std": 0.5,
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def growth_tm(minimal_growth_df):
    gdf = _read_growth_df(minimal_growth_df.copy())
    return _build_growth_tm(gdf)


@pytest.fixture
def minimal_presplit_df():
    """Pre-split data for all (replicate, condition_pre, genotype) combos."""
    rows = []
    for rep in [1, 2]:
        for cp in ["kanR-cond", "pheS-cond"]:
            for geno in ["wt", "A1V", "A2V"]:
                rows.append({
                    "library": "lib",
                    "replicate": rep,
                    "condition_pre": cp,
                    "genotype": geno,
                    "ln_cfu": 9.5,
                    "ln_cfu_std": 0.4,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _read_presplit_df
# ---------------------------------------------------------------------------

def test_read_presplit_df_passes_through_valid(minimal_presplit_df,
                                               minimal_growth_df):
    gdf = _read_growth_df(minimal_growth_df.copy())
    result = _read_presplit_df(minimal_presplit_df, gdf)
    assert len(result) == len(minimal_presplit_df)
    for col in ["replicate", "condition_pre", "genotype", "ln_cfu", "ln_cfu_std"]:
        assert col in result.columns


def test_read_presplit_df_drops_unknown_genotypes(minimal_presplit_df,
                                                   minimal_growth_df, capsys):
    df_extra = minimal_presplit_df.copy()
    extra_row = pd.DataFrame([{"replicate": 1, "condition_pre": "kanR-cond",
                                "genotype": "A99V",
                                "ln_cfu": 5.0, "ln_cfu_std": 0.1}])
    df_extra = pd.concat([df_extra, extra_row], ignore_index=True)

    gdf = _read_growth_df(minimal_growth_df.copy())
    result = _read_presplit_df(df_extra, gdf)
    captured = capsys.readouterr()
    assert "will be dropped" in captured.out
    assert "A99V" in captured.out

    assert "A99V" not in result["genotype"].values
    assert len(result) == len(minimal_presplit_df)


def test_read_presplit_df_missing_column_raises(minimal_growth_df):
    bad_df = pd.DataFrame({
        "replicate": [1], "condition_pre": ["kanR-cond"],
        "genotype": ["wt"], "ln_cfu": [9.0]
        # missing ln_cfu_std
    })
    gdf = _read_growth_df(minimal_growth_df.copy())
    with pytest.raises(Exception):
        _read_presplit_df(bad_df, gdf)


# ---------------------------------------------------------------------------
# _build_presplit_tm
# ---------------------------------------------------------------------------

def test_build_presplit_tm_shape(minimal_presplit_df, minimal_growth_df,
                                  growth_tm):
    gdf = _read_growth_df(minimal_growth_df.copy())
    psdf = _read_presplit_df(minimal_presplit_df, gdf)
    ps_tm = _build_presplit_tm(psdf, growth_tm)

    # shape should be (num_replicate, num_condition_pre, num_genotype)
    assert ps_tm.tensor_shape == (2, 2, 3)
    assert "ln_cfu" in ps_tm.tensors
    assert "ln_cfu_std" in ps_tm.tensors
    assert "good_mask" in ps_tm.tensors


def test_build_presplit_tm_genotype_alignment(minimal_presplit_df,
                                               minimal_growth_df, growth_tm):
    """Genotype ordering in presplit_tm must match growth_tm."""
    gdf = _read_growth_df(minimal_growth_df.copy())
    psdf = _read_presplit_df(minimal_presplit_df, gdf)
    ps_tm = _build_presplit_tm(psdf, growth_tm)

    growth_geno_labels = growth_tm.tensor_dim_labels[
        growth_tm.tensor_dim_names.index("genotype")
    ]
    presplit_geno_labels = ps_tm.tensor_dim_labels[
        ps_tm.tensor_dim_names.index("genotype")
    ]
    assert list(growth_geno_labels) == list(presplit_geno_labels)


def test_build_presplit_tm_partial_coverage(minimal_growth_df, growth_tm):
    """Genotypes absent from presplit_df get NaN (masked) in the tensor."""
    partial_df = pd.DataFrame([
        {"library": "lib", "replicate": 1, "condition_pre": "kanR-cond",
         "genotype": "wt", "ln_cfu": 9.5, "ln_cfu_std": 0.4},
    ])
    gdf = _read_growth_df(minimal_growth_df.copy())
    psdf = _read_presplit_df(partial_df, gdf)
    ps_tm = _build_presplit_tm(psdf, growth_tm)

    # Only one entry should be valid; the rest masked
    assert ps_tm.tensors["good_mask"].sum() == 1


# ---------------------------------------------------------------------------
# ModelOrchestrator integration (light smoke test, no JAX tracing)
# ---------------------------------------------------------------------------

def test_model_orchestrator_without_presplit(minimal_growth_df,
                                              minimal_presplit_df):
    """Passing presplit_df=None leaves data.presplit as None."""
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    binding_df = pd.DataFrame({
        "genotype": ["wt", "A1V", "A2V"],
        "titrant_name": ["iptg"] * 3,
        "titrant_conc": [0.0] * 3,
        "theta_obs": [0.5] * 3,
        "theta_std": [0.1] * 3,
    })
    orchestrator = ModelOrchestrator(minimal_growth_df, binding_df, presplit_df=None)
    assert orchestrator.data.presplit is None


def test_model_orchestrator_with_presplit(minimal_growth_df,
                                          minimal_presplit_df):
    """Passing presplit_df populates data.presplit as a PreSplitData."""
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    binding_df = pd.DataFrame({
        "genotype": ["wt", "A1V", "A2V"],
        "titrant_name": ["iptg"] * 3,
        "titrant_conc": [0.0] * 3,
        "theta_obs": [0.5] * 3,
        "theta_std": [0.1] * 3,
    })
    orchestrator = ModelOrchestrator(minimal_growth_df, binding_df,
                           presplit_df=minimal_presplit_df)
    ps = orchestrator.data.presplit
    assert ps is not None
    assert isinstance(ps, PreSplitData)
    assert ps.num_replicate == 2
    assert ps.num_condition_pre == 2
    assert ps.num_genotype == 3
    assert ps.ln_cfu_t0.shape == (2, 2, 3)
    assert ps.good_mask.shape == (2, 2, 3)
    # All entries should be valid (full coverage)
    assert bool(jnp.all(ps.good_mask))


# ---------------------------------------------------------------------------
# jax_model: presplit_obs site fires
# ---------------------------------------------------------------------------

def test_jax_model_presplit_obs_site_present(minimal_growth_df,
                                              minimal_presplit_df):
    """The presplit_obs sample site should appear in the model trace when
    data.presplit is not None."""
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    binding_df = pd.DataFrame({
        "genotype": ["wt", "A1V", "A2V"],
        "titrant_name": ["iptg"] * 3,
        "titrant_conc": [0.0] * 3,
        "theta_obs": [0.5] * 3,
        "theta_std": [0.1] * 3,
    })
    orchestrator = ModelOrchestrator(minimal_growth_df, binding_df,
                           presplit_df=minimal_presplit_df)

    model_fn = orchestrator.jax_model
    tr = trace(seed(model_fn, 0)).get_trace(orchestrator.data, orchestrator.priors)
    assert "presplit_obs" in tr


def test_jax_model_no_presplit_obs_site_without_data(minimal_growth_df):
    """The presplit_obs site must NOT appear when presplit_df is not given."""
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    binding_df = pd.DataFrame({
        "genotype": ["wt", "A1V", "A2V"],
        "titrant_name": ["iptg"] * 3,
        "titrant_conc": [0.0] * 3,
        "theta_obs": [0.5] * 3,
        "theta_std": [0.1] * 3,
    })
    orchestrator = ModelOrchestrator(minimal_growth_df, binding_df)

    model_fn = orchestrator.jax_model
    tr = trace(seed(model_fn, 0)).get_trace(orchestrator.data, orchestrator.priors)
    assert "presplit_obs" not in tr
