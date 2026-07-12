"""
Tests for the optional base_growth (direct reference-condition growth-rate)
calibration path.

Coverage:
  - ModelOrchestrator: base_growth_df=None (no change), base_growth_df passed in
  - jax_model: base_growth_obs / base_growth_k_ref sites fire (model and guide)
    when data.base_growth is set, and are absent when it is not.
  - Full tfs-configure-model / tfs-fit-model CLI round trip: base_growth_df
    survives write_configuration -> YAML -> read_configuration.
"""

import os

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, seed

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.data_class import BaseGrowthData
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.configuration_io import read_configuration


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_growth_df():
    """Minimal growth DataFrame with three genotypes (including wt)."""
    rows = []
    for rep in [1, 2]:
        for geno in ["wt", "A1V", "A2V"]:
            for t_sel in [60.0, 90.0]:
                rows.append({
                    "library": "lib",
                    "replicate": rep,
                    "condition_pre": "-kan",
                    "condition_sel": "+kan",
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
def minimal_binding_df():
    return pd.DataFrame({
        "genotype": ["wt", "A1V", "A2V"],
        "titrant_name": ["iptg"] * 3,
        "titrant_conc": [0.0] * 3,
        "theta_obs": [0.5] * 3,
        "theta_std": [0.1] * 3,
    })


@pytest.fixture
def minimal_base_growth_df():
    return pd.DataFrame({
        "genotype": ["wt", "A1V"],
        "rate": [0.021, 0.018],
        "rate_std": [0.001, 0.002],
    })


# ---------------------------------------------------------------------------
# ModelOrchestrator integration (light smoke test, no JAX tracing)
# ---------------------------------------------------------------------------

def test_model_orchestrator_without_base_growth(minimal_growth_df,
                                                  minimal_binding_df):
    """Passing base_growth_df=None leaves data.base_growth as None."""
    orchestrator = ModelOrchestrator(minimal_growth_df, minimal_binding_df,
                                     base_growth_df=None)
    assert orchestrator.data.base_growth is None
    assert orchestrator.priors.growth.base_growth is None


def test_model_orchestrator_with_base_growth(minimal_growth_df,
                                             minimal_binding_df,
                                             minimal_base_growth_df):
    """Passing base_growth_df populates data.base_growth as a BaseGrowthData
    and priors.growth.base_growth with a k_ref prior derived from wt."""
    orchestrator = ModelOrchestrator(minimal_growth_df, minimal_binding_df,
                                     base_growth_df=minimal_base_growth_df)
    bg = orchestrator.data.base_growth
    assert bg is not None
    assert isinstance(bg, BaseGrowthData)
    assert bg.num_genotype == 3
    assert bg.rate_obs.shape == (3,)
    assert bg.good_mask.sum() == 2

    base_growth_priors = orchestrator.priors.growth.base_growth
    assert base_growth_priors is not None
    assert base_growth_priors.k_ref_loc == pytest.approx(0.021)


def test_model_orchestrator_rejects_zero_rate_std(minimal_growth_df,
                                                   minimal_binding_df):
    """A base_growth_df with rate_std == 0.0 (e.g. an un-noised tfs-simulate
    base_growth_data output) must be rejected at construction time with a
    clear error, not silently produce a NaN k_ref prior that only surfaces
    much later as a numpyro 'invalid loc parameter' crash during tracing."""
    bad_base_growth_df = pd.DataFrame({
        "genotype": ["wt"], "rate": [0.021], "rate_std": [0.0],
    })
    with pytest.raises(ValueError, match="non-positive rate_std"):
        ModelOrchestrator(minimal_growth_df, minimal_binding_df,
                         base_growth_df=bad_base_growth_df)


# ---------------------------------------------------------------------------
# jax_model: base_growth_obs / base_growth_k_ref sites fire
# ---------------------------------------------------------------------------

def test_jax_model_base_growth_sites_present(minimal_growth_df,
                                              minimal_binding_df,
                                              minimal_base_growth_df):
    """base_growth_obs and base_growth_k_ref sample sites appear in the
    model trace when data.base_growth is not None."""
    orchestrator = ModelOrchestrator(minimal_growth_df, minimal_binding_df,
                                     base_growth_df=minimal_base_growth_df)

    model_fn = orchestrator.jax_model
    tr = trace(seed(model_fn, 0)).get_trace(orchestrator.data, orchestrator.priors)
    assert "base_growth_obs" in tr
    assert "base_growth_k_ref" in tr


def test_jax_model_no_base_growth_sites_without_data(minimal_growth_df,
                                                       minimal_binding_df):
    """Neither site appears when base_growth_df is not given."""
    orchestrator = ModelOrchestrator(minimal_growth_df, minimal_binding_df)

    model_fn = orchestrator.jax_model
    tr = trace(seed(model_fn, 0)).get_trace(orchestrator.data, orchestrator.priors)
    assert "base_growth_obs" not in tr
    assert "base_growth_k_ref" not in tr


def test_jax_model_guide_base_growth_k_ref_site_present(minimal_growth_df,
                                                          minimal_binding_df,
                                                          minimal_base_growth_df):
    """The guide must register base_growth_k_ref (via its own pyro.param
    variational parameters) but never the observation site itself."""
    orchestrator = ModelOrchestrator(minimal_growth_df, minimal_binding_df,
                                     base_growth_df=minimal_base_growth_df)

    guide_fn = orchestrator.jax_model_guide
    tr = trace(seed(guide_fn, 0)).get_trace(orchestrator.data, orchestrator.priors)
    assert "base_growth_k_ref" in tr
    assert "base_growth_obs" not in tr


def test_base_growth_obs_masks_genotypes_without_measurement(minimal_growth_df,
                                                               minimal_binding_df,
                                                               minimal_base_growth_df):
    """A2V has no base_growth measurement, so its base_growth_obs log_prob
    contribution must be exactly zero (fully masked out)."""
    orchestrator = ModelOrchestrator(minimal_growth_df, minimal_binding_df,
                                     base_growth_df=minimal_base_growth_df)

    model_fn = orchestrator.jax_model
    tr = trace(seed(model_fn, 0)).get_trace(orchestrator.data, orchestrator.priors)

    site = tr["base_growth_obs"]
    log_prob = site["fn"].log_prob(site["value"])

    genotype_labels = orchestrator.growth_tm.tensor_dim_labels[
        orchestrator.growth_tm.tensor_dim_names.index("genotype")
    ]
    batch_idx = np.asarray(orchestrator.data.growth.batch_idx)
    a2v_positions = [i for i, geno_idx in enumerate(batch_idx)
                     if str(genotype_labels[geno_idx]) == "A2V"]
    assert len(a2v_positions) > 0
    assert np.allclose(np.asarray(log_prob)[a2v_positions], 0.0)


# ---------------------------------------------------------------------------
# Full CLI round trip: tfs-configure-model -> tfs_configure_config.yaml ->
# tfs-fit-model (via read_configuration)
# ---------------------------------------------------------------------------

def test_configure_model_to_read_configuration_round_trip(tmp_path,
                                                           minimal_growth_df,
                                                           minimal_binding_df,
                                                           minimal_base_growth_df):
    """base_growth_df must survive the real tfs-configure-model -> YAML ->
    tfs-fit-model (read_configuration) round trip, ending up as a proper
    BaseGrowthData on the orchestrator rebuilt from the config file."""
    growth_path = os.path.join(tmp_path, "growth.csv")
    binding_path = os.path.join(tmp_path, "binding.csv")
    base_growth_path = os.path.join(tmp_path, "base_growth.csv")
    minimal_growth_df.to_csv(growth_path, index=False)
    minimal_binding_df.to_csv(binding_path, index=False)
    minimal_base_growth_df.to_csv(base_growth_path, index=False)

    out_prefix = os.path.join(tmp_path, "tfs_configure")
    configure_model(binding_path, growth_df=growth_path,
                    base_growth_df=base_growth_path,
                    out_prefix=out_prefix,
                    theta_growth_noise_model="zero")

    config_path = f"{out_prefix}_config.yaml"
    orchestrator, init_params = read_configuration(config_path)

    bg = orchestrator.data.base_growth
    assert bg is not None
    assert isinstance(bg, BaseGrowthData)
    assert bg.num_genotype == 3
    assert bg.good_mask.sum() == 2
    assert orchestrator.priors.growth.base_growth is not None
    assert orchestrator.priors.growth.base_growth.k_ref_loc == pytest.approx(0.021)
    assert "base_growth_k_ref" in init_params
