"""
The read-count growth likelihood (growth_likelihood='counts'; roadmap step 7,
planning/analysis-roadmap.md), wired through the orchestrator.
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from numpyro.handlers import seed, trace

from tfscreen.tfmodel.analysis.prediction import copy_orchestrator
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.batch_safety import (
    find_orchestrator_batch_dependent_latents,
    find_orchestrator_batch_order_mismatches,
)
from tfscreen.tfmodel.model_orchestrator import (
    ModelOrchestrator,
    _add_count_columns,
)
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")
_LIBRARY_YAML = os.path.join(_SMOKE_DATA, "library-smoke.yaml")


def _orch(**kwargs):
    return ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                             batch_size=6, growth_likelihood="counts",
                             **kwargs)


@pytest.fixture(scope="module")
def counts_model():
    return _orch(sample_offset="level")


def _trace(o):
    d = o.get_batch(o.data, o.get_random_idx(batch_key=0))
    return d, trace(seed(o.jax_model, 0)).get_trace(data=d, priors=o.priors)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def test_count_tensors(counts_model):
    g = counts_model.data.growth
    assert g.growth_likelihood == "counts"
    assert g.counts.shape == g.ln_cfu.shape
    assert g.ln_sample_reads.shape == g.ln_cfu.shape
    assert g.sample_ln_cfu.shape == g.ln_cfu.shape
    # Every measured cell's counts match the input file.
    df = pd.read_csv(_GROWTH_CSV)
    good = np.asarray(g.good_mask)
    assert np.sort(np.asarray(g.counts)[good]).tolist() == \
        np.sort(df["counts"].to_numpy(float)).tolist()


def test_lncfu_model_has_no_count_tensors():
    o = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV)
    assert o.data.growth.growth_likelihood == "lncfu"
    assert o.data.growth.counts is None


def test_depth_from_sample_reads_or_derived():
    df = pd.DataFrame({"counts": [3, 7], "adjusted_counts": [4, 8],
                       "frequency": [4 / 1000, 8 / 1000],
                       "sample_cfu": [1e8, 1e8], "sample_cfu_std": [1e6, 1e6]})
    derived = _add_count_columns(df.copy())
    assert np.allclose(derived["ln_sample_reads"], np.log(1000.0))
    assert np.allclose(derived["sample_ln_cfu"], np.log(1e8))
    given = _add_count_columns(df.assign(sample_reads=990).copy())
    assert np.allclose(given["ln_sample_reads"], np.log(990.0))


@pytest.mark.parametrize("drop,match", [("counts", "'counts' column"),
                                        ("frequency", "total reads"),
                                        ("sample_cfu", "total cells")])
def test_missing_count_columns_refused(drop, match):
    df = pd.DataFrame({"counts": [3], "adjusted_counts": [4],
                       "frequency": [0.004], "sample_cfu": [1e8],
                       "sample_cfu_std": [1e6]}).drop(columns=[drop])
    with pytest.raises(ValueError, match=match):
        _add_count_columns(df)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def test_count_observer_sites(counts_model):
    _, tr = _trace(counts_model)
    assert {"growth_phi", "growth_inv_r", "growth_obs"} <= set(tr)
    assert "growth_nu" not in tr
    assert tr["growth_obs"]["fn"].base_dist.__class__.__name__ == \
        "CountNegativeBinomial"
    assert {"sample_offset_sigma", "sample_offset_offset"} <= set(tr)


def test_count_model_is_batch_safe(counts_model):
    assert find_orchestrator_batch_dependent_latents(counts_model) == {}
    assert find_orchestrator_batch_order_mismatches(counts_model) == {}


def test_counts_refuse_growth_noise():
    with pytest.raises(ValueError, match="growth_noise"):
        _orch(growth_noise="normal_kt")


def test_unknown_likelihood_refused():
    with pytest.raises(ValueError, match="growth_likelihood"):
        ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                          growth_likelihood="poisson")


def test_prediction_copy(counts_model):
    new = copy_orchestrator(counts_model, genotypes=["wt"])
    assert new.data.growth.growth_likelihood == "counts"
    assert new.data.growth.counts is not None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_configure_round_trip(tmp_path):
    out_prefix = str(tmp_path / "cl")
    configure_model(binding_df=_BINDING_CSV, growth_df=_GROWTH_CSV,
                    library_config=_LIBRARY_YAML, out_prefix=out_prefix,
                    growth_likelihood="counts", sample_offset_model="level",
                    skip_model_stats=True)
    with open(f"{out_prefix}_config.yaml") as fh:
        config = yaml.safe_load(fh)
    assert config["components"]["growth_likelihood"] == "counts"
    assert config["components"]["sample_offset"] == "level"

    o, _ = read_configuration(f"{out_prefix}_config.yaml")
    assert o.data.growth.growth_likelihood == "counts"
    assert "growth_obs.phi_loc" in pd.read_csv(f"{out_prefix}_priors.csv")[
        "parameter"].str.cat(sep=" ")


# ---------------------------------------------------------------------------
# prediction with per-tube offsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_offset", ["level", "normal"])
def test_predict_on_new_tube_grid(sample_offset):
    """Per-tube offsets are set to zero on a prediction grid whose tubes are
    not the training tubes (they used to fail a reshape)."""
    import jax
    from numpyro.infer import Predictive
    from tfscreen.tfmodel.analysis.prediction import predict

    o = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                          growth_likelihood="counts",
                          sample_offset=sample_offset)
    prior = Predictive(o.jax_model, num_samples=3)(
        jax.random.PRNGKey(0), data=o.data, priors=o.priors)
    posterior = {k: v for k, v in prior.items()
                 if not k.endswith("_obs") and not k.endswith("_pred")}
    out = predict(o, posterior, t_sel=[0.0, 60.0, 120.0, 240.0],
                  num_samples=None)
    assert len(out) > 0
    assert np.isfinite(out.filter(like="q0.5").to_numpy()).all()


def test_configure_defaults_to_counts(tmp_path):
    """tfs-configure-model defaults to the count likelihood with a per-tube
    level offset (count-likelihood study, 2026-09-27); ModelOrchestrator
    keeps lncfu, so a config written before step 7 (no growth_likelihood
    key) still reads back as the model it was."""
    out_prefix = str(tmp_path / "d")
    configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                    out_prefix=out_prefix, skip_model_stats=True)
    with open(f"{out_prefix}_config.yaml") as fh:
        components = yaml.safe_load(fh)["components"]
    assert components["growth_likelihood"] == "counts"
    assert components["sample_offset"] == "level"
    assert components["growth_noise"] == "zero"

    o = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV)
    assert o.settings["growth_likelihood"] == "lncfu"
