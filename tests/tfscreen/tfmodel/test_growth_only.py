"""
Growth-only models: growth data with no binding data (roadmap step 1,
planning/analysis-roadmap.md).

Theta is then inferred from growth alone: no binding tensors, no binding
likelihood, no theta_binding_noise component and no binding weight.
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
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.prefit_calibration_cli import run_prefit_calibration
from tfscreen.tfmodel.scripts.summarize_fit_cli import _default_trajectory_genotypes


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")
_LIBRARY_YAML = os.path.join(_SMOKE_DATA, "library-smoke.yaml")


@pytest.fixture(scope="module")
def growth_only():
    return ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=None,
                             batch_size=6)


def _trace(orchestrator, guide=False):
    fn = orchestrator.jax_model_guide if guide else orchestrator.jax_model
    data = orchestrator.get_batch(orchestrator.data,
                                  orchestrator.get_random_idx(batch_key=0))
    return trace(seed(fn, 0)).get_trace(data=data, priors=orchestrator.priors)


# ---------------------------------------------------------------------------
# Orchestrator and model
# ---------------------------------------------------------------------------

def test_no_binding_data(growth_only):
    assert growth_only.binding_df is None
    assert growth_only.binding_tm is None
    assert growth_only.data.binding is None
    assert growth_only.data.num_binding == 0
    assert growth_only.priors.binding is None
    assert "observe_binding" not in growth_only.main_control_kwargs
    assert "theta_binding_noise" not in growth_only.main_control_kwargs


def test_model_and_guide_have_no_binding_sites(growth_only):
    for guide in (False, True):
        sites = _trace(growth_only, guide=guide)
        assert not [k for k in sites if "binding" in k]
    assert "growth_pred" in _trace(growth_only)


def test_growth_only_is_batch_safe(growth_only):
    assert find_orchestrator_batch_dependent_latents(growth_only) == {}
    assert find_orchestrator_batch_order_mismatches(growth_only) == {}


def test_growth_sites_match_joint_model(growth_only):
    """Dropping binding removes the binding sites and nothing else."""
    joint = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                              batch_size=6)
    joint_sites = {k for k in _trace(joint) if "binding" not in k}
    assert set(_trace(growth_only)) == joint_sites


@pytest.mark.parametrize("kwargs,match", [
    ({"binding_weight": 2.0}, "binding_weight"),
    ({"theta_binding_noise": "beta"}, "theta_binding_noise"),
])
def test_binding_settings_refused_without_binding(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=None, **kwargs)


def test_binding_only_needs_binding():
    with pytest.raises(ValueError, match="binding_only=True requires"):
        ModelOrchestrator(None, None, binding_only=True)


def test_growth_needed_unless_binding_only():
    with pytest.raises(ValueError, match="growth_df is required"):
        ModelOrchestrator(None, _BINDING_CSV)


def test_prediction_copy(growth_only):
    new = copy_orchestrator(growth_only, genotypes=["wt"])
    assert new.data.binding is None
    assert new.binding_df is None


# ---------------------------------------------------------------------------
# Configuration round trip
# ---------------------------------------------------------------------------

@pytest.fixture
def growth_only_config(tmp_path):
    out_prefix = str(tmp_path / "go")
    configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                    out_prefix=out_prefix, skip_model_stats=True)
    return f"{out_prefix}_config.yaml", f"{out_prefix}_priors.csv"


def test_configure_writes_no_binding(growth_only_config):
    config_file, priors_file = growth_only_config
    with open(config_file) as fh:
        config = yaml.safe_load(fh)
    assert "binding" not in config["data"]
    assert config["components"]["binding_only"] is False
    assert config["components"]["binding_weight"] is None

    # An absent component is left out of the priors CSV, not written as a
    # "None" row (which came back as NaN).
    priors = pd.read_csv(priors_file)
    assert not priors["parameter"].str.startswith("binding").any()
    assert not (priors["value"].astype(str) == "None").any()


def test_configure_round_trip(growth_only_config):
    config_file, _ = growth_only_config
    orchestrator, _ = read_configuration(config_file)
    assert orchestrator.data.binding is None
    assert orchestrator.priors.binding is None


def test_configure_needs_growth_or_binding():
    with pytest.raises(ValueError, match="At least one of binding_df"):
        configure_model()


def test_prefit_refuses_growth_only(growth_only_config):
    config_file, _ = growth_only_config
    with pytest.raises(ValueError, match="growth-only model"):
        run_prefit_calibration(config_file, seed=0)


# ---------------------------------------------------------------------------
# tfs-summarize-fit's default trajectory subset
# ---------------------------------------------------------------------------

def test_default_trajectory_genotypes(growth_only):
    genos = _default_trajectory_genotypes(growth_only, num_random=3, seed=0)
    assert genos[0] == "wt"
    assert len(genos) == len(set(genos))
    # Same seed, same subset.
    assert genos == _default_trajectory_genotypes(growth_only, num_random=3,
                                                   seed=0)
    library = set(np.asarray(growth_only.growth_tm.tensor_dim_labels[
        growth_only.growth_tm.tensor_dim_names.index("genotype")]).astype(str))
    assert set(genos) <= library
