"""
The relative-X fit (theta 'hill_relative'; roadmap step 5,
planning/analysis-roadmap.md), wired through the orchestrator, configuration,
prediction and the downstream X-scale checks (C1, C8).
"""
import os

import jax
import numpy as np
import pandas as pd
import pytest
import yaml
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from tfscreen.analysis.scripts.extract_epistasis_cli import extract_epistasis
from tfscreen.tfmodel.analysis.extraction import (
    extract_theta_curves,
    extract_theta_epistasis,
)
from tfscreen.tfmodel.analysis.prediction import copy_orchestrator, predict
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts import summarize_fit_cli


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")
_LIBRARY_YAML = os.path.join(_SMOKE_DATA, "library-smoke.yaml")

_RELATIVE = dict(theta="hill_relative", theta_growth_noise="zero")


def _orch(**kwargs):
    return ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=None,
                             **{**_RELATIVE, **kwargs})


@pytest.fixture(scope="module")
def relative():
    return _orch(batch_size=6)


@pytest.fixture(scope="module")
def prior_draws(relative):
    draws = Predictive(relative.jax_model, num_samples=4)(
        jax.random.PRNGKey(0), data=relative.data, priors=relative.priors)
    return {k: v for k, v in draws.items()
            if not k.endswith("_obs") and not k.endswith("_pred")}


# ---------------------------------------------------------------------------
# gauge and refusals
# ---------------------------------------------------------------------------

def test_default_gauge_is_measured_range(relative):
    assert relative.settings["theta_gauge_conc"] == [0.0, 1.0]
    lc = np.asarray(relative.data.growth.theta_gauge_log_conc)
    assert lc[0] == pytest.approx(np.log(1e-20))
    assert lc[1] == pytest.approx(0.0)


def test_wt_is_pinned_in_the_model(relative):
    d = relative.get_batch(relative.data, relative.get_random_idx(batch_key=0))
    tr = trace(seed(relative.jax_model, 1)).get_trace(data=d,
                                                     priors=relative.priors)
    wt = int(relative.data.growth.wt_indexes[0])
    lo, hi = tr["theta_X_low"]["value"][0, wt], tr["theta_X_high"]["value"][0, wt]
    K, n = tr["theta_log_hill_K"]["value"][0, wt], tr["theta_hill_n"]["value"][0, wt]
    lc = np.asarray(relative.data.growth.theta_gauge_log_conc)
    occ = 1 / (1 + np.exp(-n * (lc - K)))
    assert lo + (hi - lo) * occ == pytest.approx([1.0, 0.0], abs=1e-5)


def test_explicit_gauge():
    o = _orch(theta_gauge_conc=[0.0, 0.1])
    assert o.settings["theta_gauge_conc"] == [0.0, 0.1]
    assert float(o.data.growth.theta_gauge_log_conc[1]) == pytest.approx(np.log(0.1))


@pytest.mark.parametrize("gauge", [[1.0, 0.0], [0.1], [-1.0, 1.0]])
def test_bad_gauge_refused(gauge):
    with pytest.raises(ValueError, match="theta_gauge_conc"):
        _orch(theta_gauge_conc=gauge)


def test_gauge_refused_for_absolute_theta():
    with pytest.raises(ValueError, match="theta_gauge_conc"):
        ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=None,
                          theta_growth_noise="zero", theta_gauge_conc=[0, 1])


@pytest.mark.parametrize("kwargs,match", [
    ({"binding_df": _BINDING_CSV}, "binding"),
    ({"activity": "hierarchical_geno"}, "activity"),
    ({"theta_rescale": "logit"}, "theta_rescale"),
    ({"condition_growth": "power"}, "condition_growth"),
    ({"theta_growth_noise": "logit_normal"}, "theta_growth_noise"),
    ({"transformation": "mixture", "transformation_lambda": (0.3, 0.05)},
     "congression_theta_rule"),
])
def test_absolute_theta_settings_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ModelOrchestrator(growth_df=_GROWTH_CSV,
                          **{"binding_df": None, **_RELATIVE, **kwargs})


def test_mixture_with_max_rule():
    o = _orch(transformation="mixture", transformation_lambda=(0.3, 0.05),
              congression_theta_rule="max")
    d = o.get_batch(o.data, o.get_random_idx(batch_key=0))
    tr = trace(seed(o.jax_model, 0)).get_trace(data=d, priors=o.priors)
    assert np.all(np.isfinite(np.asarray(tr["growth_pred"]["value"])))


def test_requires_wt(tmp_path):
    df = pd.read_csv(_GROWTH_CSV)
    path = tmp_path / "no_wt.csv"
    df[df["genotype"] != "wt"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="wt"):
        ModelOrchestrator(growth_df=str(path), binding_df=None, **_RELATIVE)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_configure_round_trip(tmp_path):
    out_prefix = str(tmp_path / "rel")
    configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                    out_prefix=out_prefix, theta_model="hill_relative",
                    theta_gauge_conc=[0.0, 0.1], skip_model_stats=True)
    with open(f"{out_prefix}_config.yaml") as fh:
        config = yaml.safe_load(fh)
    assert config["components"]["theta"] == "hill_relative"
    assert config["components"]["theta_gauge_conc"] == [0.0, 0.1]
    o, _ = read_configuration(f"{out_prefix}_config.yaml")
    assert o.settings["theta_gauge_conc"] == [0.0, 0.1]


def test_configure_records_default_gauge(tmp_path):
    out_prefix = str(tmp_path / "rel")
    configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                    out_prefix=out_prefix, theta_model="hill_relative",
                    skip_model_stats=True)
    with open(f"{out_prefix}_config.yaml") as fh:
        config = yaml.safe_load(fh)
    assert config["components"]["theta_gauge_conc"] == [0.0, 1.0]
    other = str(tmp_path / "abs")
    configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                    out_prefix=other, skip_model_stats=True)
    with open(f"{other}_config.yaml") as fh:
        assert yaml.safe_load(fh)["components"]["theta_gauge_conc"] is None


# ---------------------------------------------------------------------------
# prediction and extraction
# ---------------------------------------------------------------------------

def test_prediction_subset_keeps_wt_for_the_gauge(relative, prior_draws):
    new = copy_orchestrator(relative, genotypes=["M42I"])
    assert set(new.growth_tm.tensor_dim_labels[-1]) == {"M42I", "wt"}
    out = predict(relative, prior_draws, genotypes=["M42I"], num_samples=None)
    assert set(out["genotype"].astype(str)) == {"M42I"}
    assert np.isfinite(out.filter(like="q0.5").to_numpy()).all()


def test_extract_curves_on_X(relative, prior_draws):
    out = extract_theta_curves(relative, prior_draws, num_samples=None)
    wt = out[out["genotype"] == "wt"].set_index("titrant_conc")
    # every draw pins wt, so every quantile does
    assert wt.loc[0.0].filter(like="q").to_numpy() == pytest.approx(1.0, abs=1e-4)
    assert wt.loc[1.0].filter(like="q").to_numpy() == pytest.approx(0.0, abs=1e-4)


def test_epistasis_on_X_is_additive_only(relative, prior_draws):
    with pytest.raises(ValueError, match="scale='add'"):
        extract_theta_epistasis(relative, prior_draws, scale="logit")
    out = extract_theta_epistasis(relative, prior_draws, scale="add")
    assert "in_regime" not in out.columns


def test_extract_epistasis_cli_refuses_non_additive_X(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"genotype": ["wt", "A1B", "C2D", "A1B/C2D"],
                  "q0.5": [1.0, 0.5, 0.4, 0.1],
                  "theta_scale": "X"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="scale='add'"):
        extract_epistasis(str(path), out_prefix=str(tmp_path / "ep"),
                          scale="logit")
    extract_epistasis(str(path), out_prefix=str(tmp_path / "ep"), scale="add")
    assert pd.read_csv(tmp_path / "ep.csv")["ep_obs"].iloc[0] == \
        pytest.approx((0.1 - 0.4) - (0.5 - 1.0))


# ---------------------------------------------------------------------------
# summarize-fit truth on the X scale
# ---------------------------------------------------------------------------

def _truth_files(tmp_path):
    theta = pd.DataFrame({
        "genotype": ["wt"] * 3 + ["A1B"] * 3,
        "titrant_name": "iptg",
        "titrant_conc": [0.0, 0.01, 1.0] * 2,
        "theta": [0.9, 0.5, 0.1, 0.6, 0.4, 0.2]})
    theta_file = tmp_path / "tfs_sim_genotype_theta.csv"
    theta.to_csv(theta_file, index=False)
    pd.DataFrame({"genotype": ["wt", "A1B"], "activity": [1.0, 0.5]}).to_csv(
        tmp_path / "tfs_sim_parameters.csv", index=False)
    return str(theta_file)


def test_x_gauge_and_ref(tmp_path):
    theta_file = _truth_files(tmp_path)
    config = {"components": {"theta": "hill_relative",
                             "theta_gauge_conc": [0.0, 1.0]}}
    g = summarize_fit_cli._x_gauge(config, theta_file, str(tmp_path))
    assert g["wt"] == {"iptg": (0.9, 0.1)}
    assert g["activity"]["A1B"] == 0.5

    gt = pd.read_csv(theta_file).rename(columns={"theta": "ref"})
    X = summarize_fit_cli._x_scale_ref(gt, g)
    assert X[:3] == pytest.approx([1.0, 0.5, 0.0])
    assert X[3] == pytest.approx((0.5 * 0.6 - 0.1) / 0.8)

    assert summarize_fit_cli._x_gauge(
        {"components": {"theta": "hill_geno"}}, theta_file, str(tmp_path)) is None


def test_x_scale_growth_ref(tmp_path):
    g = {"wt": {"iptg": (0.9, 0.1)}, "activity": {}}
    growth = pd.DataFrame({"condition_rep": ["kan"], "growth_k": [0.02],
                           "growth_m": [-0.01]})
    out = summarize_fit_cli._x_scale_growth_ref(growth, g)
    assert out["growth_k"].iloc[0] == pytest.approx(0.02 - 0.01 * 0.1)
    assert out["growth_m"].iloc[0] == pytest.approx(-0.01 * 0.8)
