"""Tests for tfscreen.tfmodel.priors_edit and the configure flags built on it."""
import os

import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel import priors_edit as P
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.scripts.configure_model_cli import (
    check_spikes_in_data,
    configure_model,
)

_SMOKE_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                           "smoke-tests", "test_data")
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_LIBRARY_YAML = os.path.join(_SMOKE_DATA, "library-smoke.yaml")
_CONDITIONS = ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]


# ---------------------------------------------------------------------------
# names and overrides
# ---------------------------------------------------------------------------

def test_parse_prior_overrides():
    assert P.parse_prior_overrides(["a=1", "b.c = 0.5"]) == {"a": 1.0, "b.c": 0.5}
    assert P.parse_prior_overrides(None) == {}
    for bad, match in ((["a"], "name=value"), (["a=x"], "not a number"),
                       (["a=1", "a=2"], "twice")):
        with pytest.raises(ValueError, match=match):
            P.parse_prior_overrides(bad)


def test_resolve_prior_name():
    names = ["growth.sample_offset.sigma_fixed", "theta.theta_n_fixed",
             "growth.a.scale", "growth.b.scale"]
    assert P.resolve_prior_name("sigma_fixed", names) == names[0]
    assert P.resolve_prior_name("sample_offset.sigma_fixed", names) == names[0]
    assert P.resolve_prior_name("theta.theta_n_fixed", names) == names[1]
    assert P.resolve_prior_name("a.scale", names) == "growth.a.scale"
    with pytest.raises(ValueError, match="ambiguous"):
        P.resolve_prior_name("scale", names)
    with pytest.raises(ValueError, match="No prior named"):
        P.resolve_prior_name("nope", names)
    # a suffix must end at a dot boundary
    with pytest.raises(ValueError, match="No prior named"):
        P.resolve_prior_name("fixed", ["theta.theta_n_fixed2"])


def test_apply_prior_overrides(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame({"parameter": ["x.sigma_fixed", "x.k_loc", "x.k_loc"],
                  "value": [0.0, 1.0, 2.0],
                  "flat_index": [np.nan, 0, 1]}).to_csv(path, index=False)
    out = P.apply_prior_overrides(str(path), {"sigma_fixed": 0.17})
    assert out == {"x.sigma_fixed": 0.17}
    df = pd.read_csv(path)
    assert df.loc[df.parameter == "x.sigma_fixed", "value"].item() == 0.17
    with pytest.raises(ValueError, match="per-condition"):
        P.apply_prior_overrides(str(path), {"k_loc": 3.0})


# ---------------------------------------------------------------------------
# growth priors
# ---------------------------------------------------------------------------

_DEFAULTS = {"k_loc": 0.02, "k_scale": 0.1, "m_loc": 0.0,
             "m_scale_plus": 0.01, "m_scale_minus": 0.001}


def _labels(replicates=True):
    if replicates:
        return pd.DataFrame({"replicate": [1, 1, 2, 2],
                             "condition_rep": ["a", "b", "a", "b"]})
    return pd.DataFrame({"condition_rep": ["a", "b"]})


def test_growth_prior_updates_broadcasts_over_replicates():
    table = pd.DataFrame({"condition_rep": ["a"], "k_loc": [0.03],
                          "m_scale": [0.005]})
    up = P.growth_prior_updates(table, _labels(), _DEFAULTS)
    pre = P.GROWTH_PRIOR_PREFIX
    assert np.allclose(up[pre + "k_loc"], [0.03, 0.02, 0.03, 0.02])
    assert np.allclose(up[pre + "m_scale_plus"], [0.005, 0.01, 0.005, 0.01])
    assert np.allclose(up[pre + "m_scale_minus"], [0.005, 0.001, 0.005, 0.001])
    assert pre + "k_scale" not in up


def test_growth_prior_updates_per_replicate():
    table = pd.DataFrame({"replicate": [2], "condition_rep": ["b"],
                          "k_loc": [0.05]})
    up = P.growth_prior_updates(table, _labels(), _DEFAULTS)
    assert np.allclose(up[P.GROWTH_PRIOR_PREFIX + "k_loc"],
                       [0.02, 0.02, 0.02, 0.05])
    with pytest.raises(ValueError, match="shares conditions"):
        P.growth_prior_updates(table, _labels(replicates=False), _DEFAULTS)


@pytest.mark.parametrize("table,match", [
    (pd.DataFrame({"condition_rep": ["c"], "k_loc": [1.0]}), "does not have"),
    (pd.DataFrame({"condition_rep": ["a"], "q": [1.0]}), "Unknown"),
    (pd.DataFrame({"condition_rep": ["a", "a"], "k_loc": [1.0, 2.0]}), "more than once"),
    (pd.DataFrame({"condition_rep": ["a"], "k_scale": [0.0]}), "> 0"),
    (pd.DataFrame({"k_loc": [1.0]}), "condition_rep"),
])
def test_growth_prior_updates_refuses(table, match):
    with pytest.raises(ValueError, match=match):
        P.growth_prior_updates(table, _labels(), _DEFAULTS)


def test_growth_priors_from_wt_rates():
    rates = pd.DataFrame({
        "condition_sel": ["kanR+kan", "kanR+kan", "kanR-kan", "kanR-kan"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "rate_mean": [0.020, 0.012, 0.025, 0.0249],
        "rate_sd": [0.004, 0.002, 0.0001, 0.0001],
        "num_replicates": [4, 4, 4, 4]})
    t = P.growth_priors_from_wt_rates(rates, (0.0, 1.0), sd_floor=0.002)
    row = t.set_index("condition_rep").loc["kanR+kan"]
    assert row.k_loc == pytest.approx(0.012)
    assert row.m_loc == pytest.approx(0.008)
    assert row.k_scale == pytest.approx(0.002)            # floored
    assert row.m_scale == pytest.approx(np.hypot(0.002, 0.001))
    assert t.set_index("condition_rep").loc["kanR-kan"].m_scale == 0.002
    with pytest.raises(ValueError, match="exactly one row"):
        P.growth_priors_from_wt_rates(rates, (0.0, 0.5))
    with pytest.raises(ValueError, match="missing column"):
        P.growth_priors_from_wt_rates(rates.drop(columns="rate_sd"), (0.0, 1.0))


# ---------------------------------------------------------------------------
# tfs-configure-model
# ---------------------------------------------------------------------------

def _configure(tmp_path, **kwargs):
    out_prefix = str(tmp_path / "cfg")
    configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                    out_prefix=out_prefix, skip_model_stats=True,
                    theta_model="hill_relative",
                    growth_shares_replicates=True, **kwargs)
    return out_prefix


def test_configure_set_priors(tmp_path):
    prefix = _configure(tmp_path, set_priors=[
        "sigma_fixed=0.17", "theta_log_hill_n_hyper_scale_fixed=0.5"])
    o, _ = read_configuration(f"{prefix}_config.yaml")
    assert float(o.priors.growth.sample_offset.sigma_fixed) == 0.17
    assert float(o.priors.theta.theta_log_hill_n_hyper_scale_fixed) == 0.5


def test_configure_set_priors_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="No prior named"):
        _configure(tmp_path, set_priors=["not_a_prior=1"])


def test_configure_growth_priors_and_wt_rates(tmp_path):
    table = tmp_path / "gp.csv"
    pd.DataFrame({"condition_rep": _CONDITIONS,
                  "k_loc": [0.015] * 4, "k_scale": [0.01] * 4,
                  "m_loc": [0.0] * 4, "m_scale": [0.01] * 4}
                 ).to_csv(table, index=False)
    rates = tmp_path / "wt.csv"
    pd.DataFrame({"condition_sel": ["kanR+kan", "kanR+kan"],
                  "titrant_conc": [0.0, 1.0], "rate_mean": [0.02, 0.012],
                  "rate_sd": [0.004, 0.004], "num_replicates": [4, 4]}
                 ).to_csv(rates, index=False)

    prefix = _configure(tmp_path, growth_priors=str(table),
                        growth_priors_wt_rates=str(rates))
    o, _ = read_configuration(f"{prefix}_config.yaml")
    g = o.priors.growth.condition_growth
    order = list(P.condition_rep_labels(o)["condition_rep"])
    k = dict(zip(order, np.asarray(g.k_loc)))
    m = dict(zip(order, np.asarray(g.m_loc)))
    assert k["kanR+kan"] == pytest.approx(0.012)   # from the wt rates
    assert m["kanR+kan"] == pytest.approx(0.008)
    assert k["pheS+4CP"] == pytest.approx(0.015)   # from the table


def test_configure_wt_rates_needs_relative_theta(tmp_path):
    rates = tmp_path / "wt.csv"
    pd.DataFrame({"condition_sel": ["kanR+kan"], "titrant_conc": [0.0],
                  "rate_mean": [0.02], "rate_sd": [0.004],
                  "num_replicates": [4]}).to_csv(rates, index=False)
    with pytest.raises(ValueError, match="hill_relative"):
        configure_model(growth_df=_GROWTH_CSV, library_config=_LIBRARY_YAML,
                        out_prefix=str(tmp_path / "x"), skip_model_stats=True,
                        growth_priors_wt_rates=str(rates))


def test_check_spikes_in_data(tmp_path):
    comp = pd.DataFrame({"genotype": ["wt", "M42I", "A1V"],
                         "in_spiked_origin": [True, True, False]})
    growth = pd.DataFrame({"genotype": ["wt", "A1V"]})
    with pytest.raises(ValueError, match="M42I"):
        check_spikes_in_data(comp, growth)
    check_spikes_in_data(comp, growth, allow_missing=True)
    check_spikes_in_data(comp, pd.DataFrame({"genotype": ["wt", "M42I"]}))


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def _fake_orchestrator(crm):
    from types import SimpleNamespace
    return SimpleNamespace(growth_tm=SimpleNamespace(
        map_groups={"condition_rep": crm}))


def test_condition_rep_labels_edge_cases():
    from types import SimpleNamespace
    # no growth model
    assert P.condition_rep_labels(SimpleNamespace(growth_tm=None)) is None
    # a condition_rep map without any label columns
    crm = pd.DataFrame({"map_condition_rep": [1, 0], "other": ["x", "y"]})
    assert P.condition_rep_labels(_fake_orchestrator(crm)) is None
    # labels come back in map_condition_rep order
    crm = pd.DataFrame({"map_condition_rep": [1, 0],
                        "condition_rep": ["b", "a"], "replicate": [1, 1]})
    out = P.condition_rep_labels(_fake_orchestrator(crm))
    assert list(out.columns) == ["replicate", "condition_rep"]
    assert list(out["condition_rep"]) == ["a", "b"]


def test_apply_prior_overrides_empty_and_unindexed(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame({"parameter": ["x.sigma_fixed", "x.k_loc"],
                  "value": [0.0, 1.0]}).to_csv(path, index=False)
    assert P.apply_prior_overrides(str(path), {}) == {}
    assert P.apply_prior_overrides(str(path), None) == {}
    # a CSV with only scalar rows (no flat_index column)
    out = P.apply_prior_overrides(str(path), {"k_loc": 2.5})
    assert out == {"x.k_loc": 2.5}
    df = pd.read_csv(path)
    assert df.loc[df.parameter == "x.k_loc", "value"].item() == 2.5
    assert df.loc[df.parameter == "x.sigma_fixed", "value"].item() == 0.0


def test_apply_prior_overrides_same_prior_twice(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame({"parameter": ["x.sigma_fixed"],
                  "value": [0.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="set twice"):
        P.apply_prior_overrides(str(path), {"sigma_fixed": 1.0,
                                            "x.sigma_fixed": 2.0})


def test_growth_prior_updates_needs_growth_model():
    table = pd.DataFrame({"condition_rep": ["a"], "k_loc": [0.03]})
    with pytest.raises(ValueError, match="need a growth model"):
        P.growth_prior_updates(table, None, _DEFAULTS)


# ---------------------------------------------------------------------------
# configure_model_cli._edit_priors refusals
# ---------------------------------------------------------------------------

def _edit_orch(growth_tm=True):
    from types import SimpleNamespace
    crm = pd.DataFrame({"map_condition_rep": [0], "condition_rep": ["kanR+kan"]})
    tm = SimpleNamespace(map_groups={"condition_rep": crm}) if growth_tm else None
    return SimpleNamespace(growth_tm=tm,
                           settings={"theta_gauge_conc": (0.0, 1.0)})


def _wt_rates_csv(tmp_path):
    rates = tmp_path / "wt.csv"
    pd.DataFrame({"condition_sel": ["kanR+kan", "kanR+kan"],
                  "titrant_conc": [0.0, 1.0], "rate_mean": [0.02, 0.012],
                  "rate_sd": [0.004, 0.004], "num_replicates": [4, 4]}
                 ).to_csv(rates, index=False)
    return str(rates)


def test_edit_priors_needs_growth_data(tmp_path):
    from tfscreen.tfmodel.scripts.configure_model_cli import _edit_priors
    with pytest.raises(ValueError, match="need growth data"):
        _edit_priors(_edit_orch(growth_tm=False), "unused.csv",
                     growth_priors="unused.csv")


def test_edit_priors_needs_linear_growth(tmp_path):
    from tfscreen.tfmodel.scripts.configure_model_cli import _edit_priors
    with pytest.raises(ValueError, match="'linear' only"):
        _edit_priors(_edit_orch(), "unused.csv",
                     growth_priors_wt_rates=_wt_rates_csv(tmp_path),
                     condition_growth_model="power")


def test_edit_priors_wt_rates_refuse_per_replicate_table(tmp_path):
    from tfscreen.tfmodel.scripts.configure_model_cli import _edit_priors
    table = tmp_path / "gp.csv"
    pd.DataFrame({"replicate": [1], "condition_rep": ["kanR+kan"],
                  "k_loc": [0.015]}).to_csv(table, index=False)
    with pytest.raises(ValueError, match="per-replicate"):
        _edit_priors(_edit_orch(), "unused.csv", growth_priors=str(table),
                     growth_priors_wt_rates=_wt_rates_csv(tmp_path),
                     theta_model="hill_relative")


def test_edit_priors_needs_growth_prior_rows(tmp_path):
    from tfscreen.tfmodel.scripts.configure_model_cli import _edit_priors
    table = tmp_path / "gp.csv"
    pd.DataFrame({"condition_rep": ["kanR+kan"], "k_loc": [0.015]}
                 ).to_csv(table, index=False)
    priors = tmp_path / "priors.csv"
    pd.DataFrame({"parameter": [P.GROWTH_PRIOR_PREFIX + "k_loc"],
                  "value": [0.02]}).to_csv(priors, index=False)
    with pytest.raises(ValueError, match="has no 'growth.condition_growth.k_scale' row"):
        _edit_priors(_edit_orch(), str(priors), growth_priors=str(table))


def test_configure_wt_rates_alone(tmp_path):
    prefix = _configure(tmp_path, growth_priors_wt_rates=_wt_rates_csv(tmp_path))
    o, _ = read_configuration(f"{prefix}_config.yaml")
    g = o.priors.growth.condition_growth
    order = list(P.condition_rep_labels(o)["condition_rep"])
    k = dict(zip(order, np.asarray(g.k_loc)))
    m = dict(zip(order, np.asarray(g.m_loc)))
    assert k["kanR+kan"] == pytest.approx(0.012)
    assert m["kanR+kan"] == pytest.approx(0.008)
    # conditions without wt rates keep the default
    assert k["pheS+4CP"] != pytest.approx(0.012)
