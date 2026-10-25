"""
Simulator pieces for the joint fit (planning/joint-fit-return.md step 5):
linear growth's occupancy warp, the growth titrant scale s, and per-genotype
binding offsets.
"""

import importlib.util
import os
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.binding_data import apply_genotype_offsets
from tfscreen.simulate.growth.growth_linkage import LinearGrowth, warp_theta
from tfscreen.simulate.growth_parameters_output import generate_growth_parameters_df
from tfscreen.simulate.library_prediction import (growth_titrant_scale_from,
                                                  library_prediction)
from tfscreen.tfmodel.generative.components.growth import linear as fit_linear

_spec = importlib.util.spec_from_file_location(
    "_thermo_binding_cfg",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_thermo_binding.py"))
_tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tb)


# --- warp -------------------------------------------------------------------

@pytest.mark.parametrize("a,x0", [(8.0, 0.5), (1.0, 0.3), (20.0, 0.7)])
def test_warp_endpoints_monotone_and_matches_the_fit(a, x0):
    t = np.linspace(0.0, 1.0, 101)
    f = warp_theta(t, a, x0)
    assert f[0] == pytest.approx(0.0, abs=1e-12)
    assert f[-1] == pytest.approx(1.0)
    assert np.all(np.diff(f) > 0)
    np.testing.assert_allclose(
        f, np.asarray(fit_linear.warp_theta(jnp.asarray(t), jnp.asarray(a),
                                            jnp.asarray(x0))), atol=1e-5)


def test_linear_growth_warp():
    g = LinearGrowth()
    t = np.array([0.0, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(g.predict(t, b=0.01, m=-0.02),
                               0.01 - 0.02 * t)
    np.testing.assert_allclose(g.predict(t, b=0.01, m=-0.02, warp_a=8.0, warp_x0=0.5),
                               0.01 - 0.02 * warp_theta(t, 8.0, 0.5))
    with pytest.raises(ValueError, match="warp_a"):
        g.predict(t, b=0.0, m=1.0, warp_a=0.0)


def test_growth_truth_writes_the_warp_in_fit_names():
    df = generate_growth_parameters_df({
        "kanR+kan": {"model": "linear", "b": 0.0, "m": -0.02,
                     "warp_a": 8.0, "warp_x0": 0.4},
        "kanR-kan": {"b": 0.02, "m": 0.0}}).set_index("condition_rep")
    assert df.loc["kanR+kan", "warp_log_a"] == pytest.approx(np.log(8.0))
    assert df.loc["kanR+kan", "warp_x0"] == pytest.approx(0.4)
    assert np.isnan(df.loc["kanR-kan", "warp_log_a"])


# --- binding offsets ----------------------------------------------------------

def test_offsets_one_per_genotype():
    df = pd.DataFrame({"genotype": ["wt", "wt", "M42I", "M42I", "M42I"],
                       "theta_obs": [0.9, 0.1, 0.5, 0.4, 0.3]})
    out, truth = apply_genotype_offsets(df, 0.2, np.random.default_rng(0))
    off = truth.set_index("genotype")["offset"]
    np.testing.assert_allclose(out["theta_obs"] - df["theta_obs"],
                               off.loc[df["genotype"]].to_numpy())
    same, none = apply_genotype_offsets(df, 0.0, np.random.default_rng(0))
    assert none is None and same is df
    with pytest.raises(ValueError):
        apply_genotype_offsets(df, -1.0, np.random.default_rng(0))


# --- growth titrant scale s ---------------------------------------------------

def test_scale_validation():
    assert growth_titrant_scale_from({}) == 1.0
    assert growth_titrant_scale_from({"growth_titrant_scale": 3}) == 3.0
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            growth_titrant_scale_from({"growth_titrant_scale": bad})


def _hill(params, genotypes, concs):
    p = params.assign(genotype=params["genotype"].astype(str)).set_index("genotype")
    c = np.where(np.asarray(concs) == 0, 1e-20, concs)
    out = []
    for g in genotypes:
        r = p.loc[g]
        occ = 1 / (1 + np.exp(-r["hill_n"] * (np.log(c) - r["log_hill_K"])))
        out.append(r["theta_low"] + (r["theta_high"] - r["theta_low"]) * occ)
    return np.array(out)


def _run(scale, spiked=True):
    cf = _tb._config("hill_geno", "stratified")
    if not spiked:
        del cf["binding_data"]["spiked_binding"]
    if scale is not None:
        cf["growth_titrant_scale"] = scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return library_prediction(cf, return_binding_theta=True)


def _theta_at(gt, conc):
    d = gt[np.isclose(gt["titrant_conc"], conc)]
    return d.assign(genotype=d["genotype"].astype(str)).set_index("genotype")["theta"]


def test_growth_sees_s_times_nominal_and_binding_nominal():
    # s = 10: growth at nominal 0.001 mM is the curve at 0.01 mM, a binding
    # concentration; growth at 0 stays 0.
    library_df, _, gt, params, _, lbt = _run(10.0, spiked=False)
    genos = list(pd.unique(library_df["genotype"].astype(str)))
    g001 = _theta_at(gt, 0.001).loc[genos].to_numpy()
    np.testing.assert_allclose(g001, _hill(params, genos, [0.01])[:, 0], rtol=1e-4,
                               atol=1e-6)
    b001 = _theta_at(lbt, 0.01).loc[genos].to_numpy()
    np.testing.assert_allclose(b001, _hill(params, genos, [0.01])[:, 0], rtol=1e-4,
                               atol=1e-6)
    np.testing.assert_allclose(g001, b001, rtol=1e-4, atol=1e-6)
    # at s = 1 the growth curve at 0.001 is not the one at 0.01
    _, _, gt1, params1, _, _ = _run(None, spiked=False)
    np.testing.assert_allclose(_theta_at(gt1, 0.001).loc[genos].to_numpy(),
                               _hill(params1, genos, [0.001])[:, 0], rtol=1e-4,
                               atol=1e-6)


def test_spiked_stratified_growth_curve_follows_s():
    library_df, _, gt, _, binding_theta_df, _ = _run(10.0, spiked=True)
    b = binding_theta_df[np.isclose(binding_theta_df["titrant_conc"], 0.01)]
    b = b.set_index("genotype")["theta_true"]
    g = _theta_at(gt, 0.001)
    assert len(b) == 2
    for geno in b.index:
        assert g.loc[geno] == pytest.approx(b.loc[geno], rel=1e-4, abs=1e-6)
