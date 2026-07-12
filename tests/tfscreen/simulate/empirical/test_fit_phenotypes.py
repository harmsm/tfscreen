"""Tests for the Stage-1 per-genotype phenotype fitter."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.empirical.fit_phenotypes import (
    _hill_theta,
    fit_phenotypes,
    fit_one_genotype,
    _build_calib_lookup,
    read_calibration,
)

# Frozen calibration: two selective markers with *different* m so that
# dk_geno is separable from theta_low; one non-selective outgrowth.
CALIB = pd.DataFrame({
    "condition_rep": ["outgrowth", "kan", "phes"],
    "growth_k":      [0.02,        0.0,   0.0],
    "growth_m":      [0.0,        -0.03, -0.06],
})
_K_MAP, _M_MAP = _build_calib_lookup(CALIB)

# Ground-truth phenotype for a couple of genotypes.
TRUTH = {
    "wt":    dict(theta_low=0.90, theta_high=0.08, log_K=np.log(3e-3),
                  n=1.4, dk_geno=0.000),
    "A1V":   dict(theta_low=0.70, theta_high=0.20, log_K=np.log(3e-2),
                  n=1.0, dk_geno=0.006),
}
LN_CFU0 = {1: 10.0, 2: 10.5}   # per-replicate starting abundance

CONCS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
T_PRE = 5.0
T_SELS = [0.0, 2.0, 4.0]
REPS = [1, 2]


def _build_growth_df(noise_sd=0.0, seed=0):
    """Synthesize a real-shaped ln_cfu DataFrame from TRUTH."""
    rng = np.random.default_rng(seed)
    rows = []
    for geno, p in TRUTH.items():
        for rep in REPS:
            for conc in CONCS:
                theta = _hill_theta(conc, p["theta_low"], p["theta_high"],
                                    p["log_K"], p["n"])
                for cond_sel in ["outgrowth", "kan", "phes"]:
                    k_pre, m_pre = _K_MAP["outgrowth"], _M_MAP["outgrowth"]
                    k_sel, m_sel = _K_MAP[cond_sel], _M_MAP[cond_sel]
                    rate_pre = k_pre + p["dk_geno"] + m_pre * theta
                    rate_sel = k_sel + p["dk_geno"] + m_sel * theta
                    for t_sel in T_SELS:
                        ln_cfu = (LN_CFU0[rep]
                                  + rate_pre * T_PRE
                                  + rate_sel * t_sel)
                        ln_cfu += rng.normal(0.0, noise_sd)
                        rows.append({
                            "genotype": geno, "titrant_name": "iptg",
                            "titrant_conc": conc,
                            "condition_pre": "outgrowth",
                            "condition_sel": cond_sel,
                            "t_pre": T_PRE, "t_sel": t_sel,
                            "replicate": rep,
                            "ln_cfu": ln_cfu, "ln_cfu_std": 0.1,
                        })
    return pd.DataFrame(rows)


def _priors_calib_df():
    """A prefit priors CSV (long form) carrying the same k/m as CALIB."""
    rows = []
    for i, r in CALIB.reset_index(drop=True).iterrows():
        rows.append({"parameter": "growth.condition_growth.k_loc",
                     "value": r["growth_k"], "flat_index": i,
                     "condition_rep": r["condition_rep"]})
        rows.append({"parameter": "growth.condition_growth.m_loc",
                     "value": r["growth_m"], "flat_index": i,
                     "condition_rep": r["condition_rep"]})
    # An unrelated scalar prior row that must be ignored.
    rows.append({"parameter": "dk_geno_hyper_loc", "value": -3.5,
                 "flat_index": np.nan, "condition_rep": np.nan})
    return pd.DataFrame(rows)


def test_read_calibration_from_priors_csv():
    wide = read_calibration(_priors_calib_df())
    assert list(wide.columns) == ["condition_rep", "growth_k", "growth_m"]
    w = wide.set_index("condition_rep")
    assert w.loc["outgrowth", "growth_k"] == pytest.approx(0.02)
    assert w.loc["kan", "growth_m"] == pytest.approx(-0.03)
    assert w.loc["phes", "growth_m"] == pytest.approx(-0.06)


def test_read_calibration_wide_passthrough():
    wide = read_calibration(CALIB)
    assert list(wide.columns) == ["condition_rep", "growth_k", "growth_m"]
    assert wide.set_index("condition_rep").loc["kan", "growth_m"] == pytest.approx(-0.03)


def test_read_calibration_priors_missing_rows_raises():
    bad = pd.DataFrame({"parameter": ["something_else"], "value": [1.0],
                        "condition_rep": ["kan"]})
    with pytest.raises(ValueError, match="priors CSV has no"):
        read_calibration(bad)


def test_read_calibration_unrecognized_raises():
    with pytest.raises(ValueError, match="calibration must be"):
        read_calibration(pd.DataFrame({"foo": [1]}))


def test_build_calib_lookup_from_priors_matches_wide():
    k_p, m_p = _build_calib_lookup(_priors_calib_df())
    k_w, m_w = _build_calib_lookup(CALIB)
    assert k_p == pytest.approx(k_w)
    assert m_p == pytest.approx(m_w)


def test_fit_phenotypes_accepts_priors_calibration():
    df = _build_growth_df(noise_sd=0.0)
    results, _ = fit_phenotypes(df, _priors_calib_df(), dk_geno_prior=None,
                                progress=False)
    res = results.set_index("genotype")
    for geno, p in TRUTH.items():
        assert res.loc[geno, "theta_low"] == pytest.approx(p["theta_low"],
                                                           abs=1e-3)


def test_fit_phenotypes_accepts_ln_cfu_var_only():
    """Real tfs-process-counts output has ln_cfu_var (not ln_cfu_std)."""
    df = _build_growth_df(noise_sd=0.0)
    df["ln_cfu_var"] = df["ln_cfu_std"] ** 2
    df = df.drop(columns=["ln_cfu_std"])   # mirror the processed CSV shape

    results, _ = fit_phenotypes(df, CALIB, dk_geno_prior=None, progress=False)
    res = results.set_index("genotype")
    assert res.loc["wt", "theta_low"] == pytest.approx(
        TRUTH["wt"]["theta_low"], abs=1e-3)


def test_parallel_matches_serial():
    """num_workers>1 (process pool) must give the same fits as serial."""
    df = _build_growth_df(noise_sd=0.02, seed=5)
    res_serial, _ = fit_phenotypes(df, CALIB, dk_geno_prior=None,
                                   progress=False, num_workers=1)
    res_par, _ = fit_phenotypes(df, CALIB, dk_geno_prior=None,
                                progress=False, num_workers=2)
    pd.testing.assert_frame_equal(
        res_serial.set_index("genotype").sort_index(),
        res_par.set_index("genotype").sort_index())


def test_noiseless_roundtrip_recovers_truth():
    """With no observation noise the fit should recover the true params."""
    df = _build_growth_df(noise_sd=0.0)
    results, fits = fit_phenotypes(df, CALIB, dk_geno_prior=None,
                                   progress=False)

    assert set(results["genotype"]) == set(TRUTH)
    assert results["converged"].all()

    res = results.set_index("genotype")
    for geno, p in TRUTH.items():
        assert res.loc[geno, "theta_low"] == pytest.approx(p["theta_low"],
                                                           abs=1e-3)
        assert res.loc[geno, "theta_high"] == pytest.approx(p["theta_high"],
                                                            abs=1e-3)
        assert res.loc[geno, "log_hill_K"] == pytest.approx(p["log_K"],
                                                            abs=2e-2)
        assert res.loc[geno, "hill_n"] == pytest.approx(p["n"], abs=3e-2)
        assert res.loc[geno, "dk_geno"] == pytest.approx(p["dk_geno"],
                                                        abs=2e-3)


def test_noisy_fit_within_tolerance_and_reports_covariance():
    """Small noise: point estimates near truth, covariance is finite/PSD."""
    df = _build_growth_df(noise_sd=0.02, seed=7)
    results, fits = fit_phenotypes(df, CALIB, progress=False)

    res = results.set_index("genotype")
    for geno, p in TRUTH.items():
        assert res.loc[geno, "theta_low"] == pytest.approx(p["theta_low"],
                                                           abs=0.08)
        assert res.loc[geno, "theta_high"] == pytest.approx(p["theta_high"],
                                                            abs=0.08)

    # Covariance is in transformed space, finite, and positive on the diagonal.
    gf = fits[("wt", "iptg")]
    cov_pheno = gf.cov_t[gf.pheno_slice, gf.pheno_slice]
    assert np.all(np.isfinite(cov_pheno))
    assert np.all(np.diag(cov_pheno) > 0)


def test_theta_bounds_stay_in_unit_interval():
    """The logit parameterization must keep theta_low/high inside (0, 1)."""
    df = _build_growth_df(noise_sd=0.05, seed=3)
    results, _ = fit_phenotypes(df, CALIB, progress=False)
    for col in ("theta_low", "theta_high"):
        assert (results[col] > 0).all()
        assert (results[col] < 1).all()


def test_missing_calibration_condition_fails_fast():
    """A condition absent from the calibration table raises a clear error."""
    df = _build_growth_df(noise_sd=0.0)
    bad_calib = CALIB[CALIB["condition_rep"] != "phes"]
    with pytest.raises(ValueError, match="absent from the calibration"):
        fit_phenotypes(df, bad_calib, progress=False)


def test_dead_rows_are_dropped():
    """NaN ln_cfu / non-positive std rows are excluded from the fit."""
    df = _build_growth_df(noise_sd=0.0)
    n_full = len(df[df["genotype"] == "wt"])
    df.loc[df["genotype"] == "wt", "ln_cfu"] = df.loc[
        df["genotype"] == "wt", "ln_cfu"].copy()
    # Kill five wt observations.
    wt_idx = df.index[df["genotype"] == "wt"][:5]
    df.loc[wt_idx, "ln_cfu"] = np.nan

    gf = fit_one_genotype(
        df[df["genotype"] == "wt"], _K_MAP, _M_MAP,
        intercept_cols=["replicate"], dk_geno_prior=None)
    assert gf.n_obs == n_full - 5
    assert gf.converged
