"""Tests for the per-genotype MLE fitter's new reusable helpers."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel.genotype_fit.fit import (
    _hill_theta,
    fit_phenotypes,
    fits_to_results_df,
    predict_theta,
    hill_theta_from_fit,
)

# Frozen calibration: two selective markers with *different* m so dk_geno is
# separable from theta_low; one non-selective outgrowth.
CALIB = pd.DataFrame({
    "condition_rep": ["outgrowth", "kan", "phes"],
    "growth_k":      [0.02,        0.0,   0.0],
    "growth_m":      [0.0,        -0.03, -0.06],
})

TRUTH = {
    "wt":  dict(theta_low=0.90, theta_high=0.08, log_K=np.log(3e-3), n=1.4, dk_geno=0.0),
    "A1V": dict(theta_low=0.70, theta_high=0.20, log_K=np.log(3e-2), n=1.0, dk_geno=0.006),
    "A2V": dict(theta_low=0.60, theta_high=0.30, log_K=np.log(1e-2), n=1.2, dk_geno=-0.004),
}
LN_CFU0 = {1: 10.0, 2: 10.5}
CONCS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
T_PRE = 5.0
T_SELS = [0.0, 2.0, 4.0]
REPS = [1, 2]


def build_growth_df(truth=TRUTH, noise_sd=0.0, seed=0):
    """Synthesize a real-shaped ln_cfu DataFrame from a TRUTH dict."""
    rng = np.random.default_rng(seed)
    k = dict(zip(CALIB["condition_rep"], CALIB["growth_k"]))
    m = dict(zip(CALIB["condition_rep"], CALIB["growth_m"]))
    rows = []
    for geno, p in truth.items():
        for rep in REPS:
            for conc in CONCS:
                theta = _hill_theta(conc, p["theta_low"], p["theta_high"],
                                    p["log_K"], p["n"])
                for cond_sel in ["outgrowth", "kan", "phes"]:
                    rate_pre = k["outgrowth"] + p["dk_geno"] + m["outgrowth"] * theta
                    rate_sel = k[cond_sel] + p["dk_geno"] + m[cond_sel] * theta
                    for t_sel in T_SELS:
                        ln_cfu = (LN_CFU0[rep] + rate_pre * T_PRE
                                  + rate_sel * t_sel
                                  + rng.normal(0.0, noise_sd))
                        rows.append({
                            "genotype": geno, "titrant_name": "iptg",
                            "titrant_conc": conc, "condition_pre": "outgrowth",
                            "condition_sel": cond_sel, "t_pre": T_PRE,
                            "t_sel": t_sel, "replicate": rep,
                            "ln_cfu": ln_cfu, "ln_cfu_std": 0.1})
    return pd.DataFrame(rows)


def test_fits_to_results_df_matches_fit_phenotypes():
    """Rebuilding the table from the fits dict reproduces results_df exactly."""
    df = build_growth_df(noise_sd=0.0)
    results, fits = fit_phenotypes(df, CALIB, dk_geno_prior=None, progress=False)
    rebuilt = fits_to_results_df(fits)
    pd.testing.assert_frame_equal(results, rebuilt)


def test_fits_to_results_df_empty_warns():
    with pytest.warns(UserWarning, match="no fits"):
        out = fits_to_results_df({})
    assert len(out) == 0


def test_predict_theta_shape_and_values():
    df = build_growth_df(noise_sd=0.0)
    _results, fits = fit_phenotypes(df, CALIB, dk_geno_prior=None, progress=False)

    theta_df = predict_theta(fits, df, theta_col="theta_raw")

    concs = np.sort(df["titrant_conc"].unique())
    assert list(theta_df.columns) == [
        "genotype", "titrant_name", "titrant_conc", "theta_raw"]
    # One row per (genotype, titrant_name, titrant_conc).
    assert len(theta_df) == len(TRUTH) * len(concs)
    assert set(theta_df["genotype"]) == set(TRUTH)

    # Values match hill_theta_from_fit for a spot-checked genotype.
    key = ("wt", "iptg")
    got = (theta_df[theta_df["genotype"] == "wt"]
           .sort_values("titrant_conc")["theta_raw"].to_numpy())
    exp = hill_theta_from_fit(fits[key], np.sort(concs))
    assert np.allclose(got, exp)
    # Recovered theta is near the ground-truth Hill (noiseless).
    truth_theta = _hill_theta(np.sort(concs), **{
        "theta_low": TRUTH["wt"]["theta_low"],
        "theta_high": TRUTH["wt"]["theta_high"],
        "log_K": TRUTH["wt"]["log_K"], "n": TRUTH["wt"]["n"]})
    assert np.allclose(got, truth_theta, atol=2e-3)


def test_predict_theta_nonfinite_fit_emits_nan():
    """A genotype whose fit failed (NaN est_t) yields NaN theta, not a drop."""
    df = build_growth_df(noise_sd=0.0)
    _results, fits = fit_phenotypes(df, CALIB, dk_geno_prior=None, progress=False)
    key = ("A1V", "iptg")
    gf = fits[key]
    fits[key] = gf._replace(est_t=np.full_like(gf.est_t, np.nan))

    theta_df = predict_theta(fits, df, theta_col="theta_raw")
    a1v = theta_df[theta_df["genotype"] == "A1V"]
    assert len(a1v) == len(np.unique(df["titrant_conc"]))
    assert a1v["theta_raw"].isna().all()
