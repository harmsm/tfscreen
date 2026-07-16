"""End-to-end tests for the tfs-fit-genotypes CLI entry function."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel.genotype_fit.fit import _hill_theta
from tfscreen.tfmodel.scripts.fit_genotypes_cli import fit_genotypes

CALIB = pd.DataFrame({
    "condition_rep": ["outgrowth", "kan", "phes"],
    "growth_k":      [0.02,        0.0,   0.0],
    "growth_m":      [0.0,        -0.03, -0.06],
})
# wt is spiked; A1V/A2V are bulk (>=2 needed to build a congression background).
TRUTH = {
    "wt":  dict(theta_low=0.90, theta_high=0.08, log_K=np.log(3e-3), n=1.4, dk_geno=0.0),
    "A1V": dict(theta_low=0.70, theta_high=0.20, log_K=np.log(3e-2), n=1.0, dk_geno=0.006),
    "A2V": dict(theta_low=0.60, theta_high=0.30, log_K=np.log(1e-2), n=1.2, dk_geno=-0.004),
}
LN_CFU0 = {1: 10.0, 2: 10.5}
CONCS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _growth_df():
    k = dict(zip(CALIB["condition_rep"], CALIB["growth_k"]))
    m = dict(zip(CALIB["condition_rep"], CALIB["growth_m"]))
    rows = []
    for geno, p in TRUTH.items():
        for rep in (1, 2):
            for conc in CONCS:
                theta = _hill_theta(conc, p["theta_low"], p["theta_high"],
                                    p["log_K"], p["n"])
                for cond_sel in ["outgrowth", "kan", "phes"]:
                    r_pre = k["outgrowth"] + p["dk_geno"] + m["outgrowth"] * theta
                    r_sel = k[cond_sel] + p["dk_geno"] + m[cond_sel] * theta
                    for t_sel in (0.0, 2.0, 4.0):
                        rows.append({
                            "genotype": geno, "titrant_name": "iptg",
                            "titrant_conc": conc, "condition_pre": "outgrowth",
                            "condition_sel": cond_sel, "t_pre": 5.0,
                            "t_sel": t_sel, "replicate": rep,
                            "ln_cfu": LN_CFU0[rep] + r_pre * 5.0 + r_sel * t_sel,
                            "ln_cfu_std": 0.1})
    return pd.DataFrame(rows)


@pytest.fixture
def paths(tmp_path):
    growth = tmp_path / "growth.csv"
    _growth_df().to_csv(growth, index=False)
    calib = tmp_path / "calib.csv"
    CALIB.to_csv(calib, index=False)
    return str(growth), str(calib), tmp_path


def test_raw_only_writes_params_and_theta(paths):
    growth, calib, tmp_path = paths
    prefix = str(tmp_path / "run")
    fit_genotypes(growth, calib, out_prefix=prefix, dk_geno_prior_sd=0.0)

    params = pd.read_csv(f"{prefix}_params.csv")
    theta = pd.read_csv(f"{prefix}_theta.csv")

    assert set(params["genotype"]) == set(TRUTH)
    for col in ("theta_low", "theta_high", "log_hill_K", "hill_n", "dk_geno"):
        assert col in params.columns
    assert list(theta.columns) == [
        "genotype", "titrant_name", "titrant_conc", "theta_raw"]

    # No congression artefacts.
    assert not (tmp_path / "run_params_deattenuated.csv").exists()
    assert not (tmp_path / "run_theta_history.csv").exists()

    # Noiseless -> recovers ground-truth theta_low.
    res = params.set_index("genotype")
    for geno, p in TRUTH.items():
        assert res.loc[geno, "theta_low"] == pytest.approx(p["theta_low"], abs=2e-3)


def test_congression_writes_deattenuated_and_history(paths):
    growth, calib, tmp_path = paths
    spiked = tmp_path / "spiked.txt"
    spiked.write_text("wt\n")
    prefix = str(tmp_path / "run")

    fit_genotypes(growth, calib, out_prefix=prefix, dk_geno_prior_sd=0.0,
                  congression_lambda=1.0, spiked_file=str(spiked),
                  save_theta_history=True)

    deatt = pd.read_csv(f"{prefix}_params_deattenuated.csv")
    theta = pd.read_csv(f"{prefix}_theta.csv")
    history = pd.read_csv(f"{prefix}_theta_history.csv")

    assert set(deatt["genotype"]) == set(TRUTH)
    assert "theta_deattenuated" in theta.columns
    assert "theta_raw" in theta.columns

    # wt is spiked -> its de-attenuated theta equals its raw theta.
    wt = theta[theta["genotype"] == "wt"]
    assert np.allclose(wt["theta_raw"], wt["theta_deattenuated"])

    # Bulk genotypes were corrected: at least one theta_deattenuated differs.
    bulk = theta[theta["genotype"] != "wt"]
    assert not np.allclose(bulk["theta_raw"], bulk["theta_deattenuated"])

    # History covers only the bulk genotypes and iterates.
    assert set(history["genotype"]) == {"A1V", "A2V"}
    assert history["iter"].max() >= 1


def test_congression_without_history_skips_history_file(paths):
    growth, calib, tmp_path = paths
    spiked = tmp_path / "spiked.txt"
    spiked.write_text("wt\n")
    prefix = str(tmp_path / "run")

    fit_genotypes(growth, calib, out_prefix=prefix, dk_geno_prior_sd=0.0,
                  congression_lambda=1.0, spiked_file=str(spiked))

    assert (tmp_path / "run_params_deattenuated.csv").exists()
    assert not (tmp_path / "run_theta_history.csv").exists()
