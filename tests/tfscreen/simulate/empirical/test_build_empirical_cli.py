"""End-to-end test for the tfs-build-empirical CLI (Stages 1-2)."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.empirical.fit_phenotypes import _hill_theta
from tfscreen.simulate.empirical.population import PopulationModel
from tfscreen.simulate.scripts.build_empirical_cli import build_empirical

_CALIB = pd.DataFrame({
    "condition_rep": ["outgrowth", "kan", "phes"],
    "growth_k":      [0.02,        0.0,   0.0],
    "growth_m":      [0.0,        -0.03, -0.06],
})
_KM = {r["condition_rep"]: (r["growth_k"], r["growth_m"])
       for _, r in _CALIB.iterrows()}


def _priors_calib_df():
    """Prefit priors CSV (long form) with the same k/m as _CALIB."""
    rows = []
    for i, r in _CALIB.reset_index(drop=True).iterrows():
        rows.append({"parameter": "growth.condition_growth.k_loc",
                     "value": r["growth_k"], "flat_index": i,
                     "condition_rep": r["condition_rep"]})
        rows.append({"parameter": "growth.condition_growth.m_loc",
                     "value": r["growth_m"], "flat_index": i,
                     "condition_rep": r["condition_rep"]})
    return pd.DataFrame(rows)

# 11 genotypes (> D+1 = 6, required by the Stage-2 population fit).
_GENOS = ["wt", "A1V", "L5P", "G7R", "K2E", "D9N",
          "T3S", "V4M", "R6Q", "E8A", "N10D"]
_CONCS = [0.0, 1e-3, 1e-2, 1e-1]
_T_PRE = 5.0
_T_SELS = [0.0, 2.0, 4.0]
_REPS = [1, 2]
_LN_CFU0 = {1: 10.0, 2: 10.4}


def _draw_truth(rng):
    truth = {}
    for g in _GENOS:
        if g == "wt":
            truth[g] = dict(theta_low=0.9, theta_high=0.05,
                            log_K=np.log(0.017), n=1.5, dk=0.0)
        else:
            truth[g] = dict(
                theta_low=rng.uniform(0.6, 0.95),
                theta_high=rng.uniform(0.02, 0.3),
                log_K=rng.normal(np.log(0.02), 0.4),
                n=rng.uniform(0.8, 2.0),
                dk=rng.normal(0.0, 0.005))
    return truth


def _build_growth_df(seed=0):
    rng = np.random.default_rng(seed)
    truth = _draw_truth(rng)
    rows = []
    for g, p in truth.items():
        for rep in _REPS:
            for conc in _CONCS:
                theta = _hill_theta(conc, p["theta_low"], p["theta_high"],
                                    p["log_K"], p["n"])
                for cond_sel in ["outgrowth", "kan", "phes"]:
                    k_pre, m_pre = _KM["outgrowth"]
                    k_sel, m_sel = _KM[cond_sel]
                    r_pre = k_pre + p["dk"] + m_pre * theta
                    r_sel = k_sel + p["dk"] + m_sel * theta
                    for t_sel in _T_SELS:
                        ln_cfu = (_LN_CFU0[rep] + r_pre * _T_PRE
                                  + r_sel * t_sel + rng.normal(0.0, 0.02))
                        rows.append({
                            "genotype": g, "titrant_name": "iptg",
                            "titrant_conc": conc,
                            "condition_pre": "outgrowth",
                            "condition_sel": cond_sel,
                            "t_pre": _T_PRE, "t_sel": t_sel, "replicate": rep,
                            "ln_cfu": ln_cfu, "ln_cfu_std": 0.1})
    return pd.DataFrame(rows)


def test_cli_writes_model_with_wt_ref(tmp_path):
    growth_csv = tmp_path / "growth.csv"
    calib_csv = tmp_path / "calib.csv"
    _build_growth_df().to_csv(growth_csv, index=False)
    # Feed the prefit *priors* CSV directly (long form) — the CLI reshapes it.
    _priors_calib_df().to_csv(calib_csv, index=False)

    out_prefix = str(tmp_path / "emp")
    build_empirical(str(growth_csv), seed=0,
                             calibration_file=str(calib_csv),
                             out_prefix=out_prefix)

    # Both artifacts written (model is a single self-contained JSON).
    assert (tmp_path / "emp_phenotype_model.json").exists()
    assert (tmp_path / "emp_stage1_fits.csv").exists()

    model = PopulationModel.load(out_prefix + "_phenotype_model")
    assert model.n_used >= 6
    # wt_ref embedded from wt's actual Stage-1 fit.
    assert model.wt_ref is not None
    assert set(model.wt_ref) == {"theta_low", "theta_high",
                                 "log_hill_K", "hill_n"}
    # wt fit should be near its true reference phenotype.
    assert model.wt_ref["theta_low"] == pytest.approx(0.9, abs=0.1)

    # Stage-1 fits CSV has a row per genotype.
    fits_df = pd.read_csv(out_prefix + "_stage1_fits.csv")
    assert set(fits_df["genotype"]) == set(_GENOS)


def test_cli_orchestrates_configure_and_prefit(mocker, tmp_path):
    """Without --calibration_file, the CLI runs configure_model + prefit."""
    growth_csv = tmp_path / "growth.csv"
    binding_csv = tmp_path / "binding.csv"
    _build_growth_df().to_csv(growth_csv, index=False)
    binding_csv.write_text(
        "genotype,titrant_name,titrant_conc,theta_obs,theta_std\n")

    out_prefix = str(tmp_path / "emp")
    priors_path = f"{out_prefix}_configure_priors.csv"

    def fake_configure(**kwargs):
        # Stand in for configure_model + prefit having written a calibrated
        # priors CSV at the config's priors_file location.
        _priors_calib_df().to_csv(priors_path, index=False)

    conf = mocker.patch(
        "tfscreen.tfmodel.scripts.configure_model_cli.configure_model",
        side_effect=fake_configure)
    prefit = mocker.patch(
        "tfscreen.tfmodel.scripts.prefit_calibration_cli.run_prefit_calibration")

    spiked_file = tmp_path / "spiked.txt"
    spiked_file.write_text("wt\nA1V\n")

    build_empirical(
        str(growth_csv), binding_file=str(binding_csv), out_prefix=out_prefix,
        spiked_file=str(spiked_file), seed=5)

    # configure_model wired with the experimental inputs and fixed prefix.
    conf.assert_called_once()
    _, ckw = conf.call_args
    assert ckw["binding_df"] == str(binding_csv)
    assert ckw["growth_df"] == str(growth_csv)
    assert ckw["out_prefix"] == f"{out_prefix}_configure"
    assert ckw["spiked"] == ["wt", "A1V"]

    # prefit called on the produced config with the seed.
    prefit.assert_called_once()
    pargs, pkw = prefit.call_args
    assert pargs[0] == f"{out_prefix}_configure_config.yaml"
    assert pkw["seed"] == 5

    # Full pipeline produced the model with wt_ref.
    assert (tmp_path / "emp_phenotype_model.json").exists()
    model = PopulationModel.load(out_prefix + "_phenotype_model")
    assert model.wt_ref is not None


def test_cli_requires_binding_or_calibration(tmp_path):
    growth_csv = tmp_path / "growth.csv"
    _build_growth_df().to_csv(growth_csv, index=False)
    with pytest.raises(ValueError, match="binding data is required"):
        build_empirical(str(growth_csv), seed=0,
                                 out_prefix=str(tmp_path / "emp"))
