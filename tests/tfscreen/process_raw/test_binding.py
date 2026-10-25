import numpy as np
import pandas as pd
import pytest

from tfscreen.process_raw.binding import (
    BINDING_COLUMNS,
    _round_sig,
    binding_table,
    format_info,
)


def _wells():
    """
    wt measured on two days (two bioreps) at 2 uM, with floating-point
    residue in one day's IPTG; M42I once at 2 uM; wt also at 20 uM; and a
    row with no theta.
    """
    rows = []
    for rep, date, iptg, offset in ((1, "d1", [0.0, 1e-6, 6e-8], 0.0),
                                    (2, "d2", [0.0, 1e-6, 6.000000000000001e-08], 0.1)):
        for c, base in zip(iptg, (0.9, 0.5, 0.8)):
            rows.append(dict(genotype="wt", biorep=rep, date=date,
                             protein_dimer_conc_molar=2e-6,
                             iptg_conc_molar=c, theta=base + offset))
    for c, t in ((0.0, 0.3), (1e-6, 0.2)):
        rows.append(dict(genotype="M42I", biorep=1, date="d3",
                         protein_dimer_conc_molar=2.0000000000000003e-06,
                         iptg_conc_molar=c, theta=t))
    rows.append(dict(genotype="M42I", biorep=2, date="d4",
                     protein_dimer_conc_molar=2e-6, iptg_conc_molar=0.0,
                     theta=np.nan))
    rows.append(dict(genotype="wt", biorep=1, date="d5",
                     protein_dimer_conc_molar=2e-5, iptg_conc_molar=0.0,
                     theta=0.99))
    return pd.DataFrame(rows)


def test_round_sig():
    out = _round_sig([0.0, 6.000000000000001e-08, 6e-08, 123456789.0], 6)
    assert out[0] == 0.0
    assert out[1] == out[2]
    assert out[3] == 123457000.0


def test_binding_table_averages_and_pools():
    df, summary, info = binding_table(_wells(), 2.0)
    assert list(df.columns) == BINDING_COLUMNS
    assert info["rows_in"] == 10
    assert info["rows_no_theta"] == 1
    assert info["rows_kept"] == 8

    wt = df[df["genotype"] == "wt"].set_index("titrant_conc")
    # The rounded 6e-8 M lines up across days: three wt points, in mM.
    assert sorted(wt.index) == [0.0, 6e-05, 0.001]
    assert wt.loc[0.0, "theta_obs"] == pytest.approx(0.95)

    # Every wt pair differs by 0.1: residuals +-0.05, 3 dof.
    assert info["pooled_dof"] == 3
    assert info["pooled_sd"] == pytest.approx(np.sqrt(6 * 0.05 ** 2 / 3))
    s = info["pooled_sd"]
    assert wt.loc[0.0, "theta_std"] == pytest.approx(s / np.sqrt(2))
    m = df[df["genotype"] == "M42I"]
    assert np.allclose(m["theta_std"], s)
    assert (df["titrant_name"] == "iptg").all()

    sm = summary.set_index(["genotype", "titrant_conc"])
    assert sm.loc[("wt", 0.0), "n_replicates"] == 2
    assert np.isnan(sm.loc[("M42I", 0.0), "replicate_sd"])


def test_technical_repeats_count_once():
    w = _wells()
    w = pd.concat([w, w.iloc[[0]].assign(theta=0.7)], ignore_index=True)
    df, summary, _ = binding_table(w, 2.0)
    sm = summary.set_index(["genotype", "titrant_conc"])
    assert sm.loc[("wt", 0.0), "n_replicates"] == 2
    assert sm.loc[("wt", 0.0), "n_measurements"] == 3
    # Day 1 averages its two wells (0.9, 0.7) to 0.8 before day 2's 1.0.
    assert sm.loc[("wt", 0.0), "theta_obs"] == pytest.approx(0.9)


def test_fixed_theta_std_and_units():
    df, _, info = binding_table(_wells(), 2000.0, protein_conc_units="nM",
                                theta_std=0.2)
    assert info["replicate_sd_used"] == 0.2
    wt0 = df[(df["genotype"] == "wt") & (df["titrant_conc"] == 0.0)]
    assert wt0["theta_std"].iloc[0] == pytest.approx(0.2 / np.sqrt(2))
    assert "0.2" in format_info(info)


def test_unreplicated_needs_theta_std():
    with pytest.raises(ValueError, match="give theta_std"):
        binding_table(_wells(), 20.0)
    df, _, _ = binding_table(_wells(), 20.0, theta_std=0.05)
    assert len(df) == 1


def test_refusals():
    with pytest.raises(ValueError, match="no measurements at protein"):
        binding_table(_wells(), 5.0)
    with pytest.raises(ValueError, match="missing column"):
        binding_table(_wells().drop(columns="date"), 2.0)
    with pytest.raises(ValueError, match="not one of"):
        binding_table(_wells(), 2.0, protein_conc_units="pM")
    with pytest.raises(ValueError, match="theta_std must be positive"):
        binding_table(_wells(), 2.0, theta_std=0.0)
    with pytest.raises(ValueError, match="sig_digits must be at least 1"):
        binding_table(_wells(), 2.0, sig_digits=0)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="protein_conc must be positive"):
            binding_table(_wells(), bad)


def test_no_replicate_cols_makes_each_row_a_replicate():
    df, summary, _ = binding_table(_wells(), 2.0, replicate_cols=())
    sm = summary.set_index(["genotype", "titrant_conc"])
    assert sm.loc[("wt", 0.0), "n_replicates"] == 2
