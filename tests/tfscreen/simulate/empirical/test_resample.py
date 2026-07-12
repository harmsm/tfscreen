"""Tests for Stage-3 resampling and override injection."""

import dataclasses
import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.empirical.fit_phenotypes import (
    PHENO_PARAMS_TRANSFORMED,
    PHENO_PARAMS_NATURAL,
)
from tfscreen.simulate.empirical.population import PopulationModel
from tfscreen.simulate.empirical.resample import (
    resample_phenotypes,
    build_overrides,
    make_empirical_overrides,
    build_empirical_binding_theta,
    population_mean_params,
    _HILL_COLS,
)
from tfscreen.simulate.binding_params import _hill_theta
from tfscreen.simulate.thermo_to_growth import thermo_to_growth


def _make_model():
    mu = np.array([0.005, 1.2, -1.2, np.log(0.02), np.log(1.3)])
    cov = np.diag([1e-4, 0.2, 0.2, 0.1, 0.05])
    return PopulationModel(
        param_names_t=list(PHENO_PARAMS_TRANSFORMED),
        param_names_natural=list(PHENO_PARAMS_NATURAL),
        mu=mu, cov=cov, n_used=100)


GENOS = ["wt", "A1B", "A1B/C2D", "E3F"]


# ---------------------------------------------------------------------------
# resample_phenotypes
# ---------------------------------------------------------------------------

def test_resample_columns_and_wt_pinned():
    model = _make_model()
    df = resample_phenotypes(model, GENOS, rng=0)

    assert list(df.columns) == ["genotype", "dk_geno"] + _HILL_COLS
    assert len(df) == len(GENOS)
    # wt is first and pinned to dk_geno == 0.
    assert df.iloc[0]["genotype"] == "wt"
    assert df.iloc[0]["dk_geno"] == 0.0
    # wt Hill params default to the population mean.
    mean = population_mean_params(model)
    for c in _HILL_COLS:
        assert df.iloc[0][c] == pytest.approx(mean[c])


def test_resample_theta_in_unit_interval_and_reproducible():
    model = _make_model()
    df1 = resample_phenotypes(model, GENOS, rng=42)
    df2 = resample_phenotypes(model, GENOS, rng=42)

    pd.testing.assert_frame_equal(df1, df2)
    assert (df1["theta_low"] > 0).all() and (df1["theta_low"] < 1).all()
    assert (df1["theta_high"] > 0).all() and (df1["theta_high"] < 1).all()
    assert (df1["hill_n"] > 0).all()


def test_resample_explicit_wt_ref():
    model = _make_model()
    wt_ref = {"theta_low": 0.99, "theta_high": 0.01,
              "log_hill_K": np.log(0.017), "hill_n": 2.0}
    df = resample_phenotypes(model, ["wt", "A1B"], rng=1, wt_ref=wt_ref)
    wt = df[df["genotype"] == "wt"].iloc[0]
    assert wt["theta_low"] == pytest.approx(0.99)
    assert wt["hill_n"] == pytest.approx(2.0)
    assert wt["dk_geno"] == 0.0


def test_resample_no_wt():
    model = _make_model()
    df = resample_phenotypes(model, ["A1B", "C2D"], rng=0)
    assert "wt" not in df["genotype"].values
    assert len(df) == 2


# ---------------------------------------------------------------------------
# build_overrides
# ---------------------------------------------------------------------------

def test_build_overrides_structure():
    model = _make_model()
    df = resample_phenotypes(model, GENOS, rng=3)
    log_conc = np.log(np.array([10.0, 100.0]))

    gc_over, params_over, dk_over = build_overrides(df, log_conc)

    assert set(gc_over) == set(GENOS)
    assert set(params_over) == set(GENOS)
    assert set(dk_over) == set(GENOS)
    # theta arrays have one entry per concentration.
    for g in GENOS:
        assert gc_over[g].shape == (2,)
        assert set(params_over[g]) == set(_HILL_COLS)
    # dk_geno override values equal the resampled dk_geno.
    for _, row in df.iterrows():
        assert dk_over[row["genotype"]] == pytest.approx(row["dk_geno"])
    # theta_gc_override equals the Hill curve of the resampled params.
    a1b = df[df["genotype"] == "A1B"].iloc[0]
    expected = _hill_theta(a1b["theta_low"], a1b["theta_high"],
                           a1b["log_hill_K"], a1b["hill_n"], log_conc)
    assert np.allclose(gc_over["A1B"], expected)


# ---------------------------------------------------------------------------
# build_empirical_binding_theta
# ---------------------------------------------------------------------------

_BINDING_CONCS = [0.0, 0.01, 1.0]
_SPIKED = ["wt", "A1B", "A1B/C2D"]


def _binding_cfg(spiked_block):
    return {"titrant_name": "IPTG", "titrant_conc": _BINDING_CONCS,
            "noise": 0.05, "spiked_binding": spiked_block}


def test_binding_none_when_no_spiked_block():
    model = _make_model()
    pheno = resample_phenotypes(model, _SPIKED, rng=0)
    cfg = {"titrant_name": "IPTG", "titrant_conc": _BINDING_CONCS}
    assert build_empirical_binding_theta(pheno, cfg, _SPIKED,
                                         np.random.default_rng(0)) is None


def test_binding_random_selection():
    model = _make_model()
    pheno = resample_phenotypes(model, _SPIKED, rng=0)
    cfg = _binding_cfg({"choose_by": "random", "num": 2})
    df = build_empirical_binding_theta(pheno, cfg, _SPIKED,
                                       np.random.default_rng(1))

    assert list(df.columns) == ["genotype", "titrant_name", "titrant_conc",
                                "theta_true"]
    chosen = set(df["genotype"].unique())
    assert len(chosen) == 2
    assert chosen <= set(_SPIKED)
    # noise-free: theta_true equals the resampled Hill curve.
    from tfscreen.simulate.binding_params import _hill_theta, _to_log_conc
    g = next(iter(chosen))
    r = pheno.set_index("genotype").loc[g]
    expected = _hill_theta(r["theta_low"], r["theta_high"], r["log_hill_K"],
                           r["hill_n"], _to_log_conc(np.array(_BINDING_CONCS)))
    got = df[df["genotype"] == g].sort_values("titrant_conc")["theta_true"]
    assert np.allclose(got.to_numpy(), expected)


def test_binding_stratified_selection():
    model = _make_model()
    pheno = resample_phenotypes(model, _SPIKED, rng=2)
    cfg = _binding_cfg({"choose_by": "stratified", "num": 2})
    df = build_empirical_binding_theta(pheno, cfg, _SPIKED,
                                       np.random.default_rng(2))
    assert set(df["genotype"].unique()) <= set(_SPIKED)
    assert len(df["genotype"].unique()) == 2


def test_binding_file_selection(tmp_path):
    model = _make_model()
    pheno = resample_phenotypes(model, _SPIKED, rng=3)
    # File lists genotypes (params ignored — theta comes from resampled).
    csv = tmp_path / "spiked.csv"
    csv.write_text(
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "wt,0.9,0.1,-4.0,1.0\n"
        "A1B,0.8,0.2,-4.0,1.0\n")
    cfg = _binding_cfg({"choose_by": str(csv)})
    df = build_empirical_binding_theta(pheno, cfg, _SPIKED,
                                       np.random.default_rng(3))
    assert set(df["genotype"].unique()) == {"wt", "A1B"}


def test_binding_file_rejects_non_spiked(tmp_path):
    model = _make_model()
    pheno = resample_phenotypes(model, ["wt", "A1B", "Z9Y"], rng=4)
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "Z9Y,0.9,0.1,-4.0,1.0\n")
    cfg = _binding_cfg({"choose_by": str(csv)})
    with pytest.raises(ValueError, match="must be spiked"):
        build_empirical_binding_theta(pheno, cfg, ["wt", "A1B"],
                                      np.random.default_rng(4))


# ---------------------------------------------------------------------------
# Integration through thermo_to_growth (exercises dk_geno_override hook)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FakeThetaParam:
    """Minimal ThetaParam-like carrier so _theta_param_to_df extracts columns."""
    theta_low: np.ndarray
    theta_high: np.ndarray
    log_hill_K: np.ndarray
    hill_n: np.ndarray


def test_injection_through_thermo_to_growth(mocker):
    model = _make_model()
    genotypes = ["wt", "A1B", "A1B/C2D"]
    concs = np.array([10.0, 100.0])
    log_conc = np.log(concs)

    pheno_df, gc_over, params_over, dk_over = make_empirical_overrides(
        model, genotypes, log_conc, rng=5)

    # Mock the prior-predictive draw: theta_gc placeholder (overwritten by the
    # override) + a theta_param exposing the four Hill fields (T=1, G=3).
    G = len(genotypes)
    theta_gc = np.full((G, len(concs)), 0.5)
    fake_param = _FakeThetaParam(
        theta_low=np.zeros((1, G)), theta_high=np.zeros((1, G)),
        log_hill_K=np.zeros((1, G)), hill_n=np.zeros((1, G)))
    mocker.patch(
        "tfscreen.simulate.thermo_to_growth.sample_theta_prior",
        return_value=(theta_gc, fake_param))

    sample_df = pd.DataFrame({
        "condition_pre": ["M9", "M9"],
        "condition_sel": ["M9+Ab", "M9+Ab"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": concs,
    })
    growth_params = {"M9": {"m": 0.001, "b": 0.020},
                     "M9+Ab": {"m": -0.010, "b": 0.005}}
    sim_data = mocker.MagicMock()
    sim_data.titrant_conc = concs

    _, genotype_theta_df, parameters_df = thermo_to_growth(
        genotypes=genotypes, sim_data=sim_data, sample_df=sample_df,
        theta_component="mock", theta_rng_key=0, growth_params=growth_params,
        theta_gc_override=gc_over, theta_params_override=params_over,
        dk_geno_override=dk_over)

    pdf = parameters_df.set_index("genotype")
    pref = pheno_df.set_index("genotype")

    # dk_geno_override came through (the new hook), including wt pinned to 0.
    assert pdf.loc["wt", "dk_geno"] == 0.0
    for g in ["A1B", "A1B/C2D"]:
        assert pdf.loc[g, "dk_geno"] == pytest.approx(pref.loc[g, "dk_geno"])

    # theta_params_override wrote the resampled Hill params.
    for g in genotypes:
        for c in _HILL_COLS:
            assert pdf.loc[g, c] == pytest.approx(pref.loc[g, c])

    # genotype_theta_df reflects the injected Hill curve for A1B.
    gtdf = genotype_theta_df[genotype_theta_df["genotype"] == "A1B"]
    gtdf = gtdf.sort_values("titrant_conc")
    assert np.allclose(gtdf["theta"].to_numpy(), gc_over["A1B"])
