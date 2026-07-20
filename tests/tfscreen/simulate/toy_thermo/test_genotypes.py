"""Tests for tfscreen.simulate.toy_thermo.genotypes."""

import numpy as np
import pytest

from tfscreen.simulate.toy_thermo.genotypes import (
    ThermoModel, MutationEffects, STATES,
    enumerate_genotypes, build_titration_df, parse_genotype,
    _energies_from_ks, _ks_from_energies,
)

LN_KS = dict(ln_K_conf=0.5, ln_K_dna=18.0, ln_K_eff=14.0)
GRID = np.logspace(-8, -2, 9)


def _model():
    return ThermoModel(protein_total=1e-6, dna_total=1e-9, **LN_KS)


# --- energy <-> K bookkeeping ------------------------------------------------

def test_energy_k_roundtrip():
    """Energies derived from Ks reproduce those Ks."""
    g = _energies_from_ks(0.5, 18.0, 14.0)
    Kc, Kd, Ke = _ks_from_energies(g)
    assert np.log(Kc) == pytest.approx(0.5)
    assert np.log(Kd) == pytest.approx(18.0)
    assert np.log(Ke) == pytest.approx(14.0)


def test_genotype_ks_wt_matches_inputs():
    """genotype_ks('wt') returns exactly the configured wild-type constants."""
    Kc, Kd, Ke = _model().genotype_ks("wt")
    assert np.log(Kc) == pytest.approx(LN_KS["ln_K_conf"])
    assert np.log(Kd) == pytest.approx(LN_KS["ln_K_dna"])
    assert np.log(Ke) == pytest.approx(LN_KS["ln_K_eff"])


def test_genotype_ks_reflects_mutation():
    """Destabilizing HD lowers K_dna; other constants unchanged."""
    model = _model()
    Kc0, Kd0, Ke0 = model.genotype_ks("wt")
    Kc1, Kd1, Ke1 = model.genotype_ks(ddg={"HD": 2.0})
    assert np.log(Kd1) == pytest.approx(np.log(Kd0) - 2.0)
    assert Kc1 == pytest.approx(Kc0)
    assert Ke1 == pytest.approx(Ke0)


def test_global_energy_shift_is_invariant():
    """Shifting all four state energies equally leaves the observable unchanged."""
    model = _model()
    wt = model.observable(GRID, genotype="wt")
    shift = {s: 3.0 for s in STATES}       # same ddG on every state
    shifted = model.observable(GRID, ddg=shift)
    assert np.allclose(wt, shifted, rtol=1e-9, atol=1e-12)


# --- MutationEffects catalog -------------------------------------------------

def test_parse_genotype():
    assert parse_genotype("wt") == []
    assert parse_genotype("A") == ["A"]
    assert parse_genotype("A/B") == ["A", "B"]


def test_ddg_sums_singles_and_epistasis():
    eff = (MutationEffects()
           .add_mutation("A", HD=2.0, H=1.0)
           .add_mutation("B", HD=-1.0, LE=3.0)
           .add_epistasis("A", "B", HD=0.5))
    d = eff.ddg_for("A/B")
    assert d["HD"] == pytest.approx(2.0 - 1.0 + 0.5)
    assert d["H"] == pytest.approx(1.0)
    assert d["LE"] == pytest.approx(3.0)
    assert d["L"] == pytest.approx(0.0)


def test_single_mutant_has_no_epistasis_term():
    eff = (MutationEffects()
           .add_mutation("A", HD=2.0)
           .add_mutation("B", HD=-1.0)
           .add_epistasis("A", "B", HD=5.0))
    assert eff.ddg_for("A")["HD"] == pytest.approx(2.0)


def test_unknown_state_rejected():
    with pytest.raises(ValueError, match="unknown state"):
        MutationEffects().add_mutation("A", ZZ=1.0)


def test_unknown_mutation_rejected():
    eff = MutationEffects().add_mutation("A", HD=1.0)
    with pytest.raises(KeyError):
        eff.ddg_for("A/B")


def test_self_epistasis_rejected():
    with pytest.raises(ValueError, match="distinct"):
        MutationEffects().add_epistasis("A", "A", HD=1.0)


def test_ddg_and_effects_mutually_exclusive():
    model = _model()
    eff = MutationEffects().add_mutation("A", HD=1.0)
    with pytest.raises(ValueError, match="either"):
        model.observable(GRID, genotype="A", effects=eff, ddg={"HD": 1.0})


# --- observable epistasis emerges from a nonlinear map -----------------------

def test_observable_epistasis_without_thermo_epistasis():
    """Zero in-state epistasis still yields nonzero observable epistasis."""
    model = _model()
    eff = (MutationEffects()
           .add_mutation("A", HD=3.0)
           .add_mutation("B", LE=-3.0)
           .add_epistasis("A", "B"))          # all-zero thermodynamic epistasis
    E = 1e-5
    wt = model.observable(E, genotype="wt", effects=eff)
    a = model.observable(E, genotype="A", effects=eff)
    b = model.observable(E, genotype="B", effects=eff)
    ab = model.observable(E, genotype="A/B", effects=eff)
    obs_epistasis = (ab - a) - (b - wt)
    assert abs(obs_epistasis) > 1e-3


# --- enumeration + DataFrame builder -----------------------------------------

def test_enumerate_genotypes():
    genos = enumerate_genotypes(["A", "B", "C"], order=2)
    assert genos == ["wt", "A", "B", "C", "A/B", "A/C", "B/C"]


def test_build_titration_df_shape_and_columns():
    model = _model()
    eff = (MutationEffects()
           .add_mutation("A", HD=2.0)
           .add_mutation("B", LE=-2.0))
    df = build_titration_df(model, GRID, effects=eff, observable_std=0.02)
    # wt + 2 singles + 1 double = 4 genotypes, each over the grid
    assert set(df["genotype"]) == {"wt", "A", "B", "A/B"}
    assert len(df) == 4 * len(GRID)
    assert list(df.columns) == ["genotype", "titrant_name", "titrant_conc",
                                "observable", "observable_std"]
    assert (df["observable_std"] == 0.02).all()
    assert (df["titrant_name"] == "effector").all()


def test_build_titration_df_no_std_column_by_default():
    model = _model()
    df = build_titration_df(model, GRID, genotypes=["wt"])
    assert "observable_std" not in df.columns
    assert np.all((df["observable"] >= 0) & (df["observable"] <= 1))


def test_build_titration_df_noise_is_reproducible():
    model = _model()
    eff = MutationEffects().add_mutation("A", HD=2.0)
    kw = dict(genotypes=["wt", "A"], noise_sd=0.03)
    a = build_titration_df(model, GRID, effects=eff,
                           rng=np.random.default_rng(0), **kw)
    b = build_titration_df(model, GRID, effects=eff,
                           rng=np.random.default_rng(0), **kw)
    c = build_titration_df(model, GRID, effects=eff,
                           rng=np.random.default_rng(1), **kw)
    assert np.allclose(a["observable"], b["observable"])
    assert not np.allclose(a["observable"], c["observable"])
    assert np.all((a["observable"] >= 0) & (a["observable"] <= 1))


def test_manual_ddg_matches_effects_catalog():
    """Passing ddg= directly equals routing the same deltas through a catalog."""
    model = _model()
    eff = MutationEffects().add_mutation("A", HD=2.5, LE=-1.0)
    via_catalog = model.observable(GRID, genotype="A", effects=eff)
    via_ddg = model.observable(GRID, ddg={"HD": 2.5, "LE": -1.0})
    assert np.allclose(via_catalog, via_ddg)
