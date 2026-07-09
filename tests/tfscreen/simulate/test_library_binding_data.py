"""Tests for simulate/library_binding_data.py (in-library binding genotypes)."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.library_binding_data import (
    generate_library_binding_df,
    _theta_matrix,
)
from tfscreen.simulate.binding_params import _to_log_conc


CONCS = [0.0, 0.001, 0.03, 1.0]


@pytest.fixture
def params_df():
    # Six genotypes with distinct Hill curves so maximin has something to span.
    return pd.DataFrame({
        "genotype":   ["wt", "M42I", "H74A", "K84L", "D88A", "A27G"],
        "theta_low":  [0.01, 0.05, 0.10, 0.02, 0.20, 0.03],
        "theta_high": [0.99, 0.90, 0.80, 0.95, 0.70, 0.88],
        "log_hill_K": [-4.0, -3.0, -5.0, -2.0, -4.5, -3.5],
        "hill_n":     [2.0, 1.5, 1.2, 1.0, 2.0, 1.8],
    })


def _all_survive(params_df):
    return pd.DataFrame({"genotype": list(params_df["genotype"])})


def _theta_true(params_df, genotype):
    row = params_df.set_index("genotype").loc[[genotype]]
    return _theta_matrix(row, _to_log_conc(CONCS))[0]


# ---------------------------------------------------------------------------
# stratified / random selection
# ---------------------------------------------------------------------------

def test_stratified_selects_num_nonspiked_nonwt(params_df):
    bdf, man = generate_library_binding_df(
        {"choose_by": "stratified", "num": 3}, "iptg", CONCS, 0.0,
        params_df, _all_survive(params_df), spiked_genotypes=["wt", "M42I"],
        rng=np.random.default_rng(0),
    )
    sel = set(man["genotype"])
    assert len(sel) == 3
    assert "wt" not in sel and "M42I" not in sel        # wt + spiked excluded
    assert sel.issubset({"H74A", "K84L", "D88A", "A27G"})
    assert (man["binding_class"] == "library").all()
    assert list(bdf.columns) == ["genotype", "titrant_name", "titrant_conc",
                                 "theta_obs", "theta_std"]
    assert len(bdf) == 3 * len(CONCS)


def test_random_selects_num_survivors(params_df):
    _, man = generate_library_binding_df(
        {"choose_by": "random", "num": 2}, "iptg", CONCS, 0.0,
        params_df, _all_survive(params_df), spiked_genotypes=["wt"],
        rng=np.random.default_rng(1),
    )
    assert len(man) == 2
    assert "wt" not in set(man["genotype"])


def test_only_survivors_are_eligible(params_df):
    # Only B and C have growth data → a num=3 request cannot be satisfied.
    growth = pd.DataFrame({"genotype": ["wt", "H74A", "K84L"]})
    with pytest.raises(ValueError, match="only 2 eligible"):
        generate_library_binding_df(
            {"choose_by": "stratified", "num": 3}, "iptg", CONCS, 0.0,
            params_df, growth, spiked_genotypes=["wt"],
            rng=np.random.default_rng(0),
        )


def test_num_required_for_stratified(params_df):
    with pytest.raises(ValueError, match="requires 'num'"):
        generate_library_binding_df(
            {"choose_by": "stratified"}, "iptg", CONCS, 0.0,
            params_df, _all_survive(params_df), spiked_genotypes=["wt"],
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# theta values and noise
# ---------------------------------------------------------------------------

def test_theta_matches_hill_params_noise_free(params_df):
    _, man = generate_library_binding_df(
        {"choose_by": "random", "num": 1}, "iptg", CONCS, 0.0,
        params_df, _all_survive(params_df), spiked_genotypes=["wt"],
        rng=np.random.default_rng(3),
    )
    g = man["genotype"].iloc[0]
    bdf, _ = generate_library_binding_df(
        {"choose_by": "random", "num": 1}, "iptg", CONCS, 0.0,
        params_df, _all_survive(params_df), spiked_genotypes=["wt"],
        rng=np.random.default_rng(3),
    )
    expected = _theta_true(params_df, g)
    got = bdf.sort_values("titrant_conc")["theta_obs"].to_numpy()
    np.testing.assert_allclose(np.sort(got), np.sort(expected), rtol=1e-6)
    assert (bdf["theta_std"] == 0.0).all()


def test_noise_sets_theta_std_and_perturbs(params_df):
    bdf, _ = generate_library_binding_df(
        {"choose_by": "random", "num": 2}, "iptg", CONCS, 0.02,
        params_df, _all_survive(params_df), spiked_genotypes=["wt"],
        rng=np.random.default_rng(4),
    )
    assert (bdf["theta_std"] == 0.02).all()
    assert (bdf["theta_obs"] >= 0.0).all() and (bdf["theta_obs"] <= 1.0).all()


# ---------------------------------------------------------------------------
# file path
# ---------------------------------------------------------------------------

def test_file_path_uses_named_genotypes_and_warns_on_nonsurvivor(params_df, tmp_path):
    # File names B, C, D; but D has no growth data → dropped with a warning.
    f = tmp_path / "lib_binding.csv"
    pd.DataFrame({
        "genotype": ["H74A", "K84L", "D88A"],
        "theta_low": [0.1, 0.02, 0.2],
        "theta_high": [0.8, 0.95, 0.7],
        "log_hill_K": [-5.0, -2.0, -4.5],
        "hill_n": [1.2, 1.0, 2.0],
    }).to_csv(f, index=False)

    growth = pd.DataFrame({"genotype": ["wt", "H74A", "K84L"]})   # D88A missing
    with pytest.warns(UserWarning, match="did not survive"):
        bdf, man = generate_library_binding_df(
            {"choose_by": str(f)}, "iptg", CONCS, 0.0,
            params_df, growth, spiked_genotypes=["wt"],
            rng=np.random.default_rng(0),
        )
    assert set(man["genotype"]) == {"H74A", "K84L"}
    assert "D88A" not in set(bdf["genotype"])
