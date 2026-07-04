"""Tests for tfscreen.simulate.base_growth_data.generate_base_growth_df."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.base_growth_data import generate_base_growth_df


def _params_df():
    return pd.DataFrame({
        "genotype": ["wt", "A47V", "K12R"],
        "dk_geno": [0.0, -0.02, -0.05],
    })


def test_basic_construction_default_genotypes():
    """With no 'genotypes' key, only wt is included, at rate == k_ref."""
    cfg = {"k_ref": 0.025}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)

    assert set(result["genotype"]) == {"wt"}
    row = result.iloc[0]
    assert row["rate"] == pytest.approx(0.025)
    assert row["rate_true"] == pytest.approx(0.025)
    assert row["rate_std"] == pytest.approx(0.0)


def test_multiple_genotypes_use_k_ref_plus_dk_geno():
    cfg = {"k_ref": 0.025, "genotypes": ["wt", "A47V", "K12R"]}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)

    result = result.set_index("genotype")
    assert result.loc["wt", "rate_true"] == pytest.approx(0.025)
    assert result.loc["A47V", "rate_true"] == pytest.approx(0.025 - 0.02)
    assert result.loc["K12R", "rate_true"] == pytest.approx(0.025 - 0.05)


def test_wt_force_included_when_omitted_from_genotypes():
    cfg = {"k_ref": 0.025, "genotypes": ["A47V"]}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)
    assert "wt" in set(result["genotype"])
    assert set(result["genotype"]) == {"wt", "A47V"}


def test_wt_case_insensitive_not_duplicated():
    """A user spelling 'WT' explicitly must not create a duplicate wt row."""
    cfg = {"k_ref": 0.025, "genotypes": ["WT", "A47V"]}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)
    assert list(result["genotype"]).count("wt") == 1


def test_rate_override_bypasses_dk_geno():
    cfg = {
        "k_ref": 0.025,
        "genotypes": ["wt", "A47V"],
        "rates": {"A47V": 0.018},
    }
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng).set_index("genotype")
    assert result.loc["A47V", "rate_true"] == pytest.approx(0.018)
    assert result.loc["wt", "rate_true"] == pytest.approx(0.025)


def test_missing_genotype_raises():
    # Q99W is validly-formatted but not one of the genotypes in the library.
    cfg = {"k_ref": 0.025, "genotypes": ["wt", "Q99W"]}
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="not found in the simulated library"):
        generate_base_growth_df(cfg, _params_df(), rng)


def test_missing_genotype_in_rates_but_valid_elsewhere_raises():
    """A genotype that IS in the library but was never listed in
    base_growth_data.genotypes cannot be targeted via rates."""
    cfg = {
        "k_ref": 0.025,
        "genotypes": ["wt"],
        "rates": {"K12R": 0.01},  # valid library genotype, but not requested
    }
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="not in base_growth_data.genotypes"):
        generate_base_growth_df(cfg, _params_df(), rng)


def test_missing_k_ref_raises():
    cfg = {"genotypes": ["wt"]}
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="k_ref"):
        generate_base_growth_df(cfg, _params_df(), rng)


def test_noise_perturbs_rate_and_sets_rate_std():
    cfg = {"k_ref": 0.025, "genotypes": ["wt", "A47V"], "noise": 0.01}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)

    assert (result["rate_std"] == 0.01).all()
    # With noise > 0 and a seeded rng, observed rate should differ from truth.
    assert not np.allclose(result["rate"], result["rate_true"])


def test_zero_noise_rate_equals_rate_true():
    cfg = {"k_ref": 0.025, "genotypes": ["wt", "A47V"]}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)
    assert np.allclose(result["rate"], result["rate_true"])
    assert (result["rate_std"] == 0.0).all()


def test_output_columns():
    cfg = {"k_ref": 0.025}
    rng = np.random.default_rng(0)
    result = generate_base_growth_df(cfg, _params_df(), rng)
    assert set(result.columns) >= {"genotype", "rate", "rate_std", "rate_true"}
