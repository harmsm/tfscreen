import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from numpy.random import Generator

from tfscreen.simulate.thermo_to_growth import (
    _assign_activity,
    _assign_dk_geno,
    _apply_growth_params,
    _sample_horseshoe_activity,
    _sample_hierarchical_activity,
    _theta_param_to_df,
    _ACTIVITY_COMPONENTS,
    thermo_to_growth,
    _THETA_RESCALE,
)


# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(42)

@pytest.fixture
def test_genotypes() -> list[str]:
    return ["wt", "A1B", "A1B/C2D"]

@pytest.fixture
def test_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "condition_pre": ["M9", "M9"],
        "condition_sel": ["M9+Ab", "M9+Ab"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [10.0, 100.0],
    })

@pytest.fixture
def simple_growth_params() -> dict:
    return {
        "M9":    {"m": 0.001, "b": 0.020},
        "M9+Ab": {"m": -0.010, "b": 0.005},
    }


# ----------------------------------------------------------------------------
# test _assign_activity
# ----------------------------------------------------------------------------

class TestAssignActivity:

    def test_returns_series(self):
        genotypes = ["wt", "A1B", "C2D"]
        result = _assign_activity(genotypes)
        assert isinstance(result, pd.Series)

    def test_all_genotypes_covered(self):
        genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
        result = _assign_activity(genotypes)
        assert set(result.index) == set(genotypes)

    def test_wt_gets_activity_wt(self):
        genotypes = ["wt", "A1B"]
        result = _assign_activity(genotypes, activity_wt=1.0)
        assert np.isclose(result.loc["wt"], 1.0)

    def test_wt_gets_custom_activity_wt(self):
        genotypes = ["wt", "A1B"]
        result = _assign_activity(genotypes, activity_wt=0.5)
        assert np.isclose(result.loc["wt"], 0.5)

    def test_zero_scale_all_equal_to_wt(self):
        genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
        result = _assign_activity(genotypes, activity_wt=1.0, activity_mut_scale=0.0)
        assert np.allclose(result.values, 1.0)

    def test_positive_scale_mutants_vary(self):
        rng = np.random.default_rng(42)
        genotypes = ["wt", "A1B", "C2D", "E3F", "G4H"]
        result = _assign_activity(genotypes, activity_wt=1.0, activity_mut_scale=0.5, rng=rng)
        mut_values = result.drop("wt").values
        assert not np.allclose(mut_values, 1.0)

    def test_activities_are_positive(self):
        rng = np.random.default_rng(0)
        genotypes = ["wt"] + [f"X{i}Y" for i in range(20)]
        result = _assign_activity(genotypes, activity_wt=1.0, activity_mut_scale=1.0, rng=rng)
        assert np.all(result.values > 0.0)

    def test_reproducible_with_same_rng_seed(self):
        genotypes = ["wt", "A1B", "C2D"]
        r1 = _assign_activity(genotypes, activity_mut_scale=0.3,
                               rng=np.random.default_rng(7))
        r2 = _assign_activity(genotypes, activity_mut_scale=0.3,
                               rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_different_seeds_give_different_values(self):
        genotypes = ["wt", "A1B", "C2D"]
        r1 = _assign_activity(genotypes, activity_mut_scale=0.3,
                               rng=np.random.default_rng(1))
        r2 = _assign_activity(genotypes, activity_mut_scale=0.3,
                               rng=np.random.default_rng(2))
        assert not np.allclose(r1.loc["A1B"], r2.loc["A1B"])


# ----------------------------------------------------------------------------
# test _assign_dk_geno
# ----------------------------------------------------------------------------

def test_assign_dk_geno_sampling_and_integration(rng):
    genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
    result = _assign_dk_geno(genotypes, rng=rng)

    assert isinstance(result, pd.Series)
    assert set(result.index) == set(genotypes)
    # wt is always zero
    assert result.loc["wt"] == 0.0
    # single mutants are each drawn from the shifted lognormal (bounded above by hyper_shift=0.02)
    assert result.loc["A1B"] < 0.02
    assert result.loc["C2D"] < 0.02
    # double mutant is an independent draw, not a sum of singles
    assert result.loc["A1B/C2D"] < 0.02
    assert result.loc["A1B/C2D"] != result.loc["A1B"] + result.loc["C2D"]


def test_assign_dk_geno_reproducible(rng):
    genotypes = ["wt", "A1B", "C2D"]
    r1 = _assign_dk_geno(genotypes, rng=np.random.default_rng(42))
    r2 = _assign_dk_geno(genotypes, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(r1.values, r2.values)


def test_assign_dk_geno_distribution(rng):
    # With many genotypes, the bulk should be negative (deleterious)
    genotypes = [f"M{i}" for i in range(500)]
    result = _assign_dk_geno(genotypes, rng=rng)
    assert (result < 0).mean() > 0.5
    # All values bounded above by hyper_shift default
    assert (result <= 0.02).all()


def test_assign_dk_geno_fixed_value_zero():
    genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
    result = _assign_dk_geno(genotypes, fixed_value=0.0)
    assert isinstance(result, pd.Series)
    assert set(result.index) == set(genotypes)
    np.testing.assert_array_equal(result.values, 0.0)


def test_assign_dk_geno_fixed_value_nonzero():
    genotypes = ["wt", "A1B", "C2D"]
    result = _assign_dk_geno(genotypes, fixed_value=-0.05)
    np.testing.assert_array_equal(result.values, -0.05)


def test_assign_dk_geno_fixed_value_skips_rng():
    """fixed_value must not call the RNG — result is identical regardless of seed."""
    genotypes = ["wt", "A1B", "C2D"]
    r1 = _assign_dk_geno(genotypes, fixed_value=0.0, rng=np.random.default_rng(1))
    r2 = _assign_dk_geno(genotypes, fixed_value=0.0, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(r1.values, r2.values)


# ----------------------------------------------------------------------------
# test _apply_growth_params
# ----------------------------------------------------------------------------

class TestApplyGrowthParams:

    @pytest.fixture
    def growth_params(self):
        return {
            "sel": {"m": -0.01, "b": 0.005},
            "pre": {"m":  0.002, "b": 0.020},
        }

    def test_single_condition_at_theta_zero(self, growth_params):
        k = _apply_growth_params(np.array(["sel"]), np.array([0.0]), growth_params)
        assert np.isclose(k[0], 0.005)

    def test_single_condition_at_theta_one(self, growth_params):
        k = _apply_growth_params(np.array(["sel"]), np.array([1.0]), growth_params)
        assert np.isclose(k[0], -0.01 + 0.005)

    def test_vector_mixed_conditions(self, growth_params):
        conds = np.array(["sel", "pre", "sel"])
        theta = np.array([0.5, 0.3, 0.0])
        k = _apply_growth_params(conds, theta, growth_params)
        expected = np.array([
            -0.01 * 0.5 + 0.005,
             0.002 * 0.3 + 0.020,
            -0.01 * 0.0 + 0.005,
        ])
        np.testing.assert_allclose(k, expected)

    def test_returns_numpy_array(self, growth_params):
        k = _apply_growth_params(np.array(["pre"]), np.array([0.5]), growth_params)
        assert isinstance(k, np.ndarray)

    def test_missing_condition_raises(self, growth_params):
        with pytest.raises(KeyError):
            _apply_growth_params(np.array(["nonexistent"]), np.array([0.5]), growth_params)

    def test_activity_scales_theta_contribution(self, growth_params):
        conds = np.array(["sel", "sel"])
        theta = np.array([1.0, 1.0])
        activity = np.array([0.5, 2.0])
        k = _apply_growth_params(conds, theta, growth_params, activity_array=activity)
        b = growth_params["sel"]["b"]
        m = growth_params["sel"]["m"]
        np.testing.assert_allclose(k, b + activity * m * theta)

    def test_activity_none_defaults_to_one(self, growth_params):
        conds = np.array(["sel"])
        theta = np.array([0.5])
        k_default = _apply_growth_params(conds, theta, growth_params, activity_array=None)
        k_explicit = _apply_growth_params(conds, theta, growth_params,
                                           activity_array=np.array([1.0]))
        np.testing.assert_allclose(k_default, k_explicit)

    def test_zero_activity_returns_baseline(self, growth_params):
        conds = np.array(["sel"])
        theta = np.array([0.7])
        k = _apply_growth_params(conds, theta, growth_params,
                                  activity_array=np.array([0.0]))
        assert np.isclose(k[0], growth_params["sel"]["b"])


# ----------------------------------------------------------------------------
# Helpers shared by thermo_to_growth integration tests
# ----------------------------------------------------------------------------

def _make_sim_data_mock(genotypes, concs):
    """Minimal SimData-like mock for thermo_to_growth."""
    mock = MagicMock()
    mock.titrant_conc = concs
    return mock


def _patch_thermo_deps(mocker, genotypes, sample_df, theta_gc):
    """Patch sample_theta_prior and set_categorical_genotype."""
    mocker.patch(
        "tfscreen.simulate.thermo_to_growth.sample_theta_prior",
        return_value=(theta_gc, MagicMock()),
    )
    mocker.patch(
        "tfscreen.simulate.thermo_to_growth.set_categorical_genotype",
        side_effect=lambda df: df,
    )


# ----------------------------------------------------------------------------
# test thermo_to_growth — missing condition validation
# ----------------------------------------------------------------------------

class TestThermo_to_growth_Validation:

    def test_missing_condition_pre_raises(self, mocker, test_genotypes, test_sample_df):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        growth_params = {"M9+Ab": {"m": -0.01, "b": 0.005}}
        with pytest.raises(ValueError, match="M9"):
            thermo_to_growth(
                genotypes=test_genotypes,
                sim_data=sim_data,
                sample_df=test_sample_df,
                theta_component="mock",
                theta_rng_key=0,
                growth_params=growth_params,
            )

    def test_missing_condition_sel_raises(self, mocker, test_genotypes, test_sample_df):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        growth_params = {"M9": {"m": 0.001, "b": 0.020}}
        with pytest.raises(ValueError, match="M9\\+Ab"):
            thermo_to_growth(
                genotypes=test_genotypes,
                sim_data=sim_data,
                sample_df=test_sample_df,
                theta_component="mock",
                theta_rng_key=0,
                growth_params=growth_params,
            )

    def test_all_conditions_present_does_not_raise(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
        )


# ----------------------------------------------------------------------------
# test thermo_to_growth — integration
# ----------------------------------------------------------------------------

def test_thermo_to_growth_integration(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """End-to-end wiring test with sample_theta_prior mocked."""
    concs = np.array([10.0, 100.0])
    # (G=3, C=2) theta matrix — use distinct values per genotype per conc
    theta_gc = np.array([
        [0.1, 0.9],   # wt
        [0.3, 0.7],   # A1B
        [0.5, 0.5],   # A1B/C2D
    ])
    sim_data = _make_sim_data_mock(test_genotypes, concs)
    _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

    phenotype_df, genotype_theta_df, parameters_df = thermo_to_growth(
        genotypes=test_genotypes,
        sim_data=sim_data,
        sample_df=test_sample_df,
        theta_component="mock",
        theta_rng_key=0,
        growth_params=simple_growth_params,
    )

    # 3 genotypes × 2 conditions = 6 rows
    assert isinstance(phenotype_df, pd.DataFrame)
    assert phenotype_df.shape[0] == 6
    assert "theta" in phenotype_df.columns
    # phenotype_df retains dk_geno and activity for selection_experiment
    assert "dk_geno" in phenotype_df.columns
    assert "activity" in phenotype_df.columns
    assert "k_pre" in phenotype_df.columns
    assert "k_sel" in phenotype_df.columns

    # With default activity_wt=1.0, activity_mut_scale=0.0 → all activity == 1.0
    np.testing.assert_allclose(phenotype_df["activity"].values, 1.0)

    # Verify growth rate formula: k = b + activity * m * theta + dk_geno
    m_pre = simple_growth_params["M9"]["m"]
    b_pre = simple_growth_params["M9"]["b"]
    m_sel = simple_growth_params["M9+Ab"]["m"]
    b_sel = simple_growth_params["M9+Ab"]["b"]
    theta = phenotype_df["theta"].to_numpy()
    dk = phenotype_df["dk_geno"].to_numpy()
    activity = phenotype_df["activity"].to_numpy()

    np.testing.assert_allclose(
        phenotype_df["k_pre"].to_numpy(), b_pre + activity * m_pre * theta + dk
    )
    np.testing.assert_allclose(
        phenotype_df["k_sel"].to_numpy(), b_sel + activity * m_sel * theta + dk
    )

    # genotype_theta_df: long form — one row per (genotype, titrant_name, titrant_conc)
    assert isinstance(genotype_theta_df, pd.DataFrame)
    assert list(genotype_theta_df.columns) == ["genotype", "titrant_name",
                                               "titrant_conc", "theta"]
    # 3 unique genotypes × 2 unique concentrations = 6 rows
    assert len(genotype_theta_df) == 6
    # Each (genotype, titrant_name, titrant_conc) triple is unique
    assert genotype_theta_df.duplicated(
        subset=["genotype", "titrant_name", "titrant_conc"]
    ).sum() == 0

    # parameters_df: one row per unique genotype, dk_geno + activity always present
    assert isinstance(parameters_df, pd.DataFrame)
    assert parameters_df.shape[0] == 3   # 3 unique genotypes
    assert list(parameters_df.columns[:3]) == ["genotype", "dk_geno", "activity"]
    # wt always gets dk_geno == 0
    wt_row = parameters_df[parameters_df["genotype"] == "wt"]
    assert np.isclose(float(wt_row["dk_geno"].iloc[0]), 0.0)
    # With activity_mut_scale=0, all activities == 1.0
    np.testing.assert_allclose(parameters_df["activity"].values, 1.0)


def test_genotype_theta_df_no_duplicates_with_repeated_genotypes(
    mocker, test_sample_df, simple_growth_params
):
    """genotype_theta_df must have one row per unique genotype even when the
    same genotype appears multiple times in the library (e.g. two sub-libraries)."""
    # Supply 5 rows where "wt" and "A1B" each appear twice
    genotypes_with_dups = ["wt", "A1B", "wt", "A1B", "A1B/C2D"]
    concs = np.array([10.0, 100.0])
    # theta_gc has 5 rows (one per library entry); duplicate genotypes get
    # distinct values, but only one row per unique genotype should be kept.
    theta_gc = np.array([
        [0.1, 0.9],   # wt  (first occurrence)
        [0.3, 0.7],   # A1B (first occurrence)
        [0.2, 0.8],   # wt  (second occurrence — should be dropped)
        [0.4, 0.6],   # A1B (second occurrence — should be dropped)
        [0.5, 0.5],   # A1B/C2D
    ])
    sim_data = _make_sim_data_mock(genotypes_with_dups, concs)
    _patch_thermo_deps(mocker, genotypes_with_dups, test_sample_df, theta_gc)

    _, genotype_theta_df, _ = thermo_to_growth(
        genotypes=genotypes_with_dups,
        sim_data=sim_data,
        sample_df=test_sample_df,
        theta_component="mock",
        theta_rng_key=0,
        growth_params=simple_growth_params,
    )

    # 3 unique genotypes × 2 concentrations = 6 rows; no duplicated (geno, conc) pairs
    assert genotype_theta_df["genotype"].nunique() == 3
    assert len(genotype_theta_df) == 6
    assert genotype_theta_df.duplicated(
        subset=["genotype", "titrant_name", "titrant_conc"]
    ).sum() == 0


def test_thermo_to_growth_propagates_rng(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """rng argument is forwarded to _assign_dk_geno."""
    rng = np.random.default_rng(12345)
    concs = np.array([10.0, 100.0])
    theta_gc = np.zeros((3, 2))
    sim_data = _make_sim_data_mock(test_genotypes, concs)
    _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

    mock_assign_dk = mocker.patch(
        "tfscreen.simulate.thermo_to_growth._assign_dk_geno",
        return_value=pd.Series({"wt": 0.0, "A1B": 0.0, "A1B/C2D": 0.0}),
    )

    thermo_to_growth(
        genotypes=test_genotypes,
        sim_data=sim_data,
        sample_df=test_sample_df,
        theta_component="mock",
        theta_rng_key=0,
        growth_params=simple_growth_params,
        rng=rng,
    )

    call_args = mock_assign_dk.call_args
    passed_rng = call_args.args[4] if call_args.args else call_args.kwargs.get("rng")
    assert passed_rng is rng


def test_thermo_to_growth_dk_geno_zero_sets_all_to_zero(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """dk_geno_zero=True must set dk_geno=0 for every genotype."""
    concs = np.array([10.0, 100.0])
    theta_gc = np.full((3, 2), 0.5)
    sim_data = _make_sim_data_mock(test_genotypes, concs)
    _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

    phenotype_df, _, parameters_df = thermo_to_growth(
        genotypes=test_genotypes,
        sim_data=sim_data,
        sample_df=test_sample_df,
        theta_component="mock",
        theta_rng_key=0,
        growth_params=simple_growth_params,
        dk_geno_zero=True,
    )

    np.testing.assert_array_equal(phenotype_df["dk_geno"].values, 0.0)
    np.testing.assert_array_equal(parameters_df["dk_geno"].values, 0.0)


def test_thermo_to_growth_dk_geno_zero_passes_fixed_value(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """dk_geno_zero=True must pass fixed_value=0.0 to _assign_dk_geno."""
    concs = np.array([10.0, 100.0])
    theta_gc = np.zeros((3, 2))
    sim_data = _make_sim_data_mock(test_genotypes, concs)
    _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

    mock_assign_dk = mocker.patch(
        "tfscreen.simulate.thermo_to_growth._assign_dk_geno",
        return_value=pd.Series({"wt": 0.0, "A1B": 0.0, "A1B/C2D": 0.0}),
    )

    thermo_to_growth(
        genotypes=test_genotypes,
        sim_data=sim_data,
        sample_df=test_sample_df,
        theta_component="mock",
        theta_rng_key=0,
        growth_params=simple_growth_params,
        dk_geno_zero=True,
    )

    assert mock_assign_dk.call_args.kwargs.get("fixed_value") == 0.0


def test_thermo_to_growth_dk_geno_zero_false_passes_none(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """dk_geno_zero=False (default) must pass fixed_value=None to _assign_dk_geno."""
    concs = np.array([10.0, 100.0])
    theta_gc = np.zeros((3, 2))
    sim_data = _make_sim_data_mock(test_genotypes, concs)
    _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

    mock_assign_dk = mocker.patch(
        "tfscreen.simulate.thermo_to_growth._assign_dk_geno",
        return_value=pd.Series({"wt": 0.0, "A1B": 0.0, "A1B/C2D": 0.0}),
    )

    thermo_to_growth(
        genotypes=test_genotypes,
        sim_data=sim_data,
        sample_df=test_sample_df,
        theta_component="mock",
        theta_rng_key=0,
        growth_params=simple_growth_params,
        dk_geno_zero=False,
    )

    assert mock_assign_dk.call_args.kwargs.get("fixed_value") is None


# ============================================================================
# test thermo_to_growth — theta_gc_override
# ============================================================================

class TestThetaGcOverride:
    """theta_gc_override must replace theta rows for the named genotypes."""

    def test_overridden_genotype_has_new_theta(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))          # all-zero baseline
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        override_values = np.array([0.8, 0.9])
        _, genotype_theta_df, _ = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_gc_override={"A1B": override_values},
        )

        a1b_rows = genotype_theta_df[genotype_theta_df["genotype"] == "A1B"]
        a1b_rows = a1b_rows.sort_values("titrant_conc").reset_index(drop=True)
        np.testing.assert_allclose(a1b_rows["theta"].values, override_values)

    def test_non_overridden_genotypes_unchanged(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        _, genotype_theta_df, _ = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_gc_override={"A1B": np.array([0.8, 0.9])},
        )

        for geno in ("wt", "A1B/C2D"):
            rows = genotype_theta_df[genotype_theta_df["genotype"] == geno]
            np.testing.assert_array_equal(rows["theta"].values, 0.0)

    def test_none_override_is_noop(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.3)
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        _, genotype_theta_df, _ = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_gc_override=None,
        )

        np.testing.assert_array_equal(genotype_theta_df["theta"].values, 0.3)

    def test_empty_override_is_noop(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.3)
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        _, genotype_theta_df, _ = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_gc_override={},
        )

        np.testing.assert_array_equal(genotype_theta_df["theta"].values, 0.3)

    def test_unknown_genotype_in_override_is_ignored(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = _make_sim_data_mock(test_genotypes, concs)
        _patch_thermo_deps(mocker, test_genotypes, test_sample_df, theta_gc)

        _, genotype_theta_df, _ = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_gc_override={"NOTINGENO": np.array([0.99, 0.99])},
        )

        np.testing.assert_array_equal(genotype_theta_df["theta"].values, 0.0)


# ============================================================================
# test thermo_to_growth — theta_params_override
# ============================================================================

def _patch_thermo_deps_with_param(mocker, theta_gc, theta_param):
    """Like _patch_thermo_deps but accepts a custom theta_param object."""
    mocker.patch(
        "tfscreen.simulate.thermo_to_growth.sample_theta_prior",
        return_value=(theta_gc, theta_param),
    )
    mocker.patch(
        "tfscreen.simulate.thermo_to_growth.set_categorical_genotype",
        side_effect=lambda df: df,
    )


class TestThetaParamsOverride:
    """theta_params_override must update Hill columns in parameters_df."""

    def _make_hill_param(self, n_geno):
        """Return a _MockThetaParam2D with shape (1, n_geno) and known values."""
        rng = np.random.default_rng(5)
        return _MockThetaParam2D(
            theta_low=rng.random((1, n_geno)),
            theta_high=rng.random((1, n_geno)),
            log_hill_K=rng.standard_normal((1, n_geno)),
            hill_n=np.abs(rng.standard_normal((1, n_geno))) + 0.5,
        )

    def test_overridden_genotype_columns_updated(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        theta_param = self._make_hill_param(3)
        _patch_thermo_deps_with_param(mocker, theta_gc, theta_param)
        sim_data = _make_sim_data_mock(test_genotypes, concs)

        override = {"A1B": {"theta_low": 0.99, "theta_high": 0.01,
                            "log_hill_K": -4.0, "hill_n": 2.0}}
        _, _, parameters_df = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_params_override=override,
        )

        row = parameters_df[parameters_df["genotype"] == "A1B"].iloc[0]
        assert np.isclose(row["theta_low"],  0.99)
        assert np.isclose(row["theta_high"], 0.01)
        assert np.isclose(row["log_hill_K"], -4.0)
        assert np.isclose(row["hill_n"],      2.0)

    def test_non_overridden_genotypes_retain_original_values(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        theta_param = self._make_hill_param(3)
        sim_data = _make_sim_data_mock(test_genotypes, concs)

        # First: baseline call with no override to get reference values
        _patch_thermo_deps_with_param(mocker, theta_gc, theta_param)
        _, _, baseline_df = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
        )
        wt_theta_low_baseline = float(
            baseline_df[baseline_df["genotype"] == "wt"]["theta_low"].iloc[0]
        )
        double_theta_low_baseline = float(
            baseline_df[baseline_df["genotype"] == "A1B/C2D"]["theta_low"].iloc[0]
        )

        # Second: override call for A1B only
        _patch_thermo_deps_with_param(mocker, theta_gc, theta_param)
        _, _, parameters_df = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_params_override={"A1B": {"theta_low": 0.99}},
        )

        wt_row = parameters_df[parameters_df["genotype"] == "wt"].iloc[0]
        assert np.isclose(wt_row["theta_low"], wt_theta_low_baseline)
        double_row = parameters_df[parameters_df["genotype"] == "A1B/C2D"].iloc[0]
        assert np.isclose(double_row["theta_low"], double_theta_low_baseline)

    def test_none_override_is_noop(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        theta_param = self._make_hill_param(3)
        sim_data = _make_sim_data_mock(test_genotypes, concs)

        _patch_thermo_deps_with_param(mocker, theta_gc, theta_param)
        _, _, parameters_df_no_override = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            rng=np.random.default_rng(0),
            theta_params_override=None,
        )

        _patch_thermo_deps_with_param(mocker, theta_gc, theta_param)
        _, _, parameters_df_empty = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            rng=np.random.default_rng(0),
            theta_params_override={},
        )

        pd.testing.assert_frame_equal(
            parameters_df_no_override.reset_index(drop=True),
            parameters_df_empty.reset_index(drop=True),
        )

    def test_unknown_column_in_override_is_silently_skipped(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        theta_param = self._make_hill_param(3)
        _patch_thermo_deps_with_param(mocker, theta_gc, theta_param)
        sim_data = _make_sim_data_mock(test_genotypes, concs)

        # "nonexistent_col" is not in parameters_df; must not raise
        _, _, parameters_df = thermo_to_growth(
            genotypes=test_genotypes,
            sim_data=sim_data,
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
            theta_params_override={"A1B": {"theta_low": 0.99,
                                           "nonexistent_col": 42.0}},
        )

        assert "nonexistent_col" not in parameters_df.columns
        row = parameters_df[parameters_df["genotype"] == "A1B"].iloc[0]
        assert np.isclose(row["theta_low"], 0.99)


# ============================================================================
# test _sample_horseshoe_activity
# ============================================================================

class TestSampleHorseshoeActivity:

    def test_returns_series(self):
        result = _sample_horseshoe_activity(["wt", "A1B", "C2D"])
        assert isinstance(result, pd.Series)

    def test_all_genotypes_covered(self):
        genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
        result = _sample_horseshoe_activity(genotypes)
        assert set(result.index) == set(genotypes)

    def test_all_values_positive(self):
        genotypes = ["wt"] + [f"M{i}" for i in range(20)]
        result = _sample_horseshoe_activity(
            genotypes, rng=np.random.default_rng(0)
        )
        assert np.all(result.values > 0.0)

    def test_wt_is_exactly_one(self):
        genotypes = ["wt", "A1B", "C2D"]
        result = _sample_horseshoe_activity(
            genotypes, rng=np.random.default_rng(5)
        )
        assert np.isclose(result.loc["wt"], 1.0)

    def test_reproducible_with_same_seed(self):
        genotypes = ["wt", "A1B", "C2D"]
        r1 = _sample_horseshoe_activity(genotypes, rng=np.random.default_rng(42))
        r2 = _sample_horseshoe_activity(genotypes, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_different_seeds_differ(self):
        genotypes = ["wt", "A1B", "C2D", "D4E"]
        r1 = _sample_horseshoe_activity(genotypes, rng=np.random.default_rng(1))
        r2 = _sample_horseshoe_activity(genotypes, rng=np.random.default_rng(2))
        assert not np.allclose(r1.loc["A1B"], r2.loc["A1B"])

    def test_tiny_tau_shrinks_to_one(self):
        """Very small global_scale_tau_scale → all activities ≈ 1.0."""
        genotypes = ["wt"] + [f"M{i}" for i in range(15)]
        result = _sample_horseshoe_activity(
            genotypes,
            params={"global_scale_tau_scale": 1e-8},
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(result.values, 1.0, atol=1e-5)

    def test_params_override_applied(self):
        """Custom params dict must replace the default hyperparameter."""
        genotypes = ["wt"] + [f"M{i}" for i in range(10)]
        # Run twice with the same seed but different tau_scale — results differ
        r_small = _sample_horseshoe_activity(
            genotypes, params={"global_scale_tau_scale": 0.001},
            rng=np.random.default_rng(99),
        )
        r_large = _sample_horseshoe_activity(
            genotypes, params={"global_scale_tau_scale": 10.0},
            rng=np.random.default_rng(99),
        )
        # Larger tau_scale → larger variance in log(activity)
        assert r_large.std() > r_small.std()


# ============================================================================
# test _sample_hierarchical_activity
# ============================================================================

class TestSampleHierarchicalActivity:

    def test_returns_series(self):
        result = _sample_hierarchical_activity(["wt", "A1B", "C2D"])
        assert isinstance(result, pd.Series)

    def test_all_genotypes_covered(self):
        genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
        result = _sample_hierarchical_activity(genotypes)
        assert set(result.index) == set(genotypes)

    def test_all_values_positive(self):
        genotypes = ["wt"] + [f"M{i}" for i in range(20)]
        result = _sample_hierarchical_activity(
            genotypes, rng=np.random.default_rng(0)
        )
        assert np.all(result.values > 0.0)

    def test_wt_is_exactly_one(self):
        genotypes = ["wt", "A1B", "C2D"]
        result = _sample_hierarchical_activity(
            genotypes, rng=np.random.default_rng(3)
        )
        assert np.isclose(result.loc["wt"], 1.0)

    def test_reproducible_with_same_seed(self):
        genotypes = ["wt", "A1B", "C2D"]
        r1 = _sample_hierarchical_activity(
            genotypes, rng=np.random.default_rng(42)
        )
        r2 = _sample_hierarchical_activity(
            genotypes, rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_different_seeds_differ(self):
        genotypes = ["wt", "A1B", "C2D", "D4E"]
        r1 = _sample_hierarchical_activity(
            genotypes, rng=np.random.default_rng(1)
        )
        r2 = _sample_hierarchical_activity(
            genotypes, rng=np.random.default_rng(2)
        )
        assert not np.allclose(r1.loc["A1B"], r2.loc["A1B"])

    def test_tiny_scale_shrinks_to_shared_mean(self):
        """Very small hyper_scale_loc → all mutant activities ≈ exp(hyper_loc)."""
        genotypes = ["wt"] + [f"M{i}" for i in range(15)]
        # hyper_loc_loc=0, so exp(hyper_loc) ≈ 1.0 when hyper_loc_scale is also tiny
        result = _sample_hierarchical_activity(
            genotypes,
            params={"hyper_loc_loc": 0.0, "hyper_loc_scale": 1e-8,
                    "hyper_scale_loc": 1e-8},
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(result.values, 1.0, atol=1e-5)

    def test_params_override_applied(self):
        """Larger hyper_scale_loc → more spread in mutant activities."""
        genotypes = ["wt"] + [f"M{i}" for i in range(20)]
        r_tight = _sample_hierarchical_activity(
            genotypes,
            params={"hyper_loc_loc": 0.0, "hyper_loc_scale": 0.001,
                    "hyper_scale_loc": 0.001},
            rng=np.random.default_rng(7),
        )
        r_wide = _sample_hierarchical_activity(
            genotypes,
            params={"hyper_loc_loc": 0.0, "hyper_loc_scale": 0.001,
                    "hyper_scale_loc": 2.0},
            rng=np.random.default_rng(7),
        )
        assert r_wide.std() > r_tight.std()


# ============================================================================
# test _ACTIVITY_COMPONENTS registry and thermo_to_growth routing
# ============================================================================

class TestThermo_to_growth_ActivityComponent:
    """
    Verify that activity_component routes correctly to the numpy samplers
    and that unknown names raise ValueError.
    """

    @pytest.fixture
    def base_call_kwargs(self, test_sample_df, simple_growth_params):
        return dict(
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
        )

    def _setup(self, mocker, genotypes, concs, theta_gc, sample_df):
        sim_data = _make_sim_data_mock(genotypes, concs)
        _patch_thermo_deps(mocker, genotypes, sample_df, theta_gc)
        return sim_data

    # -- registry set ---------------------------------------------------------

    def test_activity_components_contains_fixed(self):
        assert "fixed" in _ACTIVITY_COMPONENTS

    def test_activity_components_contains_horseshoe(self):
        assert "horseshoe_geno" in _ACTIVITY_COMPONENTS

    def test_activity_components_contains_hierarchical(self):
        assert "hierarchical_geno" in _ACTIVITY_COMPONENTS

    # -- unknown component raises ---------------------------------------------

    def test_unknown_component_raises(self, mocker, test_genotypes, base_call_kwargs):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        with pytest.raises(ValueError, match="activity_component"):
            thermo_to_growth(
                genotypes=test_genotypes,
                sim_data=sim_data,
                activity_component="horseshoe_mut",
                **base_call_kwargs,
            )

    # -- fixed path -----------------------------------------------------------

    def test_fixed_calls_assign_activity(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        mock_assign = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._assign_activity",
            return_value=pd.Series(1.0, index=["A1B/C2D", "A1B", "wt"]),
        )
        thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data, **base_call_kwargs
        )
        mock_assign.assert_called_once()

    def test_fixed_does_not_call_horseshoe_or_hierarchical(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        mock_hs = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_horseshoe_activity"
        )
        mock_hi = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_hierarchical_activity"
        )
        mocker.patch(
            "tfscreen.simulate.thermo_to_growth._assign_activity",
            return_value=pd.Series(1.0, index=["A1B/C2D", "A1B", "wt"]),
        )
        thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data,
            activity_component="fixed", **base_call_kwargs
        )
        mock_hs.assert_not_called()
        mock_hi.assert_not_called()

    # -- horseshoe path -------------------------------------------------------

    def test_horseshoe_calls_sample_horseshoe_activity(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        mock_hs = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_horseshoe_activity",
            return_value=pd.Series(1.0, index=["A1B/C2D", "A1B", "wt"]),
        )
        thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data,
            activity_component="horseshoe_geno", **base_call_kwargs,
        )
        mock_hs.assert_called_once()

    def test_horseshoe_does_not_call_assign_activity(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_horseshoe_activity",
            return_value=pd.Series(1.0, index=["A1B/C2D", "A1B", "wt"]),
        )
        mock_assign = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._assign_activity"
        )
        thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data,
            activity_component="horseshoe_geno", **base_call_kwargs,
        )
        mock_assign.assert_not_called()

    def test_horseshoe_priors_overrides_forwarded(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        """activity_priors_overrides must be forwarded as params=."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        mock_hs = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_horseshoe_activity",
            return_value=pd.Series(1.0, index=["A1B/C2D", "A1B", "wt"]),
        )
        overrides = {"global_scale_tau_scale": 0.01}
        thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data,
            activity_component="horseshoe_geno",
            activity_priors_overrides=overrides,
            **base_call_kwargs,
        )
        assert mock_hs.call_args.kwargs["params"] == overrides

    # -- hierarchical path ----------------------------------------------------

    def test_hierarchical_calls_sample_hierarchical_activity(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.zeros((3, 2))
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        mock_hi = mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_hierarchical_activity",
            return_value=pd.Series(1.0, index=["A1B/C2D", "A1B", "wt"]),
        )
        thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data,
            activity_component="hierarchical_geno", **base_call_kwargs,
        )
        mock_hi.assert_called_once()

    # -- values propagate into phenotype_df -----------------------------------

    def test_horseshoe_activities_appear_in_phenotype_df(
        self, mocker, test_genotypes, base_call_kwargs
    ):
        """Values returned by _sample_horseshoe_activity appear in phenotype_df."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        sim_data = self._setup(mocker, test_genotypes, concs, theta_gc,
                               base_call_kwargs["sample_df"])
        # test_genotypes sorted: A1B, A1B/C2D, wt
        activity_map = {"wt": 1.0, "A1B": 2.0, "A1B/C2D": 0.5}
        mocker.patch(
            "tfscreen.simulate.thermo_to_growth._sample_horseshoe_activity",
            return_value=pd.Series(activity_map),
        )
        phenotype_df, _, parameters_df = thermo_to_growth(
            genotypes=test_genotypes, sim_data=sim_data,
            activity_component="horseshoe_geno", **base_call_kwargs,
        )
        for geno, expected in activity_map.items():
            rows = phenotype_df[phenotype_df["genotype"] == geno]
            np.testing.assert_allclose(rows["activity"].values, expected)
        # Same activity values must appear in parameters_df (one row per genotype)
        for geno, expected in activity_map.items():
            row = parameters_df[parameters_df["genotype"] == geno]
            np.testing.assert_allclose(float(row["activity"].iloc[0]), expected)


# ============================================================================
# test thermo_to_growth — theta_noise_sigma_logit
# ============================================================================

class TestThermo_to_growth_ThetaNoise:
    """
    Verify logit-normal theta noise applied via theta_noise_sigma_logit.
    """

    @pytest.fixture
    def base_kwargs(self, test_sample_df, simple_growth_params):
        return dict(
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
        )

    def _run(self, mocker, genotypes, concs, theta_gc, **kwargs):
        sim_data = _make_sim_data_mock(genotypes, concs)
        _patch_thermo_deps(mocker, genotypes, None, theta_gc)
        return thermo_to_growth(genotypes=genotypes, sim_data=sim_data, **kwargs)

    def test_zero_sigma_leaves_theta_unchanged(
        self, mocker, test_genotypes, base_kwargs
    ):
        """theta_noise_sigma_logit=0 must not alter theta values."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])
        phenotype_df, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=0.0,
            rng=np.random.default_rng(0),
            **base_kwargs,
        )
        expected_thetas = sorted([0.2, 0.8, 0.3, 0.7, 0.5, 0.5])
        actual_thetas = sorted(phenotype_df["theta"].tolist())
        np.testing.assert_allclose(actual_thetas, expected_thetas, atol=1e-8)

    def test_nonzero_sigma_perturbs_theta(
        self, mocker, test_genotypes, base_kwargs
    ):
        """theta_noise_sigma_logit > 0 must produce theta values different from input."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        phenotype_df, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=1.0,
            rng=np.random.default_rng(42),
            **base_kwargs,
        )
        assert not np.allclose(phenotype_df["theta"].values, 0.5)

    def test_noisy_theta_stays_in_unit_interval(
        self, mocker, test_genotypes, base_kwargs
    ):
        """Even with large sigma_logit, theta must remain in (0, 1)."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.array([[0.01, 0.99], [0.5, 0.5], [0.3, 0.7]])
        phenotype_df, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=5.0,
            rng=np.random.default_rng(0),
            **base_kwargs,
        )
        theta_vals = phenotype_df["theta"].values
        assert np.all(theta_vals > 0.0)
        assert np.all(theta_vals < 1.0)

    def test_reproducible_with_same_rng(
        self, mocker, test_genotypes, base_kwargs
    ):
        """Same rng seed must produce identical noisy theta values."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)

        df1, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=0.5,
            rng=np.random.default_rng(7),
            **base_kwargs,
        )
        df2, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=0.5,
            rng=np.random.default_rng(7),
            **base_kwargs,
        )
        np.testing.assert_array_equal(df1["theta"].values, df2["theta"].values)

    def test_different_seeds_give_different_theta(
        self, mocker, test_genotypes, base_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)

        df1, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=0.5,
            rng=np.random.default_rng(1),
            **base_kwargs,
        )
        df2, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_noise_sigma_logit=0.5,
            rng=np.random.default_rng(2),
            **base_kwargs,
        )
        assert not np.allclose(df1["theta"].values, df2["theta"].values)


# ============================================================================
# test thermo_to_growth — theta_rescale
# ============================================================================

class TestThermo_to_growth_ThetaRescale:
    """
    Verify that the theta_rescale parameter transforms theta before it reaches
    the growth model while leaving the stored 'theta' column and
    genotype_theta_df unchanged.
    """

    @pytest.fixture
    def base_kwargs(self, test_sample_df, simple_growth_params):
        return dict(
            sample_df=test_sample_df,
            theta_component="mock",
            theta_rng_key=0,
            growth_params=simple_growth_params,
        )

    def _run(self, mocker, genotypes, concs, theta_gc, **kwargs):
        sim_data = _make_sim_data_mock(genotypes, concs)
        _patch_thermo_deps(mocker, genotypes, None, theta_gc)
        return thermo_to_growth(genotypes=genotypes, sim_data=sim_data, **kwargs)

    # -- module-level dict ----------------------------------------------------

    def test_rescale_dict_has_passthrough_and_logit(self):
        assert "passthrough" in _THETA_RESCALE
        assert "logit" in _THETA_RESCALE

    def test_passthrough_is_identity(self):
        x = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_equal(_THETA_RESCALE["passthrough"](x), x)

    def test_logit_matches_manual_formula(self):
        x = np.array([0.1, 0.5, 0.9])
        eps = 1e-6
        expected = np.log(np.clip(x, eps, 1-eps) / (1.0 - np.clip(x, eps, 1-eps)))
        np.testing.assert_allclose(_THETA_RESCALE["logit"](x), expected)

    def test_logit_clips_boundary_values(self):
        x = np.array([0.0, 1.0])
        result = _THETA_RESCALE["logit"](x)
        assert np.all(np.isfinite(result))

    # -- unknown name raises --------------------------------------------------

    def test_unknown_theta_rescale_raises(
        self, mocker, test_genotypes, base_kwargs
    ):
        concs = np.array([10.0, 100.0])
        theta_gc = np.full((3, 2), 0.5)
        with pytest.raises(ValueError, match="theta_rescale"):
            self._run(
                mocker, test_genotypes, concs, theta_gc,
                theta_rescale="nonexistent",
                **base_kwargs,
            )

    # -- passthrough leaves theta column and k unchanged ----------------------

    def test_passthrough_matches_default(
        self, mocker, test_genotypes, base_kwargs
    ):
        """Explicit passthrough must produce identical results to the default."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])

        df_default, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            rng=np.random.default_rng(42),
            **base_kwargs,
        )
        df_passthrough, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_rescale="passthrough",
            rng=np.random.default_rng(42),
            **base_kwargs,
        )
        np.testing.assert_array_equal(
            df_default["k_pre"].values, df_passthrough["k_pre"].values
        )
        np.testing.assert_array_equal(
            df_default["theta"].values, df_passthrough["theta"].values
        )

    # -- logit rescale changes k but not stored theta -------------------------

    def test_logit_changes_k_relative_to_passthrough(
        self, mocker, test_genotypes, base_kwargs
    ):
        """logit rescale must produce different k values than passthrough."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])

        df_pass, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_rescale="passthrough",
            **base_kwargs,
        )
        df_logit, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_rescale="logit",
            **base_kwargs,
        )
        assert not np.allclose(df_pass["k_pre"].values, df_logit["k_pre"].values)

    def test_logit_theta_column_stays_in_unit_interval(
        self, mocker, test_genotypes, base_kwargs
    ):
        """The stored 'theta' column must remain in (0, 1) even with logit rescale."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])

        df, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_rescale="logit",
            **base_kwargs,
        )
        theta_vals = df["theta"].values
        assert np.all(theta_vals > 0.0)
        assert np.all(theta_vals < 1.0)

    def test_logit_k_matches_hand_calculation(
        self, mocker, test_genotypes, base_kwargs, simple_growth_params
    ):
        """k = b + activity * m * logit(theta) + dk_geno for the linear model."""
        concs = np.array([10.0, 100.0])
        # Use theta=0.5 so logit(0.5)=0 and dk_geno=0 (wt only) for easy math.
        # wt gets dk_geno=0, others may not — use all identical theta and zero
        # activity_mut_scale so only the logit transform matters.
        theta_gc = np.full((3, 2), 0.5)

        df, _, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_rescale="logit",
            activity_wt=1.0,
            activity_mut_scale=0.0,
            **base_kwargs,
        )
        # logit(0.5) = 0, so k = b + activity * m * 0 + dk_geno = b + dk_geno
        b_pre = simple_growth_params["M9"]["b"]
        b_sel = simple_growth_params["M9+Ab"]["b"]
        dk = df["dk_geno"].values
        np.testing.assert_allclose(df["k_pre"].values, b_pre + dk, rtol=1e-6)
        np.testing.assert_allclose(df["k_sel"].values, b_sel + dk, rtol=1e-6)

    def test_genotype_theta_df_stays_in_unit_interval_under_logit(
        self, mocker, test_genotypes, base_kwargs
    ):
        """genotype_theta_df stores pre-rescale (0-1) theta regardless of theta_rescale."""
        concs = np.array([10.0, 100.0])
        theta_gc = np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])

        _, geno_theta_df, _ = self._run(
            mocker, test_genotypes, concs, theta_gc,
            theta_rescale="logit",
            **base_kwargs,
        )
        theta_cols = [c for c in geno_theta_df.columns if c.startswith("theta_at_")]
        vals = geno_theta_df[theta_cols].values
        assert np.all(vals > 0.0)
        assert np.all(vals < 1.0)


# ============================================================================
# test _theta_param_to_df
# ============================================================================

from dataclasses import dataclass as _stdlib_dataclass


@_stdlib_dataclass
class _MockThetaParam2D:
    """Minimal stand-in for a real ThetaParam with only 2-D per-genotype fields."""
    theta_low: object
    theta_high: object
    log_hill_K: object
    hill_n: object


@_stdlib_dataclass
class _MockThetaParamMixed:
    """ThetaParam with 2-D per-genotype fields AND a 3-D population field."""
    theta_low: object
    mu: object      # 3-D — should be ignored


class TestThetaParamToDf:
    """Unit tests for _theta_param_to_df."""

    def _make_param_single_titrant(self, G):
        """Return a _MockThetaParam2D with shape (1, G) fields."""
        rng = np.random.default_rng(0)
        return _MockThetaParam2D(
            theta_low=rng.random((1, G)),
            theta_high=rng.random((1, G)),
            log_hill_K=rng.standard_normal((1, G)),
            hill_n=np.abs(rng.standard_normal((1, G))),
        )

    def test_returns_dataframe(self):
        genotypes = ["wt", "A1V", "B2G"]
        sim_idx = np.array([0, 1, 2])
        param = self._make_param_single_titrant(3)
        df = _theta_param_to_df(param, genotypes, sim_idx)
        assert isinstance(df, pd.DataFrame)

    def test_genotype_column_present_and_correct(self):
        genotypes = ["wt", "A1V", "B2G"]
        sim_idx = np.array([0, 1, 2])
        param = self._make_param_single_titrant(3)
        df = _theta_param_to_df(param, genotypes, sim_idx)
        assert "genotype" in df.columns
        assert list(df["genotype"]) == genotypes

    def test_genotype_is_first_column(self):
        genotypes = ["wt", "A1V"]
        sim_idx = np.array([0, 1])
        param = self._make_param_single_titrant(2)
        df = _theta_param_to_df(param, genotypes, sim_idx)
        assert df.columns[0] == "genotype"

    def test_single_titrant_fields_extracted(self):
        """All four 2-D fields must appear as columns (T=1 is squeezed)."""
        genotypes = ["wt", "A1V", "B2G"]
        sim_idx = np.array([0, 1, 2])
        param = self._make_param_single_titrant(3)
        df = _theta_param_to_df(param, genotypes, sim_idx)
        for fname in ["theta_low", "theta_high", "log_hill_K", "hill_n"]:
            assert fname in df.columns, f"Missing column: {fname}"

    def test_3d_field_is_skipped(self):
        """A field with ndim=3 (e.g. population moments) must not appear."""
        genotypes = ["wt", "A1V"]
        sim_idx = np.array([0, 1])
        param = _MockThetaParamMixed(
            theta_low=np.array([[0.1, 0.2]]),   # (1, 2) — kept
            mu=np.array([[[0.5], [0.6]]]),        # (1, 2, 1) — skipped
        )
        df = _theta_param_to_df(param, genotypes, sim_idx)
        assert "theta_low" in df.columns
        assert "mu" not in df.columns

    def test_values_match_theta_param(self):
        """Extracted values must match the raw theta_param arrays."""
        genotypes = ["wt", "A1V", "B2G"]
        sim_idx = np.array([0, 1, 2])
        param = self._make_param_single_titrant(3)
        df = _theta_param_to_df(param, genotypes, sim_idx)
        np.testing.assert_allclose(df["theta_low"].values, param.theta_low[0])
        np.testing.assert_allclose(df["theta_high"].values, param.theta_high[0])

    def test_sim_indices_select_correct_rows(self):
        """sim_indices must select the right columns from theta_param."""
        # sim_data has 5 genotypes; we want 3 of them in reverse order
        G_sim = 5
        G_out = 3
        sim_idx = np.array([4, 2, 0])   # select in reverse from sim_data
        rng = np.random.default_rng(7)
        theta_low_full = rng.random((1, G_sim))
        param = _MockThetaParam2D(
            theta_low=theta_low_full,
            theta_high=rng.random((1, G_sim)),
            log_hill_K=rng.standard_normal((1, G_sim)),
            hill_n=np.abs(rng.standard_normal((1, G_sim))),
        )
        genotypes = ["g4", "g2", "g0"]
        df = _theta_param_to_df(param, genotypes, sim_idx)
        expected = theta_low_full[0, sim_idx]
        np.testing.assert_allclose(df["theta_low"].values, expected)

    def test_multi_titrant_creates_suffixed_columns(self):
        """When T > 1, columns are named field_T0, field_T1, …"""
        T, G = 2, 3
        rng = np.random.default_rng(1)
        param = _MockThetaParam2D(
            theta_low=rng.random((T, G)),
            theta_high=rng.random((T, G)),
            log_hill_K=rng.standard_normal((T, G)),
            hill_n=np.abs(rng.standard_normal((T, G))),
        )
        genotypes = ["wt", "A1V", "B2G"]
        sim_idx = np.array([0, 1, 2])
        df = _theta_param_to_df(param, genotypes, sim_idx)
        assert "theta_low_T0" in df.columns
        assert "theta_low_T1" in df.columns
        assert "theta_low" not in df.columns   # unsuffixed form absent

    def test_non_dataclass_returns_genotype_only(self):
        """A non-dataclass theta_param (e.g. MagicMock) yields only 'genotype' column."""
        from unittest.mock import MagicMock
        mock_param = MagicMock()
        genotypes = ["wt", "A1V"]
        sim_idx = np.array([0, 1])
        df = _theta_param_to_df(mock_param, genotypes, sim_idx)
        assert set(df.columns) == {"genotype"}
        assert list(df["genotype"]) == genotypes

    def test_one_row_per_genotype(self):
        genotypes = ["wt", "A1V", "B2G", "C3H"]
        sim_idx = np.array([0, 1, 2, 3])
        param = self._make_param_single_titrant(4)
        df = _theta_param_to_df(param, genotypes, sim_idx)
        assert len(df) == 4
