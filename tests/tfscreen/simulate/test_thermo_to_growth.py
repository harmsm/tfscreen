import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from numpy.random import Generator

from tfscreen.simulate.thermo_to_growth import (
    _assign_activity,
    _assign_dk_geno,
    _apply_growth_params,
    thermo_to_growth,
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

    phenotype_df, genotype_theta_df = thermo_to_growth(
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

    # genotype_theta_df: one row per unique genotype, columns theta_at_*mM
    assert isinstance(genotype_theta_df, pd.DataFrame)
    assert "genotype" in genotype_theta_df.columns
    theta_cols = [c for c in genotype_theta_df.columns if c.startswith("theta_at_")]
    assert len(theta_cols) == 2   # two unique concentrations


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
        return_value=pd.Series(0.0, index=["wt", "A1B", "A1B/C2D"]),
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
