import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from numpy.random import Generator

from tfscreen.simulate.thermo_to_growth import (
    _assign_ddG,
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
def simple_genotypes() -> list[str]:
    return ["wt", "A1B", "C2D", "A1B/C2D"]

@pytest.fixture
def simple_ddG_df() -> pd.DataFrame:
    return pd.DataFrame({"spec1": [1.0, -0.5]}, index=["A1B", "C2D"])

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
# test _assign_ddG
# ----------------------------------------------------------------------------

def test_assign_ddG_calls_combiner(mocker, simple_genotypes, simple_ddG_df):
    mock_combiner = mocker.patch(
        "tfscreen.simulate.thermo_to_growth.combine_mutation_effects",
        return_value="success",
    )
    result = _assign_ddG(simple_genotypes, simple_ddG_df, mut_combine_fcn="mean")
    mock_combiner.assert_called_once_with(
        unique_genotypes=simple_genotypes,
        single_mutant_effects=simple_ddG_df,
        mut_combine_fcn="mean",
    )
    assert result == "success"


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
        assert not np.allclose(mut_values, 1.0), "Mutants should vary when scale > 0"

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

def test_assign_dk_geno_sampling_and_integration(rng, simple_genotypes):
    result = _assign_dk_geno(simple_genotypes, rng=rng)

    assert isinstance(result, pd.Series)
    assert len(result) == len(simple_genotypes)
    assert all(result.index == simple_genotypes)
    assert result.loc["wt"] == 0.0

    expected_A1B = -0.0053917202444200745
    expected_C2D = -0.007178919203856204

    assert np.isclose(result.loc["A1B"], expected_A1B, rtol=1e-5)
    assert np.isclose(result.loc["C2D"], expected_C2D, rtol=1e-5)
    assert np.isclose(result.loc["A1B/C2D"], expected_A1B + expected_C2D, rtol=1e-5)


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
        # k = m*0 + b = b
        k = _apply_growth_params(np.array(["sel"]), np.array([0.0]), growth_params)
        assert np.isclose(k[0], 0.005)

    def test_single_condition_at_theta_one(self, growth_params):
        # k = m*1 + b = m + b
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

    def test_linearity(self, growth_params):
        # Doubling theta doubles the m contribution
        conds = np.array(["sel", "sel"])
        k_half = _apply_growth_params(conds, np.array([0.5, 0.5]), growth_params)
        k_one  = _apply_growth_params(conds, np.array([1.0, 1.0]), growth_params)
        # (k_one - b) == 2 * (k_half - b)  →  m*1 == 2 * m*0.5
        b = growth_params["sel"]["b"]
        np.testing.assert_allclose((k_one - b), 2 * (k_half - b))

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
# test thermo_to_growth — missing condition validation
# ----------------------------------------------------------------------------

class TestThermo_to_growth_Validation:

    def _patch_dependencies(self, mocker, test_genotypes, test_sample_df, growth_params):
        """Patch everything so only the validation code is exercised."""
        sorted_genotypes = np.array(test_genotypes)
        mocker.patch("tfscreen.simulate.thermo_to_growth.standardize_genotypes",
                     return_value=test_genotypes)
        mocker.patch("tfscreen.simulate.thermo_to_growth.argsort_genotypes",
                     return_value=np.arange(len(sorted_genotypes)))
        mock_theta = MagicMock(return_value=np.zeros(len(test_sample_df)))
        mocker.patch("tfscreen.simulate.thermo_to_growth.setup_observable",
                     return_value=(mock_theta, pd.DataFrame({"spec1": [0.0]}, index=["A1B"])))
        mocker.patch("tfscreen.simulate.thermo_to_growth._assign_ddG",
                     return_value=pd.DataFrame({"spec1": np.zeros(len(sorted_genotypes))},
                                               index=sorted_genotypes))
        mocker.patch("tfscreen.simulate.thermo_to_growth._assign_dk_geno",
                     return_value=pd.Series(0.0, index=sorted_genotypes))
        mocker.patch("tfscreen.simulate.thermo_to_growth.set_categorical_genotype",
                     side_effect=lambda df: df)
        return growth_params

    def test_missing_condition_pre_raises(self, mocker, test_genotypes, test_sample_df):
        # condition_pre = "M9" is absent from growth_params
        growth_params = {"M9+Ab": {"m": -0.01, "b": 0.005}}
        self._patch_dependencies(mocker, test_genotypes, test_sample_df, growth_params)
        with pytest.raises(ValueError, match="M9"):
            thermo_to_growth(
                genotypes=test_genotypes,
                sample_df=test_sample_df,
                observable_calculator="mock",
                observable_calc_kwargs={"e_name": "IPTG"},
                ddG_df="dummy.csv",
                growth_params=growth_params,
            )

    def test_missing_condition_sel_raises(self, mocker, test_genotypes, test_sample_df):
        # condition_sel = "M9+Ab" is absent from growth_params
        growth_params = {"M9": {"m": 0.001, "b": 0.020}}
        self._patch_dependencies(mocker, test_genotypes, test_sample_df, growth_params)
        with pytest.raises(ValueError, match="M9\\+Ab"):
            thermo_to_growth(
                genotypes=test_genotypes,
                sample_df=test_sample_df,
                observable_calculator="mock",
                observable_calc_kwargs={"e_name": "IPTG"},
                ddG_df="dummy.csv",
                growth_params=growth_params,
            )

    def test_all_conditions_present_does_not_raise(
        self, mocker, test_genotypes, test_sample_df, simple_growth_params
    ):
        self._patch_dependencies(mocker, test_genotypes, test_sample_df, simple_growth_params)
        # Should complete without raising
        thermo_to_growth(
            genotypes=test_genotypes,
            sample_df=test_sample_df,
            observable_calculator="mock",
            observable_calc_kwargs={"e_name": "IPTG"},
            ddG_df="dummy.csv",
            growth_params=simple_growth_params,
        )


# ----------------------------------------------------------------------------
# test thermo_to_growth — integration
# ----------------------------------------------------------------------------

def test_thermo_to_growth_integration(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """End-to-end wiring test with all external dependencies mocked."""
    sorted_genotypes = np.array(test_genotypes)
    mocker.patch("tfscreen.simulate.thermo_to_growth.standardize_genotypes",
                 return_value=test_genotypes)
    mocker.patch("tfscreen.simulate.thermo_to_growth.argsort_genotypes",
                 return_value=np.arange(len(sorted_genotypes)))

    mock_theta_fcn = mocker.Mock(return_value=np.array([0.5, 0.8]))
    mock_ddG_df = pd.DataFrame({"spec1": [0.0, 1.0]}, index=["A1B", "C2D"])
    mocker.patch("tfscreen.simulate.thermo_to_growth.setup_observable",
                 return_value=(mock_theta_fcn, mock_ddG_df))

    mock_genotype_ddG = pd.DataFrame(
        {"spec1": [0.0, 1.0, 1.5]}, index=sorted_genotypes
    )
    mocker.patch("tfscreen.simulate.thermo_to_growth._assign_ddG",
                 return_value=mock_genotype_ddG)

    mock_dk_geno = pd.Series({"wt": 0.0, "A1B": -0.01, "A1B/C2D": -0.02})
    mocker.patch("tfscreen.simulate.thermo_to_growth._assign_dk_geno",
                 return_value=mock_dk_geno)

    mocker.patch("tfscreen.simulate.thermo_to_growth.set_categorical_genotype",
                 side_effect=lambda df: df)

    phenotype_df, genotype_ddG_df = thermo_to_growth(
        genotypes=test_genotypes,
        sample_df=test_sample_df,
        observable_calculator="mock_calc",
        observable_calc_kwargs={"e_name": "IPTG"},
        ddG_df="dummy_path.csv",
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

    # With default activity_wt=1.0 and activity_mut_scale=0.0, all activity=1.0
    # k = b + 1.0 * m * theta + dk_geno
    m_pre = simple_growth_params["M9"]["m"]
    b_pre = simple_growth_params["M9"]["b"]
    m_sel = simple_growth_params["M9+Ab"]["m"]
    b_sel = simple_growth_params["M9+Ab"]["b"]
    dk = phenotype_df["genotype"].map(mock_dk_geno).to_numpy()
    theta = phenotype_df["theta"].to_numpy()
    activity = phenotype_df["activity"].to_numpy()

    np.testing.assert_allclose(
        phenotype_df["k_pre"].to_numpy(), b_pre + activity * m_pre * theta + dk
    )
    np.testing.assert_allclose(
        phenotype_df["k_sel"].to_numpy(), b_sel + activity * m_sel * theta + dk
    )

    assert isinstance(genotype_ddG_df, pd.DataFrame)
    assert "genotype" in genotype_ddG_df.columns


def test_thermo_to_growth_propagates_rng(
    mocker, test_genotypes, test_sample_df, simple_growth_params
):
    """rng argument is forwarded to _assign_dk_geno."""
    rng = np.random.default_rng(12345)
    sorted_genotypes = np.array(test_genotypes)

    mocker.patch("tfscreen.simulate.thermo_to_growth.standardize_genotypes",
                 return_value=test_genotypes)
    mocker.patch("tfscreen.simulate.thermo_to_growth.argsort_genotypes",
                 return_value=np.arange(len(sorted_genotypes)))
    mocker.patch("tfscreen.simulate.thermo_to_growth.setup_observable",
                 return_value=(
                     MagicMock(return_value=np.zeros(len(test_sample_df))),
                     pd.DataFrame({"spec1": [0.0]}, index=["A1B"]),
                 ))
    mocker.patch("tfscreen.simulate.thermo_to_growth._assign_ddG",
                 return_value=pd.DataFrame(
                     {"spec1": np.zeros(len(sorted_genotypes))},
                     index=sorted_genotypes,
                 ))
    mock_assign_dk = mocker.patch(
        "tfscreen.simulate.thermo_to_growth._assign_dk_geno",
        return_value=pd.Series(0.0, index=sorted_genotypes),
    )
    mocker.patch("tfscreen.simulate.thermo_to_growth.set_categorical_genotype",
                 side_effect=lambda df: df)

    thermo_to_growth(
        genotypes=test_genotypes,
        sample_df=test_sample_df,
        observable_calculator="mock_calc",
        observable_calc_kwargs={"e_name": "IPTG"},
        ddG_df="dummy.csv",
        growth_params=simple_growth_params,
        rng=rng,
    )

    _, kwargs = mock_assign_dk.call_args
    # rng is the 5th positional arg; check via call_args
    call_args = mock_assign_dk.call_args
    passed_rng = call_args.args[4] if call_args.args else call_args.kwargs.get("rng")
    assert passed_rng is rng
