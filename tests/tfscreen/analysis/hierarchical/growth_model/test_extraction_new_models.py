"""
Tests for the new-model branches added to extraction.py:
  - extract_parameters with theta='hill_mut'
  - extract_parameters with dk_geno='hierarchical_mut'
  - extract_parameters with activity='hierarchical_mut'
  - _build_manual_calc_df_hill  (refactored private helper)
  - _build_manual_calc_df_hill_mut
  - _extract_theta_curves_hill_mut
  - extract_theta_curves dispatcher (hill / hill_mut / other)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.extraction import (
    extract_parameters,
    extract_theta_curves,
    _build_manual_calc_df_hill,
    _build_manual_calc_df_hill_mut,
    _extract_theta_curves_hill,
    _extract_theta_curves_hill_mut,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_model(theta="none", dk_geno="none", activity="fixed",
                condition_growth="none", growth_transition="instant",
                transformation="none"):
    """Build a minimal MagicMock model with two titrants and three genotypes."""
    model = MagicMock(spec=ModelClass)
    model._theta = theta
    model._dk_geno = dk_geno
    model._activity = activity
    model._condition_growth = condition_growth
    model._growth_transition = growth_transition
    model._transformation = transformation
    model._growth_shares_replicates = False

    # T=2 titrant names (iptg=0, atc=1), G=3 genotypes (wt=0, A=1, B=2)
    model_df = pd.DataFrame({
        "genotype":         ["wt", "A",  "B",  "wt", "A",  "B"],
        "titrant_name":     ["iptg","iptg","iptg","atc","atc","atc"],
        "titrant_conc":     [1.0,  1.0,  1.0,  2.0,  2.0,  2.0],
        "genotype_idx":     [0,    1,    2,    0,    1,    2],
        "titrant_name_idx": [0,    0,    0,    1,    1,    1],
        "titrant_conc_idx": [0,    0,    0,    1,    1,    1],
        # hill model: map_theta_group = titrant_name_idx * 3 + genotype_idx
        "map_theta_group":  [0,    1,    2,    3,    4,    5],
        "map_theta":        [0,    1,    2,    3,    4,    5],
        "map_genotype":     [0,    1,    2,    0,    1,    2],
        "map_ln_cfu0":      [0,    1,    2,    3,    4,    5],
        "map_condition_rep":[0,    0,    0,    1,    1,    1],
        "replicate":        ["1",  "1",  "1",  "1",  "1",  "1"],
        "condition_rep":    ["c1", "c1", "c1", "c2", "c2", "c2"],
        "condition_pre":    ["p1", "p1", "p1", "p2", "p2", "p2"],
        "condition_sel":    ["s1", "s1", "s1", "s2", "s2", "s2"],
        "replicate_idx":    [0,    0,    0,    0,    0,    0],
        "time_idx":         [0,    0,    0,    0,    0,    0],
        "condition_pre_idx":[0,    0,    0,    1,    1,    1],
        "condition_sel_idx":[0,    0,    0,    1,    1,    1],
    })

    mock_tm = MagicMock()
    mock_tm.df = model_df
    mock_tm.map_groups = {
        "condition_rep": pd.DataFrame({
            "replicate": ["1", "1"],
            "condition_rep": ["c1", "c2"],
            "map_condition_rep": [0, 1],
        })
    }
    # dim order: replicate, time, condition_pre, condition_sel,
    #            titrant_name, titrant_conc, genotype
    mock_tm.tensor_dim_names = [
        "replicate", "time", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "genotype",
    ]
    mock_tm.tensor_dim_labels = [
        ["1"], ["0"], ["p1","p2"], ["s1","s2"],
        ["iptg","atc"], [1.0, 2.0], ["wt","A","B"],
    ]
    model.growth_tm = mock_tm
    model.growth_df = model_df.copy()
    model.growth_df["t_pre"] = 0.0
    model.growth_df["t_sel"] = 1.0
    model.growth_df["ln_cfu"] = 10.0
    model.growth_df["ln_cfu_std"] = 0.1
    # Mutation/pair labels (2 mutations, no pairs by default)
    model.mut_labels = ["A", "B"]
    model.pair_labels = []
    return model


_Q = {"median": 0.5}   # single quantile to keep assertions simple


# ---------------------------------------------------------------------------
# extract_parameters – hill_mut theta
# ---------------------------------------------------------------------------

class TestExtractParametersHillMut:

    def _posteriors(self, S=5):
        # T=2 titrants, G=3 genotypes, M=2 mutations; flat sizes 6 and 4
        return {
            # assembled per-genotype (T, G)
            "theta_theta_low":  np.random.rand(S, 2, 3),
            "theta_theta_high": np.random.rand(S, 2, 3),
            "theta_log_hill_K": np.random.rand(S, 2, 3),
            "theta_hill_n":     np.abs(np.random.rand(S, 2, 3)) + 0.1,
            # per-mutation deltas (T, M)
            "theta_d_logit_low":   np.random.rand(S, 2, 2),
            "theta_d_logit_delta": np.random.rand(S, 2, 2),
            "theta_d_log_hill_K":  np.random.rand(S, 2, 2),
            "theta_d_log_hill_n":  np.random.rand(S, 2, 2),
        }

    def test_returns_assembled_and_mutation_params(self):
        model = _make_model(theta="hill_mut")
        params = extract_parameters(model, self._posteriors(), q_to_get=_Q)
        for name in ["theta_low", "theta_high", "log_hill_K", "hill_n"]:
            assert name in params, f"Missing assembled key: {name}"
        for name in ["d_logit_low", "d_logit_delta", "d_log_hill_K", "d_log_hill_n"]:
            assert name in params, f"Missing mutation-level key: {name}"

    def test_assembled_output_shape_genotype_x_titrant(self):
        model = _make_model(theta="hill_mut")
        params = extract_parameters(model, self._posteriors(), q_to_get=_Q)
        # 3 genotypes × 2 titrant names = 6 rows
        df = params["theta_low"]
        assert len(df) == 6
        assert set(df.columns) >= {"genotype", "titrant_name", "median"}

    def test_mutation_output_shape_mutation_x_titrant(self):
        model = _make_model(theta="hill_mut")
        params = extract_parameters(model, self._posteriors(), q_to_get=_Q)
        # 2 mutations × 2 titrant names = 4 rows
        df = params["d_logit_low"]
        assert len(df) == 4
        assert set(df.columns) >= {"mutation", "titrant_name", "median"}

    def test_compound_index_selects_correct_posterior(self):
        """Each row should use the posterior slice for its (titrant_name_idx, genotype_idx)."""
        model = _make_model(theta="hill_mut")
        S = 1
        theta_low = np.arange(6, dtype=float).reshape(1, 2, 3)  # shape (S, T, G)
        d_low = np.arange(4, dtype=float).reshape(1, 2, 2)      # shape (S, T, M)
        posteriors = {
            "theta_theta_low":  theta_low,
            "theta_theta_high": theta_low + 1.0,
            "theta_log_hill_K": np.zeros((S, 2, 3)),
            "theta_hill_n":     np.ones((S, 2, 3)),
            "theta_d_logit_low":   d_low,
            "theta_d_logit_delta": np.zeros((S, 2, 2)),
            "theta_d_log_hill_K":  np.zeros((S, 2, 2)),
            "theta_d_log_hill_n":  np.zeros((S, 2, 2)),
        }
        params = extract_parameters(model, posteriors, q_to_get=_Q)

        # --- assembled (T, G) indexing ---
        df = params["theta_low"].set_index(["titrant_name", "genotype"])
        assert df.loc[("iptg", "wt"), "median"] == pytest.approx(0.0)
        assert df.loc[("iptg", "B"),  "median"] == pytest.approx(2.0)
        assert df.loc[("atc",  "A"),  "median"] == pytest.approx(4.0)
        assert df.loc[("atc",  "B"),  "median"] == pytest.approx(5.0)

        # --- per-mutation (T, M) indexing: flat index = titrant_idx*M + mut_idx ---
        dd = params["d_logit_low"].set_index(["titrant_name", "mutation"])
        # (iptg=0, A=0) → flat 0*2+0=0 → value 0.0
        assert dd.loc[("iptg", "A"), "median"] == pytest.approx(0.0)
        # (iptg=0, B=1) → flat 0*2+1=1 → value 1.0
        assert dd.loc[("iptg", "B"), "median"] == pytest.approx(1.0)
        # (atc=1,  A=0) → flat 1*2+0=2 → value 2.0
        assert dd.loc[("atc",  "A"), "median"] == pytest.approx(2.0)
        # (atc=1,  B=1) → flat 1*2+1=3 → value 3.0
        assert dd.loc[("atc",  "B"), "median"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# extract_parameters – hierarchical_mut dk_geno and activity
# ---------------------------------------------------------------------------

class TestExtractParametersHierarchicalMut:

    def _dk_geno_mut_posteriors(self, S=5):
        return {
            "ln_cfu0":        np.random.rand(S, 6),
            "dk_geno":        np.random.rand(S, 3),
            "dk_geno_d_dk_geno": np.random.rand(S, 2),  # M=2
        }

    def _activity_mut_posteriors(self, S=5):
        return {
            "activity":                np.random.rand(S, 3),
            "activity_d_log_activity": np.random.rand(S, 2),  # M=2
        }

    def test_dk_geno_hierarchical_mut_extracts_ln_cfu0_and_dk_geno(self):
        model = _make_model(dk_geno="hierarchical_mut")
        params = extract_parameters(model, self._dk_geno_mut_posteriors(), q_to_get=_Q)
        assert "ln_cfu0" in params
        assert "dk_geno" in params

    def test_dk_geno_hierarchical_mut_extracts_per_mutation(self):
        model = _make_model(dk_geno="hierarchical_mut")
        params = extract_parameters(model, self._dk_geno_mut_posteriors(), q_to_get=_Q)
        assert "d_dk_geno" in params
        df = params["d_dk_geno"]
        assert len(df) == 2   # M=2 mutations
        assert "mutation" in df.columns

    def test_dk_geno_hierarchical_mut_dk_geno_shape(self):
        model = _make_model(dk_geno="hierarchical_mut")
        params = extract_parameters(model, self._dk_geno_mut_posteriors(), q_to_get=_Q)
        assert len(params["dk_geno"]) == 3
        assert "genotype" in params["dk_geno"].columns

    def test_activity_hierarchical_mut_extracts_activity_and_per_mutation(self):
        model = _make_model(activity="hierarchical_mut")
        params = extract_parameters(model, self._activity_mut_posteriors(), q_to_get=_Q)
        assert "activity" in params
        assert len(params["activity"]) == 3
        assert "d_log_activity" in params
        df = params["d_log_activity"]
        assert len(df) == 2   # M=2 mutations
        assert "mutation" in df.columns

    def test_dk_geno_hierarchical_mut_same_genotype_output_as_hierarchical(self):
        """The assembled dk_geno and ln_cfu0 tables are identical for both variants."""
        base_post = {
            "ln_cfu0": np.ones((4, 6)) * 0.5,
            "dk_geno": np.ones((4, 3)) * 0.25,
        }
        model_hier = _make_model(dk_geno="hierarchical")
        model_mut  = _make_model(dk_geno="hierarchical_mut")
        p_hier = extract_parameters(model_hier, base_post, q_to_get=_Q)
        p_mut  = extract_parameters(model_mut,
                                    {**base_post, "dk_geno_d_dk_geno": np.ones((4, 2))},
                                    q_to_get=_Q)
        pd.testing.assert_frame_equal(p_hier["dk_geno"], p_mut["dk_geno"])
        pd.testing.assert_frame_equal(p_hier["ln_cfu0"], p_mut["ln_cfu0"])

    def test_activity_hierarchical_mut_same_genotype_output_as_horseshoe(self):
        base_post = {"activity": np.ones((4, 3)) * 0.7}
        model_hs  = _make_model(activity="horseshoe")
        model_mut = _make_model(activity="hierarchical_mut")
        p_hs  = extract_parameters(model_hs,  base_post, q_to_get=_Q)
        p_mut = extract_parameters(model_mut,
                                   {**base_post, "activity_d_log_activity": np.ones((4, 2))},
                                   q_to_get=_Q)
        pd.testing.assert_frame_equal(p_hs["activity"], p_mut["activity"])


# ---------------------------------------------------------------------------
# _build_manual_calc_df_hill
# ---------------------------------------------------------------------------

class TestBuildManualCalcDfHill:

    @pytest.fixture
    def model(self):
        return _make_model(theta="hill")

    def test_with_genotype_attaches_map_theta_group(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
            "genotype": ["A"],
        })
        result = _build_manual_calc_df_hill(model, manual_df)
        assert "map_theta_group" in result.columns
        assert result["map_theta_group"].iloc[0] == 1   # iptg(0)*3 + A(1) = 1

    def test_without_genotype_broadcasts_to_all_genotypes(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
        })
        result = _build_manual_calc_df_hill(model, manual_df)
        assert len(result) == 3   # wt, A, B
        assert set(result["genotype"]) == {"wt", "A", "B"}

    def test_missing_titrant_name_column_raises(self, model):
        bad_df = pd.DataFrame({"titrant_conc": [1.0]})
        with pytest.raises(Exception):
            _build_manual_calc_df_hill(model, bad_df)

    def test_unknown_genotype_titrant_pair_raises_value_error(self, model):
        # "iptg" exists but "unknown" genotype does not
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
            "genotype": ["unknown"],
        })
        with pytest.raises(ValueError, match=r"not found in the model data"):
            _build_manual_calc_df_hill(model, manual_df)

    def test_map_raises_exception_re_raised_as_value_error(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
        })
        with patch("pandas.Index.map", side_effect=Exception("boom")):
            with pytest.raises(ValueError, match=r"Some \(genotype, titrant_name\) pairs"):
                _build_manual_calc_df_hill(model, manual_df)


# ---------------------------------------------------------------------------
# _build_manual_calc_df_hill_mut
# ---------------------------------------------------------------------------

class TestBuildManualCalcDfHillMut:

    @pytest.fixture
    def model(self):
        return _make_model(theta="hill_mut")

    def test_with_genotype_attaches_correct_indices(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["atc"],
            "titrant_conc": [2.0],
            "genotype": ["B"],
        })
        result = _build_manual_calc_df_hill_mut(model, manual_df)
        assert result["genotype_idx"].iloc[0] == 2
        assert result["titrant_name_idx"].iloc[0] == 1

    def test_without_genotype_broadcasts_to_all_genotypes(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
        })
        result = _build_manual_calc_df_hill_mut(model, manual_df)
        assert len(result) == 3
        assert set(result["genotype"]) == {"wt", "A", "B"}

    def test_unknown_genotype_raises_value_error(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
            "genotype": ["NOTAGENOTYPE"],
        })
        with pytest.raises(ValueError, match=r"not found in the model data"):
            _build_manual_calc_df_hill_mut(model, manual_df)

    def test_unknown_titrant_name_raises_value_error(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["unknown_drug"],
            "titrant_conc": [1.0],
            "genotype": ["wt"],
        })
        with pytest.raises(ValueError, match=r"not found in the model data"):
            _build_manual_calc_df_hill_mut(model, manual_df)

    def test_missing_required_column_raises(self, model):
        bad_df = pd.DataFrame({"titrant_name": ["iptg"]})
        with pytest.raises(Exception):
            _build_manual_calc_df_hill_mut(model, bad_df)


# ---------------------------------------------------------------------------
# _extract_theta_curves_hill_mut
# ---------------------------------------------------------------------------

class TestExtractThetaCurvesHillMut:

    @pytest.fixture
    def model(self):
        return _make_model(theta="hill_mut")

    def _flat_posteriors(self, S=10):
        """Simple all-equal posteriors so medians are predictable."""
        return {
            "theta_hill_n":     np.ones((S, 2, 3)),
            "theta_log_hill_K": np.zeros((S, 2, 3)),
            "theta_theta_high": np.full((S, 2, 3), 0.9),
            "theta_theta_low":  np.full((S, 2, 3), 0.1),
        }

    def test_default_no_manual_df_output_shape(self, model):
        df = _extract_theta_curves_hill_mut(model, self._flat_posteriors(),
                                            q_to_get=_Q, manual_titrant_df=None)
        # 6 unique (genotype, titrant_name, titrant_conc) rows in model_df
        assert len(df) == 6
        assert "genotype" in df.columns
        assert "titrant_name" in df.columns
        assert "titrant_conc" in df.columns
        assert "median" in df.columns

    def test_index_columns_dropped_from_output(self, model):
        df = _extract_theta_curves_hill_mut(model, self._flat_posteriors(),
                                            q_to_get=_Q, manual_titrant_df=None)
        assert "genotype_idx" not in df.columns
        assert "titrant_name_idx" not in df.columns

    def test_hill_equation_math(self, model):
        """
        With hill_n=2, log_hill_K=-1, theta_high=0.9, theta_low=0.1,
        conc=1.0 (log=0.0):
          occupancy = sigmoid(2*(0.0 - (-1.0))) = sigmoid(2) ≈ 0.8808
          theta = 0.1 + 0.8 * 0.8808 ≈ 0.8046
        """
        S = 1
        posteriors = {
            "theta_hill_n":     np.full((S, 2, 3), 2.0),
            "theta_log_hill_K": np.full((S, 2, 3), -1.0),
            "theta_theta_high": np.full((S, 2, 3), 0.9),
            "theta_theta_low":  np.full((S, 2, 3), 0.1),
        }
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
            "genotype": ["wt"],
        })
        df = _extract_theta_curves_hill_mut(model, posteriors,
                                            q_to_get=_Q, manual_titrant_df=manual_df)
        expected_occ = 1.0 / (1.0 + np.exp(-2.0 * (0.0 - (-1.0))))
        expected_theta = 0.1 + 0.8 * expected_occ
        assert df["median"].iloc[0] == pytest.approx(expected_theta, rel=1e-5)

    def test_different_indices_use_different_posteriors(self, model):
        """
        Give each (T, G) cell a distinct posterior value; verify each row
        gets the correct cell's value.
        """
        S = 1
        base = np.arange(6, dtype=float).reshape(1, 2, 3)
        posteriors = {
            "theta_hill_n":     np.ones((S, 2, 3)),
            "theta_log_hill_K": np.zeros((S, 2, 3)),
            "theta_theta_high": base + 1.0,   # theta range > 1 is fine here
            "theta_theta_low":  base,          # theta_low[t,g] = t*3+g
        }
        # Request two specific cells
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg", "atc"],
            "titrant_conc": [1.0,    2.0],
            "genotype":     ["A",    "B"],
        })
        df = _extract_theta_curves_hill_mut(model, posteriors,
                                            q_to_get=_Q, manual_titrant_df=manual_df)
        # iptg(0)/A(1) → theta_low = 0*3+1 = 1.0
        row_iptg_A = df[(df["genotype"] == "A") & (df["titrant_name"] == "iptg")]
        # atc(1)/B(2) → theta_low = 1*3+2 = 5.0
        row_atc_B  = df[(df["genotype"] == "B") & (df["titrant_name"] == "atc")]

        # iptg row: conc=1.0 → log_conc=0, log_K=0, hill_n=1
        #   occupancy = sigmoid(0) = 0.5  →  theta = 1.0 + 1.0*0.5 = 1.5
        occ_iptg = 1.0 / (1.0 + np.exp(-1.0 * (np.log(1.0) - 0.0)))
        assert row_iptg_A["median"].iloc[0] == pytest.approx(1.0 + 1.0 * occ_iptg, rel=1e-5)

        # atc row: conc=2.0 → log_conc=log(2), log_K=0, hill_n=1
        #   occupancy = sigmoid(log(2)) = 2/3  →  theta = 5.0 + 1.0*(2/3)
        occ_atc = 1.0 / (1.0 + np.exp(-1.0 * (np.log(2.0) - 0.0)))
        assert row_atc_B["median"].iloc[0]  == pytest.approx(5.0 + 1.0 * occ_atc, rel=1e-5)

    def test_zero_concentration_handled(self, model):
        """Zero concentrations should not produce NaN (uses ZERO_CONC_VALUE)."""
        posteriors = self._flat_posteriors()
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [0.0],
            "genotype": ["wt"],
        })
        df = _extract_theta_curves_hill_mut(model, posteriors,
                                            q_to_get=_Q, manual_titrant_df=manual_df)
        assert not df["median"].isna().any()

    def test_broadcast_without_genotype(self, model):
        manual_df = pd.DataFrame({
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
        })
        df = _extract_theta_curves_hill_mut(model, self._flat_posteriors(),
                                            q_to_get=_Q, manual_titrant_df=manual_df)
        assert len(df) == 3
        assert set(df["genotype"]) == {"wt", "A", "B"}


# ---------------------------------------------------------------------------
# extract_theta_curves – dispatcher
# ---------------------------------------------------------------------------

class TestExtractThetaCurvesDispatcher:

    def test_dispatches_to_hill(self):
        model = _make_model(theta="hill")
        posteriors = {
            "theta_hill_n":     np.random.rand(5, 6),
            "theta_log_hill_K": np.random.rand(5, 6),
            "theta_theta_high": np.random.rand(5, 6),
            "theta_theta_low":  np.random.rand(5, 6),
        }
        df = extract_theta_curves(model, posteriors, q_to_get=_Q)
        assert "map_theta_group" not in df.columns
        assert "median" in df.columns

    def test_dispatches_to_hill_mut(self):
        model = _make_model(theta="hill_mut")
        posteriors = {
            "theta_hill_n":     np.random.rand(5, 2, 3),
            "theta_log_hill_K": np.random.rand(5, 2, 3),
            "theta_theta_high": np.random.rand(5, 2, 3),
            "theta_theta_low":  np.random.rand(5, 2, 3),
        }
        df = extract_theta_curves(model, posteriors, q_to_get=_Q)
        assert "genotype_idx" not in df.columns
        assert "median" in df.columns

    def test_raises_for_categorical(self):
        model = _make_model(theta="categorical")
        with pytest.raises(ValueError, match=r"theta='hill'"):
            extract_theta_curves(model, {})

    def test_raises_for_unknown_theta(self):
        model = _make_model(theta="something_new")
        with pytest.raises(ValueError, match=r"theta='hill'"):
            extract_theta_curves(model, {})

    def test_hill_and_hill_mut_agree_on_math(self):
        """
        For a single genotype/titrant pair the two implementations should
        produce the same theta when given equivalent posteriors.
        """
        S = 20
        rng = np.random.default_rng(42)
        h_n  = rng.uniform(0.5, 2.0, (S, 1))   # (S, G) for hill
        l_K  = rng.uniform(-2.0, 0.0, (S, 1))
        t_h  = rng.uniform(0.7, 1.0, (S, 1))
        t_l  = rng.uniform(0.0, 0.3, (S, 1))

        # hill model: single group
        model_hill = _make_model(theta="hill")
        # Restrict to just iptg/wt for simplicity
        model_hill.growth_tm.df = pd.DataFrame({
            "genotype": ["wt"],
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
            "map_theta_group": [0],
            "genotype_idx": [0],
            "titrant_name_idx": [0],
        })
        posteriors_hill = {
            "theta_hill_n": h_n,
            "theta_log_hill_K": l_K,
            "theta_theta_high": t_h,
            "theta_theta_low": t_l,
        }
        df_hill = extract_theta_curves(model_hill, posteriors_hill, q_to_get=_Q)

        # hill_mut model: T=1 titrants, G=1 genotype
        model_mut = _make_model(theta="hill_mut")
        model_mut.growth_tm.df = pd.DataFrame({
            "genotype": ["wt"],
            "titrant_name": ["iptg"],
            "titrant_conc": [1.0],
            "genotype_idx": [0],
            "titrant_name_idx": [0],
            "map_theta_group": [0],
        })
        model_mut.growth_tm.tensor_dim_labels[6] = ["wt"]   # 1 genotype
        posteriors_mut = {
            "theta_hill_n":    h_n[:, :, np.newaxis],    # (S, 1, 1)
            "theta_log_hill_K": l_K[:, :, np.newaxis],
            "theta_theta_high": t_h[:, :, np.newaxis],
            "theta_theta_low":  t_l[:, :, np.newaxis],
        }
        df_mut = extract_theta_curves(model_mut, posteriors_mut, q_to_get=_Q)

        assert df_hill["median"].iloc[0] == pytest.approx(
            df_mut["median"].iloc[0], rel=1e-5)
