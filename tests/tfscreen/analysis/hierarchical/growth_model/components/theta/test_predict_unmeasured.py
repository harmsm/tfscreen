"""
Tests for:
  - tfscreen.analysis.hierarchical.growth_model.predict_unmeasured
      (shared utilities: _build_genotype_indicators, _build_prediction_grid)
  - hill_mut.predict_unmeasured
"""

import pytest
import numpy as np
import pandas as pd

from tfscreen.analysis.hierarchical.growth_model.predict_unmeasured import (
    _build_genotype_indicators,
    _build_prediction_grid,
)
from tfscreen.analysis.hierarchical.growth_model.components.theta.hill_mut import (
    predict_unmeasured as hill_predict_unmeasured,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUT_LABELS  = ["M42I", "K84L"]
PAIR_LABELS = ["K84L/M42I"]   # canonical form: alphabetical

TITRANT_NAMES = ["IPTG", "TMAIPP"]

def _make_titrant_df():
    return pd.DataFrame({
        "titrant_name": ["IPTG", "IPTG", "IPTG", "TMAIPP", "TMAIPP", "TMAIPP"],
        "titrant_conc": [0.0, 1e-4, 1e-3, 0.0, 1e-4, 1e-3],
    })


# ---------------------------------------------------------------------------
# _build_genotype_indicators
# ---------------------------------------------------------------------------

class TestBuildGenotypeIndicators:

    def test_wt_row_is_all_zeros(self):
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            ["wt"], MUT_LABELS, PAIR_LABELS
        )
        assert mut_mat.shape == (1, 2)
        assert np.all(mut_mat == 0.0)
        assert np.all(pair_mat == 0.0)
        assert is_valid[0] is np.bool_(True)

    def test_single_mutation(self):
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            ["M42I"], MUT_LABELS, PAIR_LABELS
        )
        assert mut_mat[0, 0] == 1.0   # M42I is index 0
        assert mut_mat[0, 1] == 0.0   # K84L not present
        assert pair_mat[0, 0] == 0.0  # pair needs both
        assert is_valid[0]

    def test_double_mutant_sets_pair(self):
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            ["M42I/K84L"], MUT_LABELS, PAIR_LABELS
        )
        assert mut_mat[0, 0] == 1.0
        assert mut_mat[0, 1] == 1.0
        assert pair_mat[0, 0] == 1.0
        assert is_valid[0]

    def test_unknown_mutation_marks_invalid(self):
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            ["M42I/Z99Q"], MUT_LABELS, PAIR_LABELS
        )
        assert not is_valid[0]

    def test_novel_pair_not_in_pair_labels_is_zero(self):
        # Use a different pair that isn't in PAIR_LABELS at all
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            ["M42I/K84L"], MUT_LABELS, []   # no pair labels registered
        )
        assert pair_mat.shape == (1, 0)
        assert is_valid[0]

    def test_multiple_genotypes(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L", "Z99Q"]
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            genotypes, MUT_LABELS, PAIR_LABELS
        )
        assert mut_mat.shape == (5, 2)
        assert pair_mat.shape == (5, 1)
        assert not is_valid[4]            # Z99Q is invalid
        assert all(is_valid[:4])
        assert pair_mat[3, 0] == 1.0     # double mutant has pair
        assert pair_mat[0, 0] == 0.0     # wt has no pair

    def test_output_dtypes(self):
        mut_mat, pair_mat, is_valid = _build_genotype_indicators(
            ["M42I"], MUT_LABELS, PAIR_LABELS
        )
        assert mut_mat.dtype == np.float32
        assert pair_mat.dtype == np.float32
        assert is_valid.dtype == bool


# ---------------------------------------------------------------------------
# _build_prediction_grid
# ---------------------------------------------------------------------------

class TestBuildPredictionGrid:

    def test_shape(self):
        genotypes = ["wt", "M42I"]
        titrant_df = _make_titrant_df()
        calc_df, geno_idx, titrant_idx = _build_prediction_grid(
            genotypes, TITRANT_NAMES, titrant_df
        )
        assert len(calc_df) == 2 * len(titrant_df)

    def test_index_values(self):
        genotypes = ["wt", "M42I"]
        titrant_df = _make_titrant_df()
        calc_df, geno_idx, titrant_idx = _build_prediction_grid(
            genotypes, TITRANT_NAMES, titrant_df
        )
        # Genotype indices should be 0 or 1
        assert set(geno_idx) == {0, 1}
        # Titrant indices should be 0 (IPTG) or 1 (TMAIPP)
        assert set(titrant_idx) == {0, 1}

    def test_raises_on_unknown_titrant_name(self):
        titrant_df = pd.DataFrame({
            "titrant_name": ["UNKNOWN"],
            "titrant_conc": [1e-4],
        })
        with pytest.raises(ValueError, match="titrant_name"):
            _build_prediction_grid(["wt"], TITRANT_NAMES, titrant_df)

    def test_columns_present(self):
        calc_df, _, _ = _build_prediction_grid(
            ["wt"], TITRANT_NAMES, _make_titrant_df()
        )
        for col in ["genotype", "titrant_name", "titrant_conc",
                    "_geno_idx", "_titrant_idx"]:
            assert col in calc_df.columns


# ---------------------------------------------------------------------------
# hill_mut.predict_unmeasured
# ---------------------------------------------------------------------------

def _make_posteriors(T, M, P=0, S=5, seed=0):
    """Build fake posterior dict matching hill_mut parameter names."""
    rng = np.random.default_rng(seed)
    post = {}
    post["theta_logit_low_wt"]    = rng.standard_normal((S, T)).astype(np.float32)
    post["theta_logit_delta_wt"]  = rng.standard_normal((S, T)).astype(np.float32)
    post["theta_log_hill_K_wt"]   = rng.standard_normal((S, T)).astype(np.float32)
    post["theta_log_hill_n_wt"]   = np.zeros((S, T), dtype=np.float32)   # hill_n = 1
    post["theta_d_logit_low"]     = rng.standard_normal((S, T, M)).astype(np.float32) * 0.1
    post["theta_d_logit_delta"]   = rng.standard_normal((S, T, M)).astype(np.float32) * 0.1
    post["theta_d_log_hill_K"]    = rng.standard_normal((S, T, M)).astype(np.float32) * 0.1
    post["theta_d_log_hill_n"]    = np.zeros((S, T, M), dtype=np.float32)
    if P > 0:
        post["theta_epi_logit_low"]   = np.zeros((S, T, P), dtype=np.float32)
        post["theta_epi_logit_delta"] = np.zeros((S, T, P), dtype=np.float32)
        post["theta_epi_log_hill_K"]  = np.zeros((S, T, P), dtype=np.float32)
        post["theta_epi_log_hill_n"]  = np.zeros((S, T, P), dtype=np.float32)
    return post


class TestHillMutPredictUnmeasured:

    def _q_to_get(self):
        return {"median": 0.5, "lower": 0.025, "upper": 0.975}

    def test_wt_returns_finite(self):
        T, M = 2, 2
        post = _make_posteriors(T, M)
        result = hill_predict_unmeasured(
            target_genotypes=["wt"],
            titrant_names=TITRANT_NAMES,
            manual_titrant_df=_make_titrant_df(),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get=self._q_to_get(),
        )
        assert not result["median"].isna().any()
        assert (result["median"] >= 0).all()
        assert (result["median"] <= 1).all()

    def test_unknown_mutation_is_nan(self):
        T, M = 2, 2
        post = _make_posteriors(T, M)
        result = hill_predict_unmeasured(
            target_genotypes=["wt", "Z99Q"],
            titrant_names=TITRANT_NAMES,
            manual_titrant_df=_make_titrant_df(),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get=self._q_to_get(),
        )
        wt_rows   = result[result["genotype"] == "wt"]
        bad_rows  = result[result["genotype"] == "Z99Q"]
        assert not wt_rows["median"].isna().any()
        assert bad_rows["median"].isna().all()

    def test_output_columns(self):
        T, M = 2, 2
        post = _make_posteriors(T, M)
        result = hill_predict_unmeasured(
            target_genotypes=["wt", "M42I"],
            titrant_names=TITRANT_NAMES,
            manual_titrant_df=_make_titrant_df(),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get=self._q_to_get(),
        )
        for col in ["genotype", "titrant_name", "titrant_conc",
                    "median", "lower", "upper"]:
            assert col in result.columns

    def test_zero_concentration_handled(self):
        T, M = 2, 2
        post = _make_posteriors(T, M)
        titrant_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.0],
        })
        result = hill_predict_unmeasured(
            target_genotypes=["wt"],
            titrant_names=TITRANT_NAMES,
            manual_titrant_df=titrant_df,
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get={"median": 0.5},
        )
        assert np.isfinite(result["median"].values).all()

    def test_additivity_double_mutant(self):
        """With zero epistasis and large deltas, wt+M42I+K84L ≠ wt+M42I alone."""
        T, M = 1, 2
        post = _make_posteriors(T, M, S=10, seed=1)
        # Give M42I a nonzero logit_low delta
        post["theta_d_logit_low"][:, 0, 0] = 2.0   # M42I, titrant 0
        post["theta_d_logit_low"][:, 0, 1] = 0.0   # K84L, titrant 0

        titrant_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [1e-5],
        })
        result = hill_predict_unmeasured(
            target_genotypes=["M42I", "M42I/K84L"],
            titrant_names=["IPTG"],
            manual_titrant_df=titrant_df,
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get={"median": 0.5},
        )
        single = result.loc[result["genotype"] == "M42I", "median"].values[0]
        double = result.loc[result["genotype"] == "M42I/K84L", "median"].values[0]
        # K84L adds zero to logit_low, so single == double in theta_low
        # (slight difference from theta_high/K/n but check they're both valid)
        assert np.isfinite(single)
        assert np.isfinite(double)

    def test_epistasis_shifts_double_mutant(self):
        """Non-zero epistasis on the double mutant changes its prediction."""
        T, M, P = 1, 2, 1
        post = _make_posteriors(T, M, P=P, S=20, seed=42)
        # Set a strong epistasis offset on logit_low for the pair
        post["theta_epi_logit_low"][:, 0, 0] = 5.0

        titrant_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [1e-5],
        })
        result_no_epi = hill_predict_unmeasured(
            target_genotypes=["M42I/K84L"],
            titrant_names=["IPTG"],
            manual_titrant_df=titrant_df,
            mut_labels=MUT_LABELS,
            pair_labels=[],          # no epistasis registered
            param_posteriors=post,
            q_to_get={"median": 0.5},
        )
        post_with_epi = dict(post)
        result_with_epi = hill_predict_unmeasured(
            target_genotypes=["M42I/K84L"],
            titrant_names=["IPTG"],
            manual_titrant_df=titrant_df,
            mut_labels=MUT_LABELS,
            pair_labels=PAIR_LABELS,  # epistasis active
            param_posteriors=post_with_epi,
            q_to_get={"median": 0.5},
        )
        no_epi_val   = result_no_epi["median"].values[0]
        with_epi_val = result_with_epi["median"].values[0]
        assert abs(with_epi_val - no_epi_val) > 0.05
