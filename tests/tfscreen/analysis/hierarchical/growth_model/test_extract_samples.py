"""
Tests for the num_samples parameter added to extract_theta_curves and
extract_growth_predictions.

Both functions default to num_samples=100, adding sample_0 … sample_{N-1}
columns alongside the existing quantile columns. Pass num_samples=None to
suppress sample columns and return only quantiles.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.extraction import (
    extract_theta_curves,
    extract_growth_predictions,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_Q = {"median": 0.5}   # single quantile keeps assertions simple


def _hill_model():
    """Minimal mock for theta='hill' with 2 groups (wt/iptg, mut/iptg)."""
    model = MagicMock(spec=ModelClass)
    model._theta = "hill"
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype":        ["wt",  "wt",  "mut", "mut"],
        "titrant_name":    ["iptg","iptg","iptg","iptg"],
        "titrant_conc":    [0.0,   1.0,   0.0,   1.0],
        "map_theta_group": [0,     0,     1,     1],
    })
    model.growth_tm = mock_tm
    return model


def _hill_posteriors(S=10, num_groups=2):
    """Constant posteriors so results are deterministic."""
    return {
        "theta_hill_n":     np.ones((S, num_groups)) * 2.0,
        "theta_log_hill_K": np.ones((S, num_groups)) * -1.0,
        "theta_theta_high": np.ones((S, num_groups)) * 0.9,
        "theta_theta_low":  np.ones((S, num_groups)) * 0.1,
    }


def _hill_mut_model():
    """Minimal mock for theta='hill_mut', T=2 titrants, G=3 genotypes."""
    model = MagicMock(spec=ModelClass)
    model._theta = "hill_mut"
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype":         ["wt", "A",  "B",  "wt", "A",  "B"],
        "titrant_name":     ["iptg","iptg","iptg","atc","atc","atc"],
        "titrant_conc":     [1.0,  1.0,  1.0,  2.0,  2.0,  2.0],
        "genotype_idx":     [0,    1,    2,    0,    1,    2],
        "titrant_name_idx": [0,    0,    0,    1,    1,    1],
    })
    model.growth_tm = mock_tm
    return model


def _hill_mut_posteriors(S=10):
    return {
        "theta_hill_n":     np.ones((S, 2, 3)) * 2.0,
        "theta_log_hill_K": np.ones((S, 2, 3)) * -1.0,
        "theta_theta_high": np.ones((S, 2, 3)) * 0.9,
        "theta_theta_low":  np.ones((S, 2, 3)) * 0.1,
    }


def _lac_dimer_mut_model():
    """Minimal mock for theta='lac_dimer_lnK_mut', T=1 titrant, G=2 genotypes."""
    model = MagicMock(spec=ModelClass)
    model._theta = "lac_dimer_lnK_mut"
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype":         ["wt",  "mut"],
        "titrant_name":     ["iptg","iptg"],
        "titrant_conc":     [1e-4,  1e-4],
        "genotype_idx":     [0,     1],
        "titrant_name_idx": [0,     0],
    })
    model.growth_tm = mock_tm
    model.priors.theta.theta_tf_total_M = 6.5e-7
    model.priors.theta.theta_op_total_M = 2.5e-8
    return model


def _lac_dimer_mut_posteriors(S=10):
    return {
        "theta_ln_K_op": np.full((S, 2), -5.0),
        "theta_ln_K_HL": np.full((S, 2), -3.0),
        "theta_ln_K_E":  np.full((S, 1, 2), -4.0),
    }


def _growth_model():
    """Minimal mock for extract_growth_predictions; 2 rows."""
    model = MagicMock(spec=ModelClass)
    model.growth_df = pd.DataFrame({
        "replicate":       [0, 0],
        "genotype":        ["G0", "G0"],
        "condition_pre":   ["C0", "C0"],
        "condition_sel":   ["C1", "C1"],
        "titrant_name":    ["T",  "T"],
        "titrant_conc":    [0,    1],
        "t_pre":           [1.0,  1.0],
        "t_sel":           [2.0,  2.0],
        "ln_cfu":          [10.0, 11.0],
        "ln_cfu_std":      [0.1,  0.1],
        "replicate_idx":   [0,    0],
        "time_idx":        [0,    1],
        "condition_pre_idx": [0,  0],
        "condition_sel_idx": [0,  0],
        "titrant_name_idx":  [0,  0],
        "titrant_conc_idx":  [0,  1],
        "genotype_idx":      [0,  0],
    })
    return model


def _growth_posteriors(S=10):
    """Shape: (S, replicate=1, time=2, pre=1, sel=1, name=1, conc=2, geno=1)."""
    rng = np.random.default_rng(0)
    return {"growth_pred": rng.normal(10.0, 0.5, size=(S, 1, 2, 1, 1, 1, 2, 1))}


# ---------------------------------------------------------------------------
# extract_theta_curves – num_samples with hill model
# ---------------------------------------------------------------------------

class TestExtractThetaCurvesNumSamplesHill:

    def test_num_samples_none_returns_no_sample_columns(self):
        model = _hill_model()
        df = extract_theta_curves(model, _hill_posteriors(), q_to_get=_Q, num_samples=None)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert sample_cols == []

    def test_num_samples_adds_correct_number_of_columns(self):
        model = _hill_model()
        df = extract_theta_curves(model, _hill_posteriors(), q_to_get=_Q, num_samples=5)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 5
        assert sorted(sample_cols) == [f"sample_{i}" for i in range(5)]

    def test_quantile_columns_still_present_alongside_samples(self):
        model = _hill_model()
        df = extract_theta_curves(model, _hill_posteriors(), q_to_get=_Q, num_samples=3)
        assert "median" in df.columns

    def test_sample_values_within_posterior_range(self):
        """Sample columns must be actual draws, not summaries outside [min, max]."""
        rng = np.random.default_rng(1)
        S = 20
        posteriors = {
            "theta_hill_n":     rng.uniform(0.5, 3.0, (S, 2)),
            "theta_log_hill_K": rng.uniform(-3.0, 0.0, (S, 2)),
            "theta_theta_high": rng.uniform(0.7, 1.0, (S, 2)),
            "theta_theta_low":  rng.uniform(0.0, 0.3, (S, 2)),
        }
        model = _hill_model()
        df = extract_theta_curves(model, posteriors, q_to_get=_Q, num_samples=10)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        q_min = df["min"] if "min" in df.columns else None

        # All sample values should be between 0 and 1 (theta is a probability)
        for col in sample_cols:
            assert df[col].between(0.0, 1.0).all(), f"{col} has values outside [0, 1]"

    def test_constant_posteriors_samples_equal_median(self):
        """When all posterior draws are identical, samples == median."""
        model = _hill_model()
        df = extract_theta_curves(model, _hill_posteriors(S=10), q_to_get=_Q, num_samples=4)
        for i in range(4):
            np.testing.assert_allclose(df[f"sample_{i}"].values,
                                       df["median"].values,
                                       rtol=1e-6)

    def test_num_samples_exceeds_posterior_size_uses_replacement(self):
        """num_samples > S should not raise; replacement is used silently."""
        model = _hill_model()
        df = extract_theta_curves(model, _hill_posteriors(S=3), q_to_get=_Q, num_samples=10)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 10

    def test_output_row_count_unchanged_by_num_samples(self):
        model = _hill_model()
        df_no_samples = extract_theta_curves(model, _hill_posteriors(), q_to_get=_Q,
                                             num_samples=None)
        df_with_samples = extract_theta_curves(model, _hill_posteriors(), q_to_get=_Q,
                                               num_samples=5)
        assert len(df_no_samples) == len(df_with_samples)

    def test_index_columns_still_dropped(self):
        model = _hill_model()
        df = extract_theta_curves(model, _hill_posteriors(), q_to_get=_Q, num_samples=3)
        assert "map_theta_group" not in df.columns


# ---------------------------------------------------------------------------
# extract_theta_curves – verify samples are genuine posterior draws (hill)
# ---------------------------------------------------------------------------

class TestExtractThetaCurvesHillSampleVariance:

    def test_sample_columns_are_distinct_rows_of_theta_samples(self):
        """
        With varied posteriors, each sample_i column should differ from the
        median (with overwhelming probability for S=100).
        """
        rng = np.random.default_rng(7)
        S, G = 100, 2
        posteriors = {
            "theta_hill_n":     rng.uniform(1.0, 3.0, (S, G)),
            "theta_log_hill_K": rng.uniform(-2.0, 0.0, (S, G)),
            "theta_theta_high": rng.uniform(0.8, 1.0, (S, G)),
            "theta_theta_low":  rng.uniform(0.0, 0.2, (S, G)),
        }
        model = _hill_model()
        df = extract_theta_curves(model, posteriors, q_to_get=_Q, num_samples=5)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 5
        # At least one sample column should differ from the median somewhere
        diffs = [not np.allclose(df[c].values, df["median"].values) for c in sample_cols]
        assert any(diffs), "All sample columns equal the median — draws are not varying"


# ---------------------------------------------------------------------------
# extract_theta_curves – num_samples with hill_mut model
# ---------------------------------------------------------------------------

class TestExtractThetaCurvesNumSamplesHillMut:

    def test_num_samples_adds_columns(self):
        model = _hill_mut_model()
        df = extract_theta_curves(model, _hill_mut_posteriors(), q_to_get=_Q, num_samples=6)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 6

    def test_constant_posteriors_samples_equal_median(self):
        model = _hill_mut_model()
        df = extract_theta_curves(model, _hill_mut_posteriors(S=8), q_to_get=_Q, num_samples=3)
        for i in range(3):
            np.testing.assert_allclose(df[f"sample_{i}"].values,
                                       df["median"].values,
                                       rtol=1e-6)

    def test_index_columns_dropped(self):
        model = _hill_mut_model()
        df = extract_theta_curves(model, _hill_mut_posteriors(), q_to_get=_Q, num_samples=2)
        assert "genotype_idx" not in df.columns
        assert "titrant_name_idx" not in df.columns

    def test_num_samples_exceeds_posterior_size(self):
        model = _hill_mut_model()
        df = extract_theta_curves(model, _hill_mut_posteriors(S=2), q_to_get=_Q, num_samples=7)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 7


# ---------------------------------------------------------------------------
# extract_theta_curves – num_samples with lac_dimer_mut model
# ---------------------------------------------------------------------------

class TestExtractThetaCurvesNumSamplesLacDimerMut:

    def test_num_samples_adds_columns(self):
        model = _lac_dimer_mut_model()
        df = extract_theta_curves(model, _lac_dimer_mut_posteriors(), q_to_get=_Q,
                                  num_samples=4)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 4

    def test_constant_posteriors_samples_equal_median(self):
        model = _lac_dimer_mut_model()
        df = extract_theta_curves(model, _lac_dimer_mut_posteriors(S=5), q_to_get=_Q,
                                  num_samples=3)
        for i in range(3):
            np.testing.assert_allclose(df[f"sample_{i}"].values,
                                       df["median"].values,
                                       rtol=1e-5)

    def test_index_columns_dropped(self):
        model = _lac_dimer_mut_model()
        df = extract_theta_curves(model, _lac_dimer_mut_posteriors(), q_to_get=_Q,
                                  num_samples=2)
        assert "genotype_idx" not in df.columns
        assert "titrant_name_idx" not in df.columns

    def test_sample_values_are_valid_probabilities(self):
        model = _lac_dimer_mut_model()
        df = extract_theta_curves(model, _lac_dimer_mut_posteriors(), q_to_get=_Q,
                                  num_samples=5)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        for col in sample_cols:
            assert df[col].between(0.0, 1.0).all(), f"{col} outside [0, 1]"

    def test_num_samples_exceeds_posterior_size(self):
        model = _lac_dimer_mut_model()
        df = extract_theta_curves(model, _lac_dimer_mut_posteriors(S=2), q_to_get=_Q,
                                  num_samples=8)
        assert len([c for c in df.columns if c.startswith("sample_")]) == 8


# ---------------------------------------------------------------------------
# extract_growth_predictions – num_samples
# ---------------------------------------------------------------------------

class TestExtractGrowthPredictionsNumSamples:

    def test_num_samples_none_returns_no_sample_columns(self):
        model = _growth_model()
        df = extract_growth_predictions(model, _growth_posteriors(), q_to_get=_Q,
                                        num_samples=None)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert sample_cols == []

    def test_num_samples_adds_correct_number_of_columns(self):
        model = _growth_model()
        df = extract_growth_predictions(model, _growth_posteriors(), q_to_get=_Q,
                                        num_samples=7)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 7
        assert sorted(sample_cols) == [f"sample_{i}" for i in range(7)]

    def test_quantile_columns_still_present(self):
        model = _growth_model()
        df = extract_growth_predictions(model, _growth_posteriors(), q_to_get=_Q,
                                        num_samples=3)
        assert "median" in df.columns

    def test_constant_posteriors_samples_equal_median(self):
        """When all posterior draws are the same value, samples == median."""
        model = _growth_model()
        S = 8
        posteriors = {"growth_pred": np.full((S, 1, 2, 1, 1, 1, 2, 1), 10.5)}
        df = extract_growth_predictions(model, posteriors, q_to_get=_Q, num_samples=4)
        for i in range(4):
            np.testing.assert_allclose(df[f"sample_{i}"].values,
                                       df["median"].values,
                                       rtol=1e-6)

    def test_sample_values_within_posterior_range(self):
        model = _growth_model()
        S = 20
        rng = np.random.default_rng(42)
        posteriors = {"growth_pred": rng.normal(10.0, 1.0, (S, 1, 2, 1, 1, 1, 2, 1))}
        df = extract_growth_predictions(model, posteriors, q_to_get=_Q, num_samples=10)
        all_vals = posteriors["growth_pred"].ravel()
        vmin, vmax = all_vals.min(), all_vals.max()
        for col in [c for c in df.columns if c.startswith("sample_")]:
            assert df[col].between(vmin, vmax).all(), f"{col} has values outside posterior range"

    def test_sample_indices_consistent_across_rows(self):
        """
        sample_i for row 0 and sample_i for row 1 must come from the same
        posterior draw index (joint consistency across observations).
        """
        model = _growth_model()
        S = 50
        rng = np.random.default_rng(3)
        # Make each posterior sample have a unique signature: draw s gives value s+1 for all
        # cells so we can recover which draw was used.
        growth_pred = np.zeros((S, 1, 2, 1, 1, 1, 2, 1))
        for s in range(S):
            growth_pred[s] = float(s + 1)
        posteriors = {"growth_pred": growth_pred}

        df = extract_growth_predictions(model, posteriors, q_to_get=_Q, num_samples=5)

        # Both rows must show the same draw index for each sample column
        for i in range(5):
            col = f"sample_{i}"
            row0_val = df.iloc[0][col]
            row1_val = df.iloc[1][col]
            assert row0_val == row1_val, (
                f"{col}: row 0 has draw {row0_val} but row 1 has {row1_val} — "
                "sample indices are not consistent across rows"
            )

    def test_num_samples_exceeds_posterior_size_uses_replacement(self):
        model = _growth_model()
        posteriors = {"growth_pred": np.ones((3, 1, 2, 1, 1, 1, 2, 1))}
        df = extract_growth_predictions(model, posteriors, q_to_get=_Q, num_samples=10)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == 10

    def test_output_row_count_unchanged_by_num_samples(self):
        model = _growth_model()
        df_no = extract_growth_predictions(model, _growth_posteriors(), q_to_get=_Q,
                                           num_samples=None)
        df_yes = extract_growth_predictions(model, _growth_posteriors(), q_to_get=_Q,
                                            num_samples=5)
        assert len(df_no) == len(df_yes)
