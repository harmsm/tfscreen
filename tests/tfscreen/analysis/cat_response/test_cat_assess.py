"""
Tests for the post-hoc assessment helpers: omnibus chi-square, per-point
significance, region-of-practical-equivalence classification, and BH FDR.
"""
import numpy as np
import pytest
from scipy.stats import chi2, norm

from tfscreen.analysis.cat_response.cat_assess import (
    assess_best_model,
    compute_rope,
    classify_equiv,
    benjamini_hochberg,
    residual_runs_p,
    residual_autocorr,
    goodness_of_fit_p,
    _omnibus_chi2,
    _nonzero_chi2,
)


class TestNonzeroChi2:
    def test_signal_is_significant(self):
        stat, df, p = _nonzero_chi2(np.array([1.0, 2.0, 3.0]),
                                    np.array([0.1, 0.1, 0.1]))
        assert df == 3 and stat > 100 and p < 1e-6

    def test_consistent_with_zero(self):
        stat, df, p = _nonzero_chi2(np.array([0.1, -0.2, 0.05]),
                                    np.array([2.0, 3.0, 1.0]))
        assert df == 3 and p > 0.5

    def test_drops_bad_points(self):
        stat, df, p = _nonzero_chi2(np.array([1.0, np.nan, 3.0]),
                                    np.array([0.1, 0.1, 0.0]))
        assert df == 1                       # only the first point is usable

    def test_no_usable_points_is_nan(self):
        stat, df, p = _nonzero_chi2(np.array([1.0, 2.0]),
                                    np.array([0.0, -1.0]))
        assert df == 0 and np.isnan(stat) and np.isnan(p)


# --- residual autocorrelation (shape gate) -----------------------------------

class TestResidualAutocorr:
    def test_smooth_structure_positive_and_significant(self):
        # A smooth ramp = strong positive autocorrelation -> small p.
        resid = np.array([-3, -2, -1, 0, 1, 2, 3, 4], dtype=float)
        ac, p = residual_autocorr(resid)
        assert ac > 0.5
        assert p < 0.05

    def test_alternating_not_positive(self):
        resid = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        ac, p = residual_autocorr(resid)
        assert ac < 0.0        # negative autocorrelation
        assert p > 0.5         # not flagged as positive structure

    def test_too_few_points_is_nan(self):
        ac, p = residual_autocorr(np.array([1.0, -1.0, 1.0]))
        assert np.isnan(ac) and np.isnan(p)

    def test_all_zero_is_nan(self):
        ac, p = residual_autocorr(np.zeros(6))
        assert np.isnan(ac) and np.isnan(p)


# --- residual runs test ------------------------------------------------------

class TestResidualRunsP:
    def test_clustered_residuals_flagged(self):
        """Same-sign clustering (systematic misfit) -> small p."""
        resid = np.array([-3, -2, -1, -0.5, 1, 2, 3, 4], dtype=float)
        assert residual_runs_p(resid) < 0.05

    def test_random_looking_residuals_not_flagged(self):
        """A well-mixed sign sequence -> large p (adequate)."""
        resid = np.array([1, -1, 2, -2, 1, -1, 2, -2], dtype=float)
        assert residual_runs_p(resid) > 0.5

    def test_over_dispersion_not_flagged(self):
        """One-sided: too-many-runs (alternating) is not a shape error."""
        resid = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        assert residual_runs_p(resid) > 0.05

    def test_too_few_points_is_nan(self):
        assert np.isnan(residual_runs_p(np.array([1.0, -1.0, 1.0])))

    def test_all_one_sign_is_nan(self):
        assert np.isnan(residual_runs_p(np.array([1, 2, 3, 4, 5.0])))

    def test_zeros_and_nonfinite_dropped(self):
        # Zeros/NaN removed; remaining 4 clustered residuals still assessable.
        resid = np.array([0.0, np.nan, -2, -1, 1, 2], dtype=float)
        assert np.isfinite(residual_runs_p(resid))


# --- goodness of fit ---------------------------------------------------------

class TestGoodnessOfFitP:
    def test_matches_chi2_sf(self):
        assert goodness_of_fit_p(10.0, 12, 2) == pytest.approx(chi2.sf(10.0, 10))

    def test_large_chi2_small_p(self):
        assert goodness_of_fit_p(100.0, 12, 2) < 0.001

    def test_nonpositive_df_is_nan(self):
        assert np.isnan(goodness_of_fit_p(5.0, 4, 4))
        assert np.isnan(goodness_of_fit_p(5.0, 3, 4))


def _linear(params, x):
    m, b = params
    return m * x + b


# --- omnibus -----------------------------------------------------------------

class TestOmnibus:

    def test_identity_cov_is_sum_of_squared_z(self):
        """With Sigma = I, W = sum(y_est^2) and df = len(y_est)."""
        y_est = np.array([1.0, -2.0, 0.5])
        W, df, p = _omnibus_chi2(y_est, np.eye(3))
        assert df == 3
        assert W == pytest.approx(np.sum(y_est ** 2))
        assert p == pytest.approx(chi2.sf(W, 3))

    def test_rank_deficient_uses_pseudo_inverse_and_rank_df(self):
        """A singular covariance yields df = rank, not the matrix dimension."""
        # Rank-1 covariance (two identical predicted points).
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        y_est = np.array([2.0, 2.0])
        W, df, p = _omnibus_chi2(y_est, cov)
        assert df == 1
        assert np.isfinite(W)
        assert np.isfinite(p)

    def test_nan_cov_returns_nan(self):
        W, df, p = _omnibus_chi2(np.array([1.0, 2.0]),
                                 np.full((2, 2), np.nan))
        assert np.isnan(W) and df == 0 and np.isnan(p)

    def test_assess_best_model_flags_and_rollup(self):
        # Observed points [1,2,3] with tight errors -> all clearly nonzero, and
        # the data-based nonzero test is significant. The fitted line also gives
        # a (reported) model omnibus.
        x = np.array([1.0, 2.0, 3.0])
        params = np.array([1.0, 0.0])
        cov = np.diag([1e-4, 1e-4])
        y_obs = np.array([1.0, 2.0, 3.0])
        y_std = np.array([0.01, 0.01, 0.01])
        per_point, rollup = assess_best_model(_linear, params, cov, x,
                                              y_obs, y_std)

        assert per_point["y_model"] == pytest.approx([1.0, 2.0, 3.0])
        # Per-point z-test now reads the observed data.
        assert per_point["z"] == pytest.approx(y_obs / y_std)
        assert np.all(per_point["sig_nonzero"])
        assert rollup["n_nonzero"] == 3
        assert rollup["any_nonzero"] is True
        assert rollup["nonzero_df"] == 3          # data test: one df per point
        assert rollup["nonzero_p"] < 1e-6
        assert rollup["omnibus_df"] == 2          # model test: two free params
        assert rollup["omnibus_p"] < 1e-6

    def test_assess_best_model_data_beats_overconfident_model(self):
        # The reported failure mode: observed error bars overlap zero, but the
        # fitted curve is confident. The DATA test must not call it nonzero.
        x = np.array([1.0, 2.0, 3.0])
        params = np.array([1.0, 0.0])
        cov = np.diag([1e-6, 1e-6])          # overconfident fit
        y_obs = np.array([1.0, 2.0, 3.0])
        y_std = np.array([50.0, 50.0, 50.0])  # huge observed errors
        per_point, rollup = assess_best_model(_linear, params, cov, x,
                                              y_obs, y_std)
        assert rollup["n_nonzero"] == 0            # no observed point clears 0
        assert rollup["nonzero_p"] > 0.5           # data: consistent with zero
        assert rollup["omnibus_p"] < 1e-6          # model: overconfident nonzero

    def test_assess_best_model_nan_cov(self):
        # NaN fit covariance -> model omnibus NaN, but the data-based tests still
        # run on the observed points.
        x = np.array([1.0, 2.0, 3.0])
        y_obs = np.array([0.0, 0.0, 0.0])
        y_std = np.array([0.1, 0.1, 0.1])
        per_point, rollup = assess_best_model(
            _linear, np.array([1.0, 0.0]), np.full((2, 2), np.nan), x,
            y_obs, y_std
        )
        assert np.all(~per_point["sig_nonzero"])
        assert np.isnan(rollup["omnibus_p"])
        assert np.isfinite(rollup["nonzero_p"])
        assert rollup["n_nonzero"] == 0


# --- equivalence -------------------------------------------------------------

class TestClassifyEquiv:

    def test_ci_inside_region_is_equiv(self):
        # |y| + z*se = 0.1 + 1.96*0.01 ~ 0.12 < rope=0.5 -> equiv.
        equiv = classify_equiv(np.array([0.1]), np.array([0.01]), rope_cutoff=0.5)
        assert equiv[0]

    def test_ci_straddles_boundary_not_equiv(self):
        # 0.4 + 1.96*0.1 ~ 0.596 > 0.5 -> not equiv (too wide / too far).
        equiv = classify_equiv(np.array([0.4]), np.array([0.1]), rope_cutoff=0.5)
        assert not equiv[0]

    def test_far_from_zero_not_equiv(self):
        equiv = classify_equiv(np.array([5.0]), np.array([0.01]), rope_cutoff=0.5)
        assert not equiv[0]

    def test_nan_std_and_bad_rope(self):
        assert not classify_equiv(np.array([0.0]), np.array([np.nan]),
                                  rope_cutoff=0.5)[0]
        assert not classify_equiv(np.array([0.0]), np.array([0.01]),
                                  rope_cutoff=np.nan)[0]

    def test_boundary_uses_alpha(self):
        # Larger alpha -> narrower CI -> easier to be equivalent.
        y, s, r = np.array([0.3]), np.array([0.1]), 0.5
        # alpha=0.05 -> 0.3+1.96*0.1=0.496 < 0.5 -> equiv (just).
        assert classify_equiv(y, s, r, alpha=0.05)[0]


# --- compute_rope ------------------------------------------------------------

class TestComputeRope:

    def test_median_times_multiplier(self):
        stds = [0.1, 0.2, 0.3, np.nan]
        # median of finite [0.1,0.2,0.3] = 0.2 -> *2 = 0.4
        assert compute_rope(stds, rope_multiplier=2.0) == pytest.approx(0.4)

    def test_all_nan_returns_nan(self):
        assert np.isnan(compute_rope([np.nan, np.nan]))

    def test_empty_returns_nan(self):
        assert np.isnan(compute_rope([]))


# --- benjamini-hochberg ------------------------------------------------------

class TestBenjaminiHochberg:

    def test_matches_manual_step_up(self):
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        q = benjamini_hochberg(p)
        m = 5
        expected = p * m / np.arange(1, m + 1)
        expected = np.minimum.accumulate(expected[::-1])[::-1]
        assert np.allclose(q, expected)

    def test_monotone_nondecreasing_in_p(self):
        p = np.array([0.5, 0.001, 0.2, 0.04])
        q = benjamini_hochberg(p)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-12)

    def test_nan_passthrough_and_excluded(self):
        p = np.array([0.01, np.nan, 0.5])
        q = benjamini_hochberg(p)
        assert np.isnan(q[1])
        # m=2 finite tests, not 3.
        assert q[0] == pytest.approx(min(0.01 * 2 / 1, q[2]))

    def test_clipped_to_one(self):
        q = benjamini_hochberg(np.array([0.9, 0.95]))
        assert np.all(q <= 1.0)

    def test_all_nan(self):
        q = benjamini_hochberg(np.array([np.nan, np.nan]))
        assert np.all(np.isnan(q))
