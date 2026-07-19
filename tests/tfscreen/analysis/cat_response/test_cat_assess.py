"""
Tests for the post-hoc assessment helpers: omnibus chi-square, per-point
significance, region-of-practical-equivalence classification, and BH FDR.
"""
import numpy as np
import pytest
from scipy.stats import chi2, norm

from tfscreen.analysis.cat_response.cat_assess import (
    assess_best_model,
    compute_delta,
    classify_equiv,
    benjamini_hochberg,
    _omnibus_chi2,
)


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
        # y = 1*x + 0 at x=[1,2,3] -> y_model = [1,2,3], all clearly nonzero.
        x = np.array([1.0, 2.0, 3.0])
        params = np.array([1.0, 0.0])
        cov = np.diag([1e-4, 1e-4])   # tiny -> small pred se -> significant
        per_point, rollup = assess_best_model(_linear, params, cov, x)

        assert per_point["y_model"] == pytest.approx([1.0, 2.0, 3.0])
        assert np.all(per_point["sig_nonzero"])
        assert np.all(per_point["direction"] == 1)
        assert rollup["n_nonzero"] == 3
        assert rollup["any_nonzero"] is True
        assert rollup["omnibus_df"] == 2          # two free params
        assert rollup["omnibus_p"] < 1e-6

    def test_assess_best_model_nan_cov(self):
        x = np.array([1.0, 2.0, 3.0])
        per_point, rollup = assess_best_model(
            _linear, np.array([1.0, 0.0]), np.full((2, 2), np.nan), x
        )
        assert np.all(~per_point["sig_nonzero"])
        assert np.isnan(rollup["omnibus_p"])
        assert rollup["n_nonzero"] == 0


# --- equivalence -------------------------------------------------------------

class TestClassifyEquiv:

    def test_ci_inside_region_is_equiv(self):
        # |y| + z*se = 0.1 + 1.96*0.01 ~ 0.12 < delta=0.5 -> equiv.
        equiv = classify_equiv(np.array([0.1]), np.array([0.01]), delta=0.5)
        assert equiv[0]

    def test_ci_straddles_boundary_not_equiv(self):
        # 0.4 + 1.96*0.1 ~ 0.596 > 0.5 -> not equiv (too wide / too far).
        equiv = classify_equiv(np.array([0.4]), np.array([0.1]), delta=0.5)
        assert not equiv[0]

    def test_far_from_zero_not_equiv(self):
        equiv = classify_equiv(np.array([5.0]), np.array([0.01]), delta=0.5)
        assert not equiv[0]

    def test_nan_std_and_bad_delta(self):
        assert not classify_equiv(np.array([0.0]), np.array([np.nan]),
                                  delta=0.5)[0]
        assert not classify_equiv(np.array([0.0]), np.array([0.01]),
                                  delta=np.nan)[0]

    def test_boundary_uses_alpha(self):
        # Larger alpha -> narrower CI -> easier to be equivalent.
        y, s, d = np.array([0.3]), np.array([0.1]), 0.5
        # alpha=0.05 -> 0.3+1.96*0.1=0.496 < 0.5 -> equiv (just).
        assert classify_equiv(y, s, d, alpha=0.05)[0]


# --- compute_delta -----------------------------------------------------------

class TestComputeDelta:

    def test_median_times_c(self):
        stds = [0.1, 0.2, 0.3, np.nan]
        # median of finite [0.1,0.2,0.3] = 0.2 -> *2 = 0.4
        assert compute_delta(stds, delta_c=2.0) == pytest.approx(0.4)

    def test_all_nan_returns_nan(self):
        assert np.isnan(compute_delta([np.nan, np.nan]))

    def test_empty_returns_nan(self):
        assert np.isnan(compute_delta([]))


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
