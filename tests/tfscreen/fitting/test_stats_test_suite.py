
import pytest
import numpy as np
from tfscreen.fitting.stats_test_suite import stats_test_suite

def test_stats_test_suite_perfect_fit():
    param_real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    param_est = param_real.copy()
    param_std = np.array([0.1, 0.1, 0.1, 0.1, 0.1]) # Small std
    
    # Perfect fit
    res = stats_test_suite(param_est, param_std, param_real)
    
    assert res["pct_success"] == 1.0
    assert res["rmse"] == 0.0
    assert res["normalized_rmse"] == 0.0
    assert res["pearson_r"] == 1.0
    assert res["r_squared"] == 1.0
    assert res["mean_error"] == 0.0
    # In perfect fit, estimates == real. 
    # real >= estimate - 1.96*std ? real >= real - 1.96*std (True, since std > 0)
    # real <= estimate + 1.96*std ? real <= real + 1.96*std (True)
    assert res["coverage_prob"] == 1.0 
    
    # Residuals are all 0. Correlation undefined?
    # pearsonr of constant input issues warning and returns nan usually.
    # The function uses scipy.stats.pearsonr
    # Let's check what happens with 0 residuals.
    # diff = 0. pearsonr(0, real) might be NaN.
    # Let's add slight noise to avoid NaN if checking strictness
    
def test_stats_test_suite_with_noise():
    param_real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Add small bias
    param_est = param_real + 0.1
    # param_std big enough to cover bias (0.1 < 1.96 * 0.1) -> 0.1 < 0.196. Yes.
    param_std = np.ones(5) * 0.1
    
    res = stats_test_suite(param_est, param_std, param_real)
    
    assert res["pct_success"] == 1.0
    assert np.isclose(res["mean_error"], 0.1)
    # RMSE: sqrt(mean(0.1^2)) = 0.1
    assert np.isclose(res["rmse"], 0.1)
    assert np.isclose(res["coverage_prob"], 1.0) # Bias small enough to be covered
    
    # Signal range: 97.5 percentile of [1..5] - 2.5 percentile of [1..5]
    # np.percentile([1,2,3,4,5], [2.5, 97.5])
    # 2.5% ~ 1.1, 97.5% ~ 4.9. Range ~ 3.8
    # normalized_rmse = 0.1 / range.
    assert np.isfinite(res["normalized_rmse"])

def test_stats_test_suite_nans():
    param_real = np.array([1.0, 2.0, 3.0])
    param_est = np.array([1.0, np.nan, 3.0])
    param_std = np.array([0.1, 0.1, 0.1])
    
    res = stats_test_suite(param_est, param_std, param_real)
    
    assert res["pct_success"] == 2/3
    # Metrics should be calculated on the 2 valid points
    assert res["rmse"] == 0.0

def test_zero_signal_range():
    param_real = np.array([1.0, 1.0, 1.0])
    param_est = np.array([1.1, 1.1, 1.1])
    param_std = np.array([0.1, 0.1, 0.1])
    
    res = stats_test_suite(param_est, param_std, param_real)
    
    # Signal range is 0
    assert res["normalized_rmse"] == np.inf

import warnings
def test_het_breuschpagan_linalg_error(mocker):
    # Mock het_breuschpagan to raise LinAlgError
    mocker.patch("tfscreen.fitting.stats_test_suite.het_breuschpagan", side_effect=np.linalg.LinAlgError)
    
    param_real = np.array([1.0, 2.0, 3.0])
    param_est = np.array([1.1, 2.1, 3.1])
    param_std = np.array([0.1, 0.1, 0.1])
    
    with pytest.warns(UserWarning, match="het_breuschpagan test did not converge"):
         res = stats_test_suite(param_est, param_std, param_real)
    
    assert np.isnan(res["bp_p_value"])

