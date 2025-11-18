
import pytest
import numpy as np

# Import the main function and its helpers for testing
from tfscreen.analysis.independent.model_pre_growth import (
    model_pre_growth, 
    _process_dk_geno, 
    _process_lnA0,
)

# --- Tests for Helper Functions ---

def test_process_dk_geno_subset_averaging():
    """
    Tests the core logic of averaging dk_geno over a masked subset and
    broadcasting the result to the entire group.
    """
    k_est = np.array([1.2, 1.4, 1.9,   2.1, 2.8])
    k_wt = 1.0
    # Initial dk:      [0.2, 0.4, 0.9,   1.1, 1.8]
    groups = np.array([  0,   0,   0,     1,   1])
    mask = np.array(  [True, True, False, True, False])

    # Expected logic:
    # Group 0 mean is calculated from members 0 and 1: (0.2 + 0.4) / 2 = 0.3
    # Group 1 mean is calculated from member 3: 1.1 / 1 = 1.1
    # Result is broadcast to all members of original groups.
    expected_dk_geno = np.array([0.3, 0.3, 0.3, 1.1, 1.1])
    
    result = _process_dk_geno(k_est, k_wt, groups, mask)
    np.testing.assert_allclose(result, expected_dk_geno)

def test_process_lnA0_no_groups():
    """
    Tests _process_lnA0 when no grouping is applied.
    """
    lnA0_est = np.array([10.0, 12.0])
    lnA0_std = np.array([0.1, 0.2])
    kt_pre = np.array([2.0, 2.0])

    lnA0_pre, out_est, out_std = _process_lnA0(lnA0_est, lnA0_std, kt_pre, None)
    
    np.testing.assert_allclose(lnA0_pre, [8.0, 10.0])
    np.testing.assert_allclose(out_est, lnA0_est) # Should be unchanged
    np.testing.assert_allclose(out_std, lnA0_std) # Should be unchanged

def test_process_lnA0_with_groups_and_error_propagation():
    """
    Tests _process_lnA0 with grouping and error propagation.
    """
    # Two groups, [0,0] and [1,1,1]
    lnA0_est = np.array([10.0, 12.0,  20.0, 22.0, 24.0])
    lnA0_std = np.array([0.1, 0.1,   0.2, 0.2, 0.2]) # Original fit error
    kt_pre = np.array([2.0, 2.0,   3.0, 3.0, 3.0])
    groups = np.array([0, 0,     1, 1, 1])

    expected_pre = np.array([9.0, 9.0, 19.0, 19.0, 19.0])
    expected_est = np.array([11.0, 11.0, 22.0, 22.0, 22.0])
    expected_std = np.array([1.00498756, 1.00498756, 1.17189337, 1.17189337, 1.17189337])

    lnA0_pre, out_est, out_std = _process_lnA0(lnA0_est, lnA0_std, kt_pre, groups)

    np.testing.assert_allclose(lnA0_pre, expected_pre)
    np.testing.assert_allclose(out_est, expected_est)
    np.testing.assert_allclose(out_std, expected_std, rtol=1e-6)


# --- Tests for Main Orchestrating Function ---

@pytest.fixture
def base_data():
    """Provides a base set of inputs for testing the main function."""
    return {
        "k_est": np.array([1.2, 1.4, 2.5]),
        "lnA0_est": np.array([10, 11, 20]),
        "lnA0_std": np.array([0.1, 0.1, 0.2]),
        "k_wt": 1.0,
        "t_pre": 5.0
    }

def test_model_pre_growth_no_groups(base_data):
    """Tests the simplest case with no optional grouping."""
    dk_geno, lnA0_pre, lnA0_est, lnA0_std = model_pre_growth(**base_data)
    
    # dk_geno = k_est - k_wt
    np.testing.assert_allclose(dk_geno, [0.2, 0.4, 1.5])
    # k_pre = k_wt; kt_pre = 1.0 * 5.0 = 5.0
    # lnA0_pre = lnA0_est - kt_pre
    np.testing.assert_allclose(lnA0_pre, [5.0, 6.0, 15.0])
    # lnA0_est and lnA0_std should be unchanged
    np.testing.assert_allclose(lnA0_est, base_data["lnA0_est"])
    np.testing.assert_allclose(lnA0_std, base_data["lnA0_std"])
    
def test_model_pre_growth_dk_groups_only(base_data):
    """Tests averaging dk_geno over the whole group (now with a mask)."""
    groups = np.array([0, 0, 1])
    
    # FIX: Provide a mask that is True for all elements to test the 
    # logic of averaging over an entire group.
    mask = np.array([True, True, True])

    # Expected: dk_geno for group 0 is mean(0.2, 0.4) = 0.3
    expected_dk_geno = np.array([0.3, 0.3, 1.5])

    # The function call now includes the mask, satisfying the validation rule.
    dk_geno, _, _, _ = model_pre_growth(**base_data, 
                                        dk_geno_groups=groups, 
                                        dk_geno_mask=mask)
    
    np.testing.assert_allclose(dk_geno, expected_dk_geno)

def test_model_pre_growth_dk_groups_and_mask(base_data):
    """Tests averaging dk_geno over a masked subset."""
    groups = np.array([0, 0, 1])
    mask = np.array([True, False, True]) # Use only first sample for group 0 mean

    # Expected: dk_geno for group 0 is mean(0.2) = 0.2
    expected_dk_geno = np.array([0.2, 0.2, 1.5])

    dk_geno, _, _, _ = model_pre_growth(**base_data, dk_geno_groups=groups, dk_geno_mask=mask)
    np.testing.assert_allclose(dk_geno, expected_dk_geno)
    
@pytest.mark.parametrize(
    "kwargs, match_string",
    [
        ({"k_est": np.array([[1],[2]])}, "must be 1D arrays"),
        ({"dk_geno_mask": np.array([True, False])}, "must be a 1D boolean array"), # Wrong length
        ({"dk_geno_mask": np.array([1, 0, 1])}, "must be a 1D boolean array"), # Wrong dtype
        (
            {"dk_geno_groups": np.array([0, 0, 1])}, # mask is missing
            "must be specified together"
        ),
        (
            {"dk_geno_mask": np.array([True, True, False])}, # groups are missing
            "must be specified together"
        )
    ]
)
def test_model_pre_growth_validation(base_data, kwargs, match_string):
    """Tests the validation logic in the main function."""
    # Update base data with the invalid argument
    base_data.update(kwargs)
    with pytest.raises(ValueError, match=match_string):
        model_pre_growth(**base_data)

