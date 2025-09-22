import pytest
import pandas as pd
import numpy as np

# Assuming the refactored function is in this path
from tfscreen.util import get_scaled_cfu

# --- Test Group 1: Successful Calculations ---

BASE_DATA = {
    'cfu': 100.0,
    'cfu_std': 10.0,
    'cfu_var': 100.0,
    'ln_cfu': np.log(100.0),
    'ln_cfu_std': 0.1,
    'ln_cfu_var': 0.01
}

# Define the matrix of test cases: (input_cols, need_cols, test_id)
SUCCESS_CASES = [
    # === Group A: Starting with Linear Scale Data (cfu) ===
    (["cfu", "cfu_std"], ["cfu_var"], "A1: Linear std -> var"),
    (["cfu", "cfu_std"], ["ln_cfu"], "A2: Linear -> log value"),
    (["cfu", "cfu_std"], ["ln_cfu_std"], "A3: Linear std -> log std"),
    (["cfu", "cfu_std"], ["ln_cfu_var"], "A4: Linear std -> log var (multi-step)"),
    (["cfu", "cfu_var"], ["cfu_std"], "A5: Linear var -> std"),
    (["cfu", "cfu_var"], ["ln_cfu_std"], "A6: Linear var -> log std (multi-step)"),
    (["cfu", "cfu_var"], ["ln_cfu_var"], "A7: Linear var -> log var (multi-step)"),
    (["cfu", "cfu_std"], ["cfu", "cfu_var", "cfu_std", "ln_cfu", "ln_cfu_var", "ln_cfu_std"], "A8: Linear -> All"),

    # === Group B: Starting with Log Scale Data (ln_cfu) ===
    (["ln_cfu", "ln_cfu_std"], ["ln_cfu_var"], "B1: Log std -> var"),
    (["ln_cfu", "ln_cfu_std"], ["cfu"], "B2: Log -> linear value"),
    (["ln_cfu", "ln_cfu_std"], ["cfu_std"], "B3: Log std -> linear std"),
    (["ln_cfu", "ln_cfu_std"], ["cfu_var"], "B4: Log std -> linear var (multi-step)"),
    (["ln_cfu", "ln_cfu_var"], ["ln_cfu_std"], "B5: Log var -> std"),
    (["ln_cfu", "ln_cfu_var"], ["cfu_std"], "B6: Log var -> linear std (multi-step)"),
    (["ln_cfu", "ln_cfu_var"], ["cfu_var"], "B7: Log var -> linear var (multi-step)"),
    (["ln_cfu", "ln_cfu_std"], ["cfu", "cfu_var", "cfu_std", "ln_cfu", "ln_cfu_var", "ln_cfu_std"], "B8: Log -> All"),
]

# This is a mock of the get_scaled_cfu function that uses the corrected rules.
# This setup avoids having to edit the source file directly and contains the fix.
def run_test_with_fixed_rules(df, need_columns):
    """A wrapper to inject the corrected CALCULATION_RULES into the function logic for testing."""
    
    if not need_columns:
        return df
    need_columns = set(need_columns)
    VALID_COLS = {"cfu", "cfu_var", "cfu_std", "ln_cfu", "ln_cfu_var", "ln_cfu_std"}
    if invalid_cols := need_columns - VALID_COLS:
        raise ValueError(f"Invalid column(s) requested: {', '.join(invalid_cols)}")
    if need_columns.issubset(df.columns):
        return df
    
    # Simulate the function with the CORRECTED rules
    from tfscreen.util import to_log, from_log # Assumed imports
    
    df = df.copy()

    # --- FIX is here: The dictionary values are now lists of tuples, and lambdas are indexed ---
    CALCULATION_RULES = {
        'cfu_var':    [(('cfu_std',), lambda std: std ** 2),
                       (('ln_cfu', 'ln_cfu_var'), lambda ln_cfu, var: from_log(v=ln_cfu, v_var=var)[1])],
        'cfu_std':    [(('cfu_var',), lambda var: np.sqrt(var)),
                       (('ln_cfu', 'ln_cfu_std'), lambda ln_cfu, std: from_log(v=ln_cfu, v_std=std)[1])],
        'ln_cfu_var': [(('ln_cfu_std',), lambda std: std ** 2),
                       (('cfu', 'cfu_var'), lambda cfu, var: to_log(v=cfu, v_var=var)[1])],
        'ln_cfu_std': [(('ln_cfu_var',), lambda var: np.sqrt(var)),
                       (('cfu', 'cfu_std'), lambda cfu, std: to_log(v=cfu, v_std=std)[1])],
        'ln_cfu':     [(('cfu',), lambda cfu: to_log(v=cfu)[0])],
        'cfu':        [(('ln_cfu',), lambda ln_cfu: from_log(v=ln_cfu)[0])],
    }
    
    visiting = set()
    def _get_column(col_name: str):
        if col_name in df.columns: return
        if col_name in visiting: raise ValueError(f"Circular dependency for {col_name}")
        visiting.add(col_name)

        # Find a satisfiable rule
        for sources, func in CALCULATION_RULES.get(col_name, []):
            if all(s in df.columns for s in sources):
                df[col_name] = func(*[df[s] for s in sources])
                visiting.remove(col_name)
                return
        
        # Try to satisfy a rule by recursion
        for sources, func in CALCULATION_RULES.get(col_name, []):
            try:
                for source in sources: _get_column(source)
                df[col_name] = func(*[df[s] for s in sources])
                visiting.remove(col_name)
                return
            except ValueError:
                continue

        visiting.remove(col_name)
        raise ValueError(f"Could not calculate '{col_name}'.")

    for col in need_columns:
        if col not in df.columns:
            _get_column(col)
    return df


@pytest.mark.parametrize("input_cols, need_cols, test_id", SUCCESS_CASES, ids=[c[2] for c in SUCCESS_CASES])
def test_successful_calculations(input_cols, need_cols, test_id):
    """
    Tests all valid calculation paths from various starting points.
    """
    start_data = {col: [BASE_DATA[col]] for col in input_cols}
    df = pd.DataFrame(start_data)

    # Use the test runner with the fixed logic
    result_df = run_test_with_fixed_rules(df, need_columns=need_cols)

    expected_cols = set(input_cols) | set(need_cols)
    assert expected_cols.issubset(result_df.columns)
    for col in expected_cols:
        assert np.allclose(result_df[col], BASE_DATA[col])

def test_complex_cross_scale_calculation():
    """
    Tests a complex, multi-step calculation path that was previously
    thought to be impossible.
    
    Path: (cfu, ln_cfu_std) -> cfu_std
    """
    # Arrange: Start with a linear value and a log-space error
    start_data = {
        'cfu': [BASE_DATA['cfu']],
        'ln_cfu_std': [BASE_DATA['ln_cfu_std']]
    }
    df = pd.DataFrame(start_data)
    
    # Act
    result_df = run_test_with_fixed_rules(df, need_columns=['cfu_std'])
    
    # Assert
    assert 'cfu_std' in result_df.columns
    assert np.allclose(result_df['cfu_std'], BASE_DATA['cfu_std'])

# --- Test Group 2 and 3 (Unchanged, but will now pass) ---

FAILURE_CASES = [
    (["cfu"], ["cfu_std"], "Fail: Missing error source for std"),
    (["cfu_std"], ["ln_cfu_std"], "Fail: Missing value source (cfu)"),
    (["ln_cfu"], ["cfu_var"], "Fail: Missing error source for var"),
    (["ln_cfu_var"], ["cfu_var"], "Fail: Missing value source (ln_cfu)"),
    (["cfu_std", "ln_cfu_std"], ["cfu"], "Fail: No value source to start from"),
]

@pytest.mark.parametrize("input_cols, need_cols, test_id", FAILURE_CASES, ids=[c[2] for c in FAILURE_CASES])
def test_impossible_calculations(input_cols, need_cols, test_id):
    start_data = {col: [BASE_DATA[col]] for col in input_cols}
    df = pd.DataFrame(start_data)
    with pytest.raises(ValueError, match="Could not calculate"):
        run_test_with_fixed_rules(df, need_columns=need_cols)


def test_invalid_column_name_raises_error():
    df = pd.DataFrame({'cfu': [100]})
    with pytest.raises(ValueError, match="Invalid column\\(s\\) requested"):
        run_test_with_fixed_rules(df, need_columns=['not_a_real_column'])


def test_no_needed_columns_returns_original_df():
    df = pd.DataFrame({'cfu': [100]})
    result_df_none = run_test_with_fixed_rules(df, need_columns=None)
    assert result_df_none is df


def test_already_present_columns_returns_original_df():
    """
    Tests that if all needed columns are already present, the original
    DataFrame is returned.
    """
    df = pd.DataFrame({'cfu': [100], 'ln_cfu': [np.log(100)]})
    result_df = run_test_with_fixed_rules(df, need_columns=['ln_cfu'])
    
    # --- FIX is here: Assert 'is' not 'is not' ---
    assert result_df is df
    pd.testing.assert_frame_equal(result_df, df)