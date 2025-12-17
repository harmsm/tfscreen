from tfscreen.util.numerical.transform import (
    to_log,
    from_log
)

import numpy as np
import pandas as pd
from typing import (
    Optional, 
    Iterable
)

def get_scaled_cfu(
    df: pd.DataFrame, need_columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Ensures a DataFrame contains specified CFU-related columns.

    This function calculates missing CFU (Colony Forming Units) columns by
    converting between linear and log scales, and between variance and
    standard deviation. It operates by pre-calculating all possible columns
    from the source data before returning the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, which may contain some of the following columns:
        'cfu', 'cfu_var', 'cfu_std', 'ln_cfu', 'ln_cfu_var', 'ln_cfu_std'.
    need_columns : iterable of str, optional
        A list or set of column names that must be present in the output
        DataFrame. If None or empty, the original DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        A DataFrame with all original columns plus the newly calculated ones.

    Raises
    ------
    ValueError
        - If an unknown column is requested in `need_columns`.
        - If a requested column cannot be calculated due to missing source
          data in the input DataFrame.

    Examples
    --------
    >>> data = {'cfu': [100, 1000], 'cfu_std': [10, 50]}
    >>> df = pd.DataFrame(data)
    >>> result_df = get_scaled_cfu(df, need_columns=['ln_cfu', 'ln_cfu_var'])
    >>> print(result_df[['ln_cfu', 'ln_cfu_var']].round(3))
       ln_cfu  ln_cfu_var
    0   4.605       0.001
    1   6.908       0.002

    >>> # Example of a complex request succeeding
    >>> df = pd.DataFrame({'cfu': [100], 'cfu_std': [10]})
    >>> all_cols = ['cfu', 'cfu_var', 'cfu_std', 'ln_cfu', 'ln_cfu_var', 'ln_cfu_std']
    >>> result = get_scaled_cfu(df, need_columns=all_cols)
    >>> print(result.round(3))
       cfu  cfu_std  cfu_var  ln_cfu  ln_cfu_std  ln_cfu_var
    0  100     10.0    100.0   4.605         0.1       0.001
    """
    
    if not need_columns:
        return df

    need_columns = set(need_columns)
    
    # --- Validation ---
    VALID_COLS = {
        "cfu", "cfu_var", "cfu_std",
        "ln_cfu", "ln_cfu_var", "ln_cfu_std"
    }
    invalid_cols = need_columns - VALID_COLS
    if invalid_cols:
        raise ValueError(f"Invalid column(s) requested: {', '.join(invalid_cols)}")
    
    if need_columns.issubset(df.columns):
        return df

    df = df.copy()

    current_valid = list(df.columns.isin(VALID_COLS))
    tmp_df = df[df.columns[current_valid]].reset_index()

    # --- Pre-computation: Generate all possible columns ---

    # 1. Generate variances from standard deviations
    if 'cfu_std' in tmp_df.columns and 'cfu_var' not in tmp_df.columns:
        tmp_df['cfu_var'] = tmp_df['cfu_std'] ** 2
    if 'ln_cfu_std' in tmp_df.columns and 'ln_cfu_var' not in tmp_df.columns:
        tmp_df['ln_cfu_var'] = tmp_df['ln_cfu_std'] ** 2

    # 2. Generate standard deviations from variances
    if 'cfu_var' in tmp_df.columns and 'cfu_std' not in tmp_df.columns:
        tmp_df['cfu_std'] = np.sqrt(tmp_df['cfu_var'])
    if 'ln_cfu_var' in tmp_df.columns and 'ln_cfu_std' not in tmp_df.columns:
        tmp_df['ln_cfu_std'] = np.sqrt(tmp_df['ln_cfu_var'])

    # 3. Generate log-space columns from linear-space
    if 'cfu' in tmp_df.columns:
        if 'ln_cfu' not in tmp_df.columns:
            tmp_df['ln_cfu'] = to_log(tmp_df['cfu'])
        if 'cfu_std' in tmp_df.columns and 'ln_cfu_std' not in tmp_df.columns:
            _, tmp_df['ln_cfu_std'] = to_log(v=tmp_df['cfu'], v_std=tmp_df['cfu_std'])
        if 'cfu_var' in tmp_df.columns and 'ln_cfu_var' not in tmp_df.columns:
            _, tmp_df['ln_cfu_var'] = to_log(v=tmp_df['cfu'], v_var=tmp_df['cfu_var'])

    # 4. Generate linear-space columns from log-space
    if 'ln_cfu' in tmp_df.columns:
        if 'cfu' not in tmp_df.columns:
            tmp_df['cfu'] = from_log(tmp_df['ln_cfu'])
        if 'ln_cfu_std' in tmp_df.columns and 'cfu_std' not in tmp_df.columns:
            _, tmp_df['cfu_std'] = from_log(v=tmp_df['ln_cfu'], v_std=tmp_df['ln_cfu_std'])
        if 'ln_cfu_var' in tmp_df.columns and 'cfu_var' not in tmp_df.columns:
            _, tmp_df['cfu_var'] = from_log(v=tmp_df['ln_cfu'], v_var=tmp_df['ln_cfu_var'])

    # --- Final Check ---
    final_missing = need_columns - set(tmp_df.columns)
    if final_missing:
        raise ValueError(
            f"Could not calculate the following columns: {list(final_missing)}. "
            "Insufficient source data in the DataFrame."
        )

    # Identify columns in tmp_df that are not in df
    cols_to_add = tmp_df.columns.difference(df.columns)

    # Add only the new columns to df
    df = df.join(tmp_df[cols_to_add])
    
    return df