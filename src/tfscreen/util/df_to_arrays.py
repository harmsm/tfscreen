import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def df_to_arrays(df, pivot_on="genotype"):
    """
    Transforms a long-format DataFrame into wide-format NumPy arrays.

    This function reshapes a DataFrame from a long to a wide format based on
    the columns specified in `pivot_on`. Each unique combination of values
    in `pivot_on` becomes a row in the output arrays. It automatically handles
    groups with varying numbers of observations by padding with `np.nan`.

    Parameters
    ----------
    df : pd.DataFrame
        The long-format pandas DataFrame to process.
    pivot_on : str or list[str]
        Column name(s) to group by. These define the rows of the output.

    Returns
    -------
    row_ids : pd.MultiIndex
        An index where each entry corresponds to a row in the output arrays,
        containing the unique values from the `pivot_on` columns.
    out : dict[str, np.ndarray]
        A dictionary where keys are the original numeric column names and
        values are the corresponding 2D NumPy arrays in wide format.

    Example
    -------
    >>> data = {
    ...     "strain": ["A", "A", "A", "B", "B"],
    ...     "condition": ["X", "X", "X", "Y", "Y"],
    ...     "time": [0, 1, 2, 0, 1],
    ...     "OD": [0.1, 0.2, 0.4, 0.15, 0.35],
    ...     "fluorescence": [10, 22, 45, 12, 25]
    ... }
    >>> df = pd.DataFrame(data)
    >>> row_ids, arrays = df_to_arrays(df, pivot_on=["strain", "condition"])
    >>> print(row_ids)
    MultiIndex([('A', 'X'),
                ('B', 'Y')],
               names=['strain', 'condition'])
    >>> print(arrays["OD"])
    [[0.1  0.2  0.4 ]
     [0.15 0.35  nan]]
    """

    if isinstance(pivot_on, str):
        pivot_on = [pivot_on]

    if df.empty:
        if len(pivot_on) > 1:
            row_ids = pd.MultiIndex.from_tuples([], names=pivot_on)
        else:
            row_ids = pd.Index([], name=pivot_on[0])
        return row_ids, {}

    numeric_cols = [
        c for c in df.columns if c not in pivot_on and is_numeric_dtype(df[c])
    ]
    
    df = df.copy()
    df['_observation_idx'] = df.groupby(pivot_on).cumcount()
    
    
    # Determine the definitive row identifiers and their order from the input df.
    # This is robust to the pivot_table returning an empty index later.
    if len(pivot_on) > 1:
        unique_rows_df = df[pivot_on].drop_duplicates().sort_values(by=pivot_on)
        row_ids = pd.MultiIndex.from_frame(unique_rows_df)
    else:
        # For a single column, unique().sort() is simpler.
        unique_vals = np.sort(df[pivot_on[0]].unique())
        row_ids = pd.Index(unique_vals, name=pivot_on[0])

    pivoted = df.pivot_table(
        index=pivot_on,
        columns="_observation_idx",
        values=numeric_cols,
    )
    
    # Reindex the pivot result. This forces all groups to be present as rows,
    # even if they were dropped because they only contained NaNs.
    pivoted = pivoted.reindex(row_ids)


    available_cols = pivoted.columns.get_level_values(0).unique()
    out = {col: pivoted[col].values for col in available_cols}
    missing_cols = set(numeric_cols) - set(available_cols)
    
    if missing_cols:
        if out:
            target_shape = next(iter(out.values())).shape
        else:
            num_rows = len(row_ids) # Now row_ids is correct
            num_cols = (df['_observation_idx'].max() + 1) if not df.empty else 0
            target_shape = (num_rows, num_cols)

        for col in missing_cols:
            out[col] = np.full(target_shape, np.nan)
            
    return row_ids, out