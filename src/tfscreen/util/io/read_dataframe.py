import pandas as pd

import warnings

from tfscreen.genetics.genotype_sorting import set_categorical_genotype

def read_dataframe(source, index_column=None):
    """
    Reads a spreadsheet from a file path or DataFrame.

    Handles .csv, .tsv, and .xlsx/.xls files. It can also intelligently
    set the index, including finding and using the common 'Unnamed: 0'
    column that pandas creates when an index is saved to a file.

    If a 'genotype' column is present (and is not used as the index),
    it is automatically standardized and converted to an ordered
    pd.Categorical. Standardization normalizes mutation notation
    (e.g., sorts multi-mutant alleles by site: Q2T/A1P → A1P/Q2T,
    collapses self-mutations to 'wt', etc.) and raises ValueError for
    any unrecognizable genotype string. The category order follows the
    original row order of the DataFrame; rows are not reordered.

    Parameters
    ----------
    source : pandas.DataFrame or str
        A pandas DataFrame or the file path to read.
    index_column : str, optional
        The desired column to be used as the DataFrame index. If this
        column is not found, the function will look for 'Unnamed: 0'
        and use it instead. If None, it will try to find and drop a
        spurious default index (a column named 'Unnamed: 0' with
        values 0, 1, 2, ...).

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame.

    Raises
    ------
    ValueError
        If a 'genotype' column is present and contains an unrecognizable
        genotype string (e.g., 'AfiftyT', '50', or conflicting mutations
        at the same site).
    """
    # 1. Handle different source types (path vs. DataFrame)
    if isinstance(source, str):
        path = source
        ext = path.split(".")[-1].strip().lower()
        try:
            if ext in ["xlsx", "xls"]:
                df = pd.read_excel(path)
            elif ext == "csv":
                df = pd.read_csv(path)
            elif ext == "tsv":
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path, sep=None, engine="python")
        except FileNotFoundError:
            raise ValueError(f"File not found at path: {path}")
        except Exception as e:
            raise IOError(f"Error reading file {path}: {e}")

    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError("`source` must be a file path (str) or pandas DataFrame.")

    # Handle the 'Unnamed: 0' column if it exists. This is a common artifact of
    # `df.to_csv()` without `index=False`.
    unnamed_col = "Unnamed: 0"
    if unnamed_col in df.columns:

        # If a specific index is requested, assume 'Unnamed: 0' is it.
        if index_column is not None and index_column not in df.columns:
            warnings.warn(f"Renaming column '{unnamed_col}' to '{index_column}'")
            df = df.rename(columns={unnamed_col: index_column})
        else:
            
            # Otherwise, check if it's just a spurious default index.
            # A spurious index is an integer column with values 0, 1, 2...
            col_data = df[unnamed_col]
            is_spurious = pd.api.types.is_integer_dtype(col_data) and \
                          col_data.equals(pd.RangeIndex(start=0, stop=len(df)).to_series())
            
            if is_spurious:
                # If it looks like a junk index, drop it.
                df = df.drop(columns=unnamed_col)

    # 3. Set the final index if requested
    if index_column is not None:
        if index_column in df.columns:
            df = df.set_index(index_column)
        elif df.index.name != index_column:
            # If after all that, we still can't find it, raise an error.
            raise ValueError(f"Column '{index_column}' not found in the DataFrame.")

    # 4. Standardize and categorize the genotype column if present as a column
    if "genotype" in df.columns:
        df = set_categorical_genotype(df, standardize=True, sort=False)

    return df