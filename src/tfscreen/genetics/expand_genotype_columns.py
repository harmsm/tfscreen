import pandas as pd
import numpy as np
from typing import Union, Iterable

def expand_genotype_columns(
    genotype_input: Union[pd.DataFrame, Iterable[str]]
) -> pd.DataFrame:
    """
    Expand slash-separated genotype strings into individual columns.

    This function takes an iterable of genotype strings or a DataFrame with a
    'genotype' column. For each genotype (e.g., "A15G/P42Q"), it parses the
    individual mutations and adds new columns for each component: `wt_aa_1`,
    `resid_1`, `mut_aa_1`, `wt_aa_2`, `resid_2`, etc. It also adds a
    `num_muts` column with the total count of mutations. The resid_x columns are
    Int64; the wt_aa* and mut_aa* will be strings. 

    Parameters
    ----------
    genotype_input : pandas.DataFrame or iterable of str
        The input data. This can either be a DataFrame that already contains
        a 'genotype' column, or an iterable (like a list or pandas Series)
        of genotype strings.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the original data plus the added columns. The
        new columns are inserted immediately after the 'genotype' column.

    Raises
    ------
    ValueError
        If the input is a DataFrame and it does not contain a 'genotype' column.

    Examples
    --------
    >>> import pandas as pd
    >>> genotypes = ["wt", "A15G", "P42Q/A15G", "V10L/I20M/L30F"]
    >>> expand_genotype_columns(genotypes)
             genotype wt_aa_1 resid_1 mut_aa_1 wt_aa_2 resid_2 mut_aa_2  num_muts
    0              wt     NaN     NaN      NaN     NaN     NaN      NaN         0
    1            A15G       A      15        G     NaN     NaN      NaN         1
    2       A15G/P42Q       A      15        G       P      42        Q         2
    3  V10L/I20M/L30F       V      10        L       I      20        M         3
    """
    
    # --- Input Validation and DataFrame Preparation ---
    if isinstance(genotype_input, pd.DataFrame):
        if "genotype" not in genotype_input.columns:
            raise ValueError("Input DataFrame must have a 'genotype' column.")
        df = genotype_input.copy()
    else:
        # Assumes an iterable of strings
        df = pd.DataFrame({"genotype": np.asarray(genotype_input)})

    # --- Parse Mutations into Temporary Columns ---
    # Split "A15G/P42Q" into columns "A15G" and "P42Q"
    indiv_muts = df["genotype"].str.split("/", expand=True)

    # --- 3. Extract Components from Each Mutation ---
    pat = r'([A-Z])(\d+)([A-Z])'  # Regex for e.g., "A15G"
    new_cols_df = pd.DataFrame()
    
    for k in indiv_muts.columns:

        # For each column of mutations, extract its three parts
        extracted = indiv_muts[k].str.extract(pat)

        # Name the new columns (wt_aa_1, resid_1, mut_aa_1, etc.) and cast the
        # resid to Int64
        resid_col = f"resid_{k+1}"
        extracted.columns = [f"wt_aa_{k+1}", resid_col, f"mut_aa_{k+1}"]
        extracted[resid_col] = extracted[resid_col].astype("Int64")

        new_cols_df = pd.concat([new_cols_df, extracted], axis=1)
        

    # --- Calculate Number of Mutations ---
    # Count how many non-null residues were found for each row
    resid_cols = [c for c in new_cols_df.columns if c.startswith('resid_')]
    new_cols_df["num_muts"] = new_cols_df[resid_cols].notna().sum(axis=1)

    # --- Combine and Reorder Columns ---
    # Concatenate the original df with the new columns
    df = pd.concat([df, new_cols_df], axis=1)
    
    # Programmatically reorder the columns for a clean output
    original_cols = list(df.columns[: -len(new_cols_df.columns)])
    new_cols_ordered = list(new_cols_df.columns)
    
    try:
        geno_idx = original_cols.index('genotype')
        final_cols = (
            original_cols[:geno_idx+1] + 
            new_cols_ordered + 
            original_cols[geno_idx+1:]
        )
        df = df[final_cols]
    except ValueError:
        # 'genotype' wasn't in the original columns (e.g., from list input)
        # so just put it first.
        df = df[["genotype"] + new_cols_ordered]

    return df