import numpy as np
import pandas as pd

import string
import re


def argsort_genotypes(genotypes):
    """
    Sort a 1D array of genotypes into a canonical order.

    The sorting hierarchy is as follows:
    1. 'wt' (case-insensitive) is always first.
    2. Increasing number of mutations (singles, then doubles, etc.).
    3. For mutants with the same number of mutations, sort lexicographically
       by the first mutation's site number, then its final residue.
    4. Sorting then proceeds to the second mutation's site/residue, and so on.

    Parameters
    ----------
    genotypes : array-like
        A 1D array or list of genotype strings. Mutations are expected in the
        format "A15V" and separated by slashes for multi-mutants.

    Returns
    -------
    np.ndarray
        A 1D integer array of indices that would sort the input `genotypes`.
    """ 
    
    # Handle empty input gracefully
    if len(genotypes) == 0:
        return np.array([], dtype=int)

    # Create a mapping from amino acid character to a sortable integer
    aa_to_idx = {a: i for i, a in enumerate(string.ascii_letters)}
    genotypes = np.array(genotypes, dtype=object)
    num_genotypes = len(genotypes)

    sort_columns = []

    # Column 0: wt vs. not wt (wt is 0, mutants are 1)
    is_wt = np.char.lower(genotypes.astype(str)) == 'wt'
    sort_columns.append(~is_wt)

    # Column 1: Number of mutations
    # Set wt num_muts to 0, otherwise count slashes.
    num_muts = np.char.count(genotypes.astype(str), '/') + 1
    num_muts[is_wt] = 0
    sort_columns.append(num_muts)

    # Use regex to robustly extract (site, mut_residue) for all mutations
    # This creates a MultiIndex DataFrame: (genotype_idx, match_num) -> (site, mut)
    regex = re.compile(r"[A-Z](\d+)([A-Z])")
    all_muts_df = (pd.Series(genotypes)
                     .str.extractall(regex)
                     .rename(columns={0: "site", 1: "mut"}))

    if not all_muts_df.empty:
        all_muts_df["site"] = pd.to_numeric(all_muts_df["site"])
        all_muts_df["mut"] = all_muts_df["mut"].map(aa_to_idx)

    # Create sorting columns for each mutation's site and residue
    # FIX: Add explicit size check before calling .max() to prevent ValueError
    max_muts = 0
    if num_muts.size > 0:
        max_muts = num_muts.max()

    for i in range(max_muts):
        # Unstack the ith mutation for all genotypes
        if all_muts_df.empty:
             mut_level_df = pd.DataFrame()
        else:
            mut_level_df = all_muts_df.unstack(level=1)

        # Site column for the ith mutation (NaN if no ith mutation)
        site_col = np.full(num_genotypes, np.nan)
        if not mut_level_df.empty and ('site', i) in mut_level_df.columns:
            s = mut_level_df[('site', i)]
            site_col[s.index] = s.values
        sort_columns.append(site_col)

        # Residue column for the ith mutation (NaN if no ith mutation)
        mut_col = np.full(num_genotypes, np.nan)
        if not mut_level_df.empty and ('mut', i) in mut_level_df.columns:
            m = mut_level_df[('mut', i)]
            mut_col[m.index] = m.values
        sort_columns.append(mut_col)

    # Build array for lexsort. Transpose so each row is a genotype.
    to_sort = np.array(sort_columns, dtype=float).T

    # Replace NaNs with a value lower than any real value. This ensures that
    # genotypes with fewer mutations (e.g., singles) sort before those with
    # more mutations (e.g., doubles) when comparing their non-existent sites.
    # FIX: Add explicit size check to prevent potential reduction errors
    min_val = 0
    if to_sort.size > 0:
        # Check if array is all NaNs before calling nanmin
        if not np.all(np.isnan(to_sort)):
            min_val = np.nanmin(to_sort)

    to_sort[np.isnan(to_sort)] = min_val - 1

    # np.lexsort sorts by the last column first, so we reverse the keys.
    keys = tuple(to_sort.T[::-1])
    return np.lexsort(keys)