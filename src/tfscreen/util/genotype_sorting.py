import numpy as np
import pandas as pd

import string
import re

def standardize_genotypes(genotypes):
    """
    Take a list of genotypes and standardize their names. 

    Looks for genotypes with the name convention XsiteY (e.g., A47T), where X 
    and Y are single letters denoting wildtype and mutant states. 'site' must
    be coercible as an integer. The function expects multiple mutations to be
    separated by '/'. The function recognizes 'wt' or 'wildtype' (case
    insensitive) and converts to lowercase 'wt' in the output. Genotypes with
    only self->self mutations (e.g., A47A) are also converted to 'wt'. The 
    function drops multiple identical mutations (e.g. A47T/A47T -> A47T). 
    Multi-mutation sites are returned in site-sorted order (Q2T/A1P -> A1P/Q2T). 
    The function will raise an error if a genotype has multiple mutations at 
    the same site (e.g., A1A/A1V). 

    Parameters
    ----------
    genotypes : list-like
        list of genotypes to process. these do not have to be unique
    
    Returns
    -------
    list : 
        genotypes in clean, standardized format in the same order as the
        original input

    Raises
    ------
    ValueError
        Raised if there are unparsable genotypes (e.g., AfiftyT, 50, dumb) or
        nonsensical genotypes (A1V/A1Q). 
    """

    # Work on the subset of unique genotypes in the input
    unique_genotypes = list(set(genotypes))

    # Build the genotype_mapper dictionary, which maps input genotype names to 
    # clean, standardized names
    genotype_mapper = {}
    for g in unique_genotypes:

        # Ensure string representation
        g = f"{g}"

        # If wildtype (case-insensitive wt or wildtype), record as wt and 
        # continue
        if g.lower() in ["wt","wildtype"]:
            genotype_mapper[g] = "wt"
            continue

        # This will hold all mutations in the genotype for sorting by site 
        # number. Since it's a set, it only allows one copy of each mutation.
        mut_tuples = set()

        # Split on "/" to access individual mutations
        mutations = g.split("/")
        for m in mutations:

            # Minimum size must be 3 (e.g., A1T)
            if len(m) < 3:
                err = f"could not parse mutation '{m}' from genotype '{g}'\n"
                raise ValueError(err)
            
            # grab A and T from A278T. 
            wt_state = m[0]
            mut_state = m[-1]

            # If we have a self-to-self mutation, ignore the mutation.
            if wt_state == mut_state:
                continue

            # Try to extract the site number (must be coercible as an int)
            try:
                site_number = int(m[1:-1])
            except ValueError as e:
                err = f"could not get site number from mutation '{m}' in genotype '{g}'\n"
                raise ValueError(err) from e
            
            # Tuple with site_number first so sorting by tuple will sort by site
            mut_tuple = (site_number,wt_state,mut_state)

            # Store tuple. Set means we can only put each tuple in once, causing
            # something like A1V/A1V to end up as A1V. 
            mut_tuples.add(mut_tuple)

        # This happens with nothing but self -> self mutations. record as wt. 
        if len(mut_tuples) == 0:
            genotype_mapper[g] = "wt"
            continue

        # Convert mut_tuples to a list to sort
        sort_on = list(mut_tuples)

        # make sure we don't have a duplicated site (A1V and A1T). 
        num_unique_sites = len(set([s[0] for s in sort_on]))
        if num_unique_sites != len(sort_on):
            err = f"genotype '{g}' has multiple mutations at the same site\n"
            raise ValueError(err)

        # Sort from low to high site number
        sort_on.sort()

        # Reassemble genotype from sort_on and record the mapping
        clean_g = "/".join([f"{s[1]}{s[0]}{s[2]}" for s in sort_on])
        genotype_mapper[g] = clean_g

    # Map original genotype list back to a list of standardized genotype names
    clean_genotypes = [genotype_mapper[g] for g in genotypes]

    return clean_genotypes


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
    min_val = 0
    if to_sort.size > 0:

        # Check if array is all NaNs before calling nanmin
        if not np.all(np.isnan(to_sort)):
            min_val = np.nanmin(to_sort)

    to_sort[np.isnan(to_sort)] = min_val - 1

    # np.lexsort sorts by the last column first, so we reverse the keys.
    keys = tuple(to_sort.T[::-1])
    
    return np.lexsort(keys)

def set_categorical_genotype(df,standardize=False,sort=False):
    """
    Converts the 'genotype' column to an ordered categorical dtype.

    This function standardizes the sorting of genotypes within a DataFrame. It
    uses the `argsort_genotypes` helper to determine the canonical order
    (e.g., "wt", then single mutants, then double mutants, sorted by site
    number) and applies this order to the 'genotype' column by converting it
    to a pandas `Categorical` dtype. Optionally, it can also sort the entire
    DataFrame based on this new genotype order.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame. If it does not contain a "genotype" column,
        a copy of the DataFrame is returned unmodified.
    standardize : bool, optional
        If True, the genotype names will be standardized before categories are
        assigned. This does things like convert A7T/P2L -> P2L/A7T and convert
        A7A -> 'wt'. Defaults to False. 
    sort : bool, optional
        If True, the returned DataFrame will be sorted by the newly ordered
        "genotype" column. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the 'genotype' column converted to an ordered
        categorical type. The row order is preserved unless `sort_genotype`
        is True.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'genotype': ['C3A/D4B', 'A1G', 'wt', 'C3A'], 'value': [4, 2, 1, 3]}
    >>> df = pd.DataFrame(data)
    >>> new_df = set_categorical_genotype(df, sort_genotype=True)
    >>> print(new_df)
      genotype  value
    2       wt      1
    1      A1G      2
    3      C3A      3
    0  C3A/D4B      4
    >>> print(new_df.dtypes)
    genotype    category
    value          int64
    dtype: object
    """

    # work on a copy
    df = df.copy()

    # no genotype column, do nothing. (But this is still a copy in case the 
    # downstream code assumes the function spits out a copy as it would for 
    # df with genotype)
    if "genotype" not in df.columns:
        return df

    # Standardize if requested
    if standardize:
        df["genotype"] = standardize_genotypes(df["genotype"])

    # Get the canonical order of the genotypes in this dataframe
    all_genotypes = pd.unique(df["genotype"])
    idx = argsort_genotypes(all_genotypes)
    genotype_order = all_genotypes[idx]

    # Make the genotype column categorical with the order defined above
    df['genotype'] = pd.Categorical(df['genotype'],
                                    categories=genotype_order,
                                    ordered=True)
    
    # Final sort if requested
    if sort:
        df = df.sort_values(["genotype"])

    return df

