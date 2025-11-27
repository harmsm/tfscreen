import tfscreen

import pandas as pd
import numpy as np

from typing import List, Optional

def mutant_cycle_pivot(
    df: pd.DataFrame,
    extract_columns: List[str],
    condition_selector: List[str] | str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Restructures a DataFrame for mutant cycle analysis.

    This function transforms a DataFrame from a "long" format into a "wide"
    format suitable for calculating epistasis. It identifies double mutants
    (e.g., "A15G/P75K") and, for each, creates a row containing the data for
    the double mutant itself, its constituent single mutants ("A15G", "P75K"),
    and the wildtype ("wt").

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing genotype and associated data. Must have a
        'genotype' column.
    extract_columns : list[str],
        A list of column names whose values will be extracted and placed into
        the new wide-format columns (e.g., 'fitness', 'expression').
    condition_selector : list[str] or str or None, optional
        Column name(s) to group the DataFrame by. The analysis is performed
        independently on each group. If None, treat the whole dataframe in a 
        single analysis. 
    verbose : bool, default False
        If True, print status messages about skipped groups or dropped data.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where each row corresponds to a double mutant.
        Columns from `extract_columns` are expanded into four versions:
        `00_*`, `01_*`, `10_*`, and `11_*`.

    Notes
    -----
    - The function expects genotypes in the format 'MUT1/MUT2'. It standardizes
      the order of mutations to ensure 'MUT1/MUT2' and 'MUT2/MUT1' are treated
      as identical.
    - Only second-order interactions (wt, single mutants, double mutants) are
      considered. Higher-order mutants are filtered out.
    - Groups with duplicate genotypes or missing wildtype data are skipped.
    """
    if df.empty:
        return pd.DataFrame()

    # Work on a copy to avoid modifying the original DataFrame
    df_proc = df.copy()

    # Standardize genotype strings (e.g., 'B/A' -> 'A/B')
    standard_genos = tfscreen.genetics.standardize_genotypes(df_proc["genotype"])
    df_proc["genotype"] = pd.Series(data=standard_genos, index=df_proc.index)

    # Filter for entries relevant to second-order epistasis (wt, singles, doubles)
    is_relevant = (df_proc["genotype"].str.count('/') < 2)
    df_proc = df_proc[is_relevant].copy()

    # Split genotypes into component mutations
    geno_cols = df_proc["genotype"].str.split("/", expand=True)
    df_proc["m0"] = geno_cols[0]

    if 1 in geno_cols.columns:
        df_proc["m1"] = geno_cols[1]
    else:
        df_proc["m1"] = None

    if condition_selector is None:
        grouper = [(None,df_proc)]
    else:
        grouper = df_proc.groupby(condition_selector)

    result_dfs = []
    for group, sub_df in grouper:

        # Handle genotypes that appear more than once within a group
        if not sub_df["genotype"].is_unique:
            sub_df = sub_df.drop_duplicates("genotype", keep=False)
            if verbose:
                print(f"Duplicate genotypes found for group '{group}'. Dropping all instances.")
        
        # A wildtype entry is required to calculate epistasis
        if "wt" not in sub_df["genotype"].values:
            if verbose:
                print(f"No 'wt' entry for group '{group}'. Skipping.")
            continue
            
        # Identify the double mutants that will form the basis of our output
        is_double = ~pd.isna(sub_df["m1"])
        cycle_df = sub_df.loc[is_double, :].copy()
        
        if cycle_df.empty:
            continue

        # --- Vectorized Lookup ---
        # Create a single lookup table (genotype -> row of values)
        mapper_df = sub_df.set_index("genotype")[extract_columns]

        # Perform four vectorized lookups to get values for wt, m0, m1, and m0m1
        vals_00 = mapper_df.reindex(["wt"] * len(cycle_df))
        vals_01 = mapper_df.reindex(cycle_df["m0"])
        vals_10 = mapper_df.reindex(cycle_df["m1"])
        vals_11 = mapper_df.reindex(cycle_df["genotype"])
        
        # Assign the retrieved values to new columns in the output DataFrame
        # The .values call is crucial to assign by position, ignoring the index.
        cycle_df[[f"00_{c}" for c in extract_columns]] = vals_00.values
        cycle_df[[f"01_{c}" for c in extract_columns]] = vals_01.values
        cycle_df[[f"10_{c}" for c in extract_columns]] = vals_10.values
        cycle_df[[f"11_{c}" for c in extract_columns]] = vals_11.values

        result_dfs.append(cycle_df)

    if not result_dfs:
        return pd.DataFrame()
        
    return pd.concat(result_dfs).reset_index(drop=True)


def extract_epistasis(
    df: pd.DataFrame,
    y_obs: str,
    y_std: Optional[str] = None,
    condition_selector: List[str] | str | None=None,
    scale: str = "add",
    keep_extra: bool = False
) -> pd.DataFrame:
    """
    Calculate epistasis between pairs of mutations for a given observable.

    This function orchestrates the calculation of second-order epistasis. It
    first uses `mutant_cycle_pivot` to restructure the data into a "wide"
    format, where each row contains the observable (`y_obs`) for a double
    mutant, its two single-mutant parents, and the wildtype. It then
    calculates the epistasis and propagates the error.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame in a "long" format, containing at least a 'genotype'
        column and the columns specified by `y_obs` and `y_std`.
    y_obs : str
        The name of the column containing the measurement for which epistasis
        will be calculated (e.g., 'fitness', 'dG').
    y_std : str, optional
        The name of the column containing the standard error for `y_obs`.
        If provided, the error on the epistasis (`ep_std`) will be calculated.
    condition_selector : list[str] or str or None
        Column name(s) that define a unique experimental condition. Epistasis
        is calculated independently for each condition. If None, treat all 
        conditions at once
    scale : {"add", "mult"}, default "add"
        The scale for calculating epistasis.
        - "add": epsilon = (Y_{11} - Y_{10}) - (Y_{01} - Y_{00})
        - "mult": epsilon = (Y_{11} / Y_{10}) / (Y_{01} / Y_{00})
    keep_extra : bool, default False
        If True, all columns from the original DataFrame are kept in the
        output. If False, only key identifiers and the calculated epistasis
        values are returned.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a double mutant under a
        specific condition. It includes the extracted values for the four
        states of the mutant cycle and the calculated `ep_obs` and `ep_std`.

    Raises
    ------
    ValueError
        If `scale` is not one of "add" or "mult".
    """

    # Determine the epistatic scale 
    if scale not in ["add","mult"]:
        err = "scale should be either 'add' (additive) or 'mult' (multiplicative)\n"
        raise ValueError(err)
    
    # Figure out what columns to extract
    extract_columns = [y_obs]
    if y_std is not None:
        extract_columns.append(y_std)
    
    # Build a dataframe with mutant cycles
    cycles = mutant_cycle_pivot(df,
                                extract_columns=extract_columns,
                                condition_selector=condition_selector)
    # Drop extra columns
    if not keep_extra:

        keep = ["genotype"]
        
        if condition_selector is not None:
            if isinstance(condition_selector, str):
                condition_selector = [condition_selector]
            keep.extend(condition_selector)

        for c in extract_columns:
            keep.extend([f"{mut}_{c}" for mut in ["00","01","10","11"]])
        cycles = cycles[keep]
    
    # Grab observations from cycles
    obs_00 = cycles[f"00_{y_obs}"]
    obs_10 = cycles[f"10_{y_obs}"]
    obs_01 = cycles[f"01_{y_obs}"]
    obs_11 = cycles[f"11_{y_obs}"]
    
    # Grab standard errors from cycles
    if y_std is not None:
        std_00 = cycles[f"00_{y_std}"]
        std_10 = cycles[f"10_{y_std}"]
        std_01 = cycles[f"01_{y_std}"]
        std_11 = cycles[f"11_{y_std}"]
    
    # Additive scale
    if scale == "add":
        ep_obs = (obs_11 - obs_10) - (obs_01 - obs_00)
        if y_std is not None:
            ep_std = np.sqrt(std_11**2 + std_10**2 + std_01**2 + std_00**2)
    
    # Multiplicative scale
    else: 
        ep_obs = (obs_11 / obs_10) / (obs_01 / obs_00)
        if y_std is not None:    
            rel_err_sq = ((std_11 / obs_11)**2 + (std_10 / obs_10)**2 +
                          (std_01 / obs_01)**2 + (std_00 / obs_00)**2)
            ep_std = np.abs(ep_obs) * np.sqrt(rel_err_sq)

    # Record epistasis and std_dev
    cycles["ep_obs"] = ep_obs
    if y_std is not None:
        cycles["ep_std"] = ep_std

    return cycles
