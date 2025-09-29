import tfscreen
import pandas as pd
import numpy as np

def extract_epistasis(df,
                      obs_column,
                      std_column,
                      conditions=None,
                      scale="add",
                      drop_extra_columns=True):
    """
    Calculate mutational epistasis for all double-mutant cycles.

    This function identifies all thermodynamic cycles (wt, m1, m2, m12), and
    for each cycle, calculates the epistasis on an observable. The calculation
    can be performed on an additive or multiplicative scale. The analysis is
    performed independently for each unique combination of experimental
    conditions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing genotypes, observables, and conditions.
    obs_column : str
        Name of the column with the observable to analyze (e.g., "theta_est").
    std_column : str
        Name of the column with the standard error of the observable.
    conditions : str or list of str, optional
        Column name(s) that define unique experimental conditions. Epistasis
        will be calculated separately for each condition.
    scale : {"add", "mult"}, default "add"
        The scale for calculating epistasis.
        - "add": (m12 - m2) - (m1 - wt)
        - "mult": (m12 / m2) / (m1 / wt)
    drop_extra_columns : bool, default True
        If True, drops all columns not used in the calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one double-mutant cycle
        under a specific condition, containing the calculated epistasis,
        its standard error, and the underlying observable values.
    """
    # Validate scale parameter
    if scale not in ["add", "mult"]:
        raise ValueError("scale should be 'add' or 'mult'")

    # Read dataframe.
    df = tfscreen.util.read_dataframe(df)

    # Standardize conditions to be a list
    if conditions is None:
        conditions = []
    elif isinstance(conditions, str):
        conditions = [conditions]

    if obs_column in conditions or std_column in conditions:
        raise ValueError("Observable/std columns cannot also be conditions.")

    # Select only necessary columns if requested
    if drop_extra_columns:
        columns_to_keep = ["genotype"] + conditions + [obs_column, std_column]
        df = df[columns_to_keep]

    # Ensure every genotype exists for every condition
    df = tfscreen.util.expand_on_conditions(df, conditions)

    # Get all valid mutant cycles from the set of all genotypes
    cycle_array = tfscreen.util.build_cycles(df["genotype"].unique())
    if cycle_array.shape[0] == 0:
        return pd.DataFrame() # No cycles found, return empty frame

    # Group by condition and calculate epistasis for each.
    all_results = []
    
    # Correctly handle whether to group or not
    if conditions:
        grouped = df.groupby(conditions, observed=True)
    else:
        # Create an iterable of a single tuple (None, entire_dataframe)
        # to mimic the groupby iterator for the loop below.
        grouped = [(None, df)]

    # Group by conditions. If none, this loop runs once on the whole df.
    for _, group_df in grouped:

        # Create a fast lookup series (genotype -> value) for this condition
        obs_map = pd.Series(group_df[obs_column].values, index=group_df["genotype"])
        std_map = pd.Series(group_df[std_column].values, index=group_df["genotype"])

        # Map genotypes from cycles to observables and their errors
        wt_obs, m1_obs, m2_obs, m12_obs = [obs_map.loc[c].to_numpy() for c in cycle_array.T]
        wt_std, m1_std, m2_std, m12_std = [std_map.loc[c].to_numpy() for c in cycle_array.T]

        # Calculate epistasis (additive or multiplicative)
        if scale == "add":
            ep_obs = (m12_obs - m2_obs) - (m1_obs - wt_obs)
            ep_std = np.sqrt(wt_std**2 + m1_std**2 + m2_std**2 + m12_std**2)
        else: # scale == "mult"
            ep_obs = (m12_obs / m2_obs) / (m1_obs / wt_obs)
            # Propagate error for f = (d/c) / (b/a) -> f = ad/bc
            rel_err_sq = ((wt_std / wt_obs)**2 + (m1_std / m1_obs)**2 +
                          (m2_std / m2_obs)**2 + (m12_std / m12_obs)**2)
            ep_std = np.abs(ep_obs) * np.sqrt(rel_err_sq)

        # Build output dataframe for this condition
        out_df = pd.DataFrame({
            "genotype": cycle_array[:, 3],
            "ep_obs": ep_obs,
            "ep_std": ep_std,
            "wt_obs": wt_obs, "wt_std": wt_std,
            "m1_obs": m1_obs, "m1_std": m1_std,
            "m2_obs": m2_obs, "m2_std": m2_std,
            "m12_obs": m12_obs, "m12_std": m12_std,
        })

        # Add condition values for this group
        if conditions:
            for cond in conditions:
                out_df[cond] = group_df[cond].iloc[0]

        all_results.append(out_df)

    # Combine results from all conditions
    final_df = pd.concat(all_results, ignore_index=True)

    # Add parsed mutation info columns
    mut_info = final_df["genotype"].str.extract(r'([A-Z])(\d+)([A-Z])/([A-Z])(\d+)([A-Z])')
    mut_info.columns = ["wt_1", "site_1", "mut_1", "wt_2", "site_2", "mut_2"]
    final_df = final_df.join(mut_info)

    # Sort final output by genotype, then by specified conditions
    final_genotypes = pd.unique(final_df["genotype"])
    idx = tfscreen.util.argsort_genotypes(final_genotypes)
    final_genotype_order = final_genotypes[idx]
    
    final_df["genotype"] = pd.Categorical(final_df["genotype"],
                                          categories=final_genotype_order,
                                          ordered=True)    
    
    final_sort_order = ['genotype'] + conditions
    final_df = final_df.sort_values(by=final_sort_order).reset_index(drop=True)

    return final_df
