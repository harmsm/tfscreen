import tfscreen

import pandas as pd

from typing import Union, List

def get_single_with_wt(
    df: pd.DataFrame,
    condition_selector: Union[str, List[str]]
) -> pd.DataFrame:
    """Filter for single mutants and add explicit wildtype rows for each site.

    This function processes a DataFrame of mutational data to create a clean
    dataset of single-site effects. It performs three main operations:
    1. Filters the data to keep only single mutants and wildtype rows.
    2. For each mutated site (e.g., position 29), it creates a new
       "no-mutation" row (e.g., "H29H") by copying the data from the
       corresponding wildtype row for that condition.
    3. Discards the original, generic "wt" rows.

    The final DataFrame contains only single mutants and their site-specific
    wildtype counterparts, making it suitable for direct comparison or
    plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing mutational data. It must have a
        'genotype' column and any columns specified in `condition_selector`.
    condition_selector : str or list of str
        The column name or list of column names that together define a unique
        experimental condition.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only single-site data (single mutants plus
        the newly generated "no-mutation" rows), sorted by residue and
        mutant amino acid.
    """

    # empty input DataFrame.
    if df.empty:
        return df.copy()

    # Expand genotype column to have columns for wt_aa_1, resid_1, mut_aa_1, 
    # etc.
    if "num_muts" not in df.columns:
        edf = tfscreen.genetics.expand_genotype_columns(df)
    else:
        edf = df.copy()

    # This chain grabs genotypes with fewer than two mutations, renames
    # wt_aa_1 --> wt_aa, resid_1 --> resid, etc. and then drops any remaining
    # columns corresponding to more than the one mutation. 
    drop_regex = r"(?:wt_aa_|mut_aa_|resid_)"
    sdf = (edf[edf["num_muts"] < 2]
           .rename(columns={"wt_aa_1":"wt_aa",
                            "resid_1":"resid",
                            "mut_aa_1":"mut_aa"})
           .loc[:, lambda df: ~df.columns.str.contains(drop_regex)]
           .reset_index(drop=True)
          )

    # Grab all wt rows, dropping genotype, wt_aa, resid, and mut_aa. This is 
    # the template for building new rows like H29H. 
    wt_data = sdf[sdf["genotype"] == "wt"].drop(
        columns=['genotype', 'wt_aa', 'resid', 'mut_aa']
    ).copy()

    # empty dataframe
    if wt_data.empty:
        return (sdf[sdf["genotype"] != "wt"]
                .sort_values(["resid", "mut_aa"])
                .reset_index(drop=True))


    # Grab unique, non-nan resid/wt_aa pairs for non-wt genotypes
    site_info_df = (sdf
                    .loc[sdf['genotype'] != 'wt',['resid', 'wt_aa']]
                    .drop_duplicates()
                    .dropna())

    # Grab unique conditions
    unique_conditions = sdf[condition_selector].drop_duplicates()

    # Create a new dataframe that has all unique residues in all conditions. 
    new_rows_scaffold = pd.merge(site_info_df, unique_conditions, how='cross')

    # Build mut_aa and genotype columns for these new wildtype rows. 
    new_rows_scaffold['mut_aa'] = new_rows_scaffold['wt_aa']
    new_rows_scaffold['genotype'] = (
        new_rows_scaffold['wt_aa'] + 
        new_rows_scaffold['resid'].astype(int).astype(str) + 
        new_rows_scaffold['mut_aa']
    )
    
    # Merge the wildtype data onto the scaffold using the condition as
    # the key
    new_rows_populated = pd.merge(
        new_rows_scaffold, wt_data, on=condition_selector, how='left'
    )
    
    # Concatenate the new rows with the original DataFrame
    concat_df = pd.concat([sdf, new_rows_populated], ignore_index=True)
    final_df = (concat_df[concat_df["genotype"] != "wt"]
                .sort_values(["resid","mut_aa"])
                .reset_index(drop=True))

    return final_df
