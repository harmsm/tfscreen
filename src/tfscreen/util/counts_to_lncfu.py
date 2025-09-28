from tfscreen.util import (
    read_dataframe,
    argsort_genotypes,
    check_columns
)

import pandas as pd
import numpy as np

def _filter_low_observation_genotypes(df: pd.DataFrame, min_genotype_obs: int) -> pd.DataFrame:
    """Filter out genotypes with total counts below a threshold for each library.

    Parameters
    ----------
    df : pd.DataFrame
        The combined dataframe containing sample info, genotypes, and counts.
        Must include 'library', 'genotype', and 'counts' columns.
    min_genotype_obs : int
        The minimum total counts a genotype must have within a given library
        to be retained.

    Returns
    -------
    pd.DataFrame
        A dataframe with low-observation genotypes removed.
    """
    # Calculate the sum of counts for each genotype within each library
    genotype_counts_per_library = df.groupby(['library', 'genotype'],observed=True)['counts'].transform('sum')

    # Filter the dataframe, keeping rows where the genotype's total count in that library is sufficient
    filtered_df = df[genotype_counts_per_library >= min_genotype_obs].copy()

    return filtered_df

def _impute_missing_genotypes(df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every sample in a library has a row for every genotype in that library.

    After filtering, some samples may be missing genotypes that are present
    elsewhere in the same library. This function adds those missing rows back
    with a count of 0.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe after filtering for low-observation genotypes.
    sample_df : pd.DataFrame
        The original sample metadata dataframe, used to get column names.

    Returns
    -------
    pd.DataFrame
        A dataframe where each sample has a row for every genotype in its
        respective library.
    """
    if df.empty:
        return df

    # Create a scaffold with every combination of sample and genotype within each library.
    lib_structure = df[['library', 'sample']].drop_duplicates()
    genotype_structure = df[['library', 'genotype']].drop_duplicates()
    scaffold = pd.merge(lib_structure, genotype_structure, on='library', how='outer')

    # Merge the original data onto the scaffold.
    # This introduces NaNs for counts where a genotype was not observed in a sample.
    complete_df = pd.merge(scaffold, df, on=['library', 'sample', 'genotype'], how='left')

    # Fill missing counts with 0.
    complete_df['counts'] = complete_df['counts'].fillna(0).astype(int)

    # Fill the missing sample-specific data by grouping by sample and forward/backward filling.
    # This is efficient as all info for a given sample is identical.
    # This logic preserves all columns from the original sample_df.
    sample_info_cols = sample_df.columns.tolist()
    for col in sample_info_cols:
        if col in complete_df.columns:
            complete_df[col] = complete_df.groupby('sample')[col].transform(lambda x: x.ffill().bfill())

    return complete_df

def _calculate_frequencies(df: pd.DataFrame, pseudocount: int) -> pd.DataFrame:
    """Add a pseudocount and calculate genotype frequencies for each sample.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe after filtering. Must include 'sample' and 'counts'
        columns.
    pseudocount : int
        An integer value to add to each count to avoid zero frequencies.

    Returns
    -------
    pd.DataFrame
        The dataframe with 'adjusted_counts' and 'frequency' columns added.
    """
    # Add pseudocount to create adjusted counts
    df['adjusted_counts'] = df['counts'] + pseudocount

    # Calculate the total adjusted counts for each sample
    total_counts_per_sample = df.groupby('sample')['adjusted_counts'].transform('sum')

    # Calculate frequency of each genotype in its sample
    df['frequency'] = df['adjusted_counts'] / total_counts_per_sample

    return df

def _calculate_concentrations_and_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the cfu/mL for each genotype and propagate the variance."""
    
    # Rename sample-level columns first for clarity
    df = df.rename(columns={"cfu_per_mL": "sample_cfu_per_mL",
                            "cfu_per_mL_std": "sample_cfu_per_mL_std"})

    # Calculate the cfu/mL for each genotype
    df['cfu'] = df['frequency'] * df['sample_cfu_per_mL']

    # Convert input standard deviation into variance
    sample_cfu_per_mL_var = (df["sample_cfu_per_mL_std"])**2

    # --- Propagate Variance ---
    # 1. Variance in frequency (from binomial uncertainty)
    total_counts_per_sample = df.groupby('sample')['adjusted_counts'].transform('sum')
    var_frequency = df['frequency'] * (1 - df['frequency']) / total_counts_per_sample

    # 2. Propagate error for Z = X * Y
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_var_freq = np.nan_to_num(var_frequency / (df['frequency']**2))
        relative_var_sample_cfu = np.nan_to_num(sample_cfu_per_mL_var / (df['sample_cfu_per_mL']**2))

    relative_var_sum = relative_var_freq + relative_var_sample_cfu

    # Calculate the final variance for the genotype cfu
    df['cfu_var'] = (df['cfu']**2) * relative_var_sum

    # Calculate log-transformed values and their variance
    df['ln_cfu_var'] = relative_var_sum
    with np.errstate(divide='ignore'):
        df['ln_cfu'] = np.log(df['cfu'])

    # Handle cases where cfu is 0 or negative
    mask_zero_cfu = df['cfu'] <= 0
    df.loc[mask_zero_cfu, 'ln_cfu'] = np.nan
    df.loc[mask_zero_cfu, 'ln_cfu_var'] = np.nan

    return df


def counts_to_lncfu(
    sample_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    min_genotype_obs: int = 10,
    pseudocount: int = 1
) -> pd.DataFrame:
    """
    Combine sample information with genotype counts to calculate concentrations.

    This function takes sample metadata and genotype counts, merges them,
    filters out low-observation genotypes, and then calculates the frequency
    and concentration (cfu/mL) for each genotype in each sample. It also
    propagates experimental variance to estimate the variance of the
    genotype-specific concentrations.

    Parameters
    ----------
    sample_df : pd.DataFrame
        DataFrame indexed by a unique 'sample' string. Must contain metadata
        for each sample. The required columns are 'library', 'cfu_per_mL',
        and 'cfu_per_mL_std'. Any additional columns (e.g., 'replicate',
        'time', 'titrant_name') will be preserved in the final output.
    counts_df : pd.DataFrame
        DataFrame with 'sample', 'genotype', and 'counts' columns.
    min_genotype_obs : int, optional
        The minimum total number of observations (counts) a genotype must have
        across an entire library to be included in the analysis. Default is 10.
    pseudocount : int, optional
        A small integer count to add to all observations to prevent zero
        frequencies and aid in statistical calculations. Default is 1.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the combined data, with calculated
        frequencies, genotype-specific cfu/mL, and propagated variances.
        The DataFrame is sorted, and the 'genotype' column is cast as a
        categorical type.

    Raises
    ------
    ValueError
        If `sample_df` is not indexed by 'sample'.
    """
    
    # Read/clean up dataframes
    sample_df = read_dataframe(sample_df,index_column="sample")
    check_columns(sample_df,required_columns=["library",
                                              "cfu_per_mL",
                                              "cfu_per_mL_std"])
    
    counts_df = read_dataframe(counts_df)
    check_columns(counts_df,required_columns=["sample",
                                              "genotype",
                                              "counts"])
    
    # 1. Combine the two dataframes
    # Merge on the index of sample_df and the 'sample' column of counts_df
    combined_df = pd.merge(counts_df, sample_df, left_on=['sample'], right_index=True, how='left')

    # 2. Remove genotypes with too few observations per library
    filtered_df = _filter_low_observation_genotypes(combined_df, min_genotype_obs)

    # 3. Impute missing genotypes so all samples in a library have all genotypes
    imputed_df = _impute_missing_genotypes(filtered_df, sample_df)

    # If imputed_df is empty after filtering, return it.
    if imputed_df.empty:
        return imputed_df

    # 4. Add pseudocount and calculate frequency
    freq_df = _calculate_frequencies(imputed_df, pseudocount)

    # 5. Calculate genotype cfu/mL and propagate variance
    final_df = _calculate_concentrations_and_variance(freq_df)

    # 6. Define genotype as a categorical variable
    all_genotypes = pd.unique(final_df["genotype"])
    idx = argsort_genotypes(all_genotypes)
    genotype_order = all_genotypes[idx]

    final_df['genotype'] = pd.Categorical(final_df['genotype'],
                                          categories=genotype_order,
                                          ordered=True)
    
    # Sort final dataframe
    sort_columns = ['genotype', 'library', 'sample']
    final_df = final_df.sort_values(by=sort_columns,ignore_index=True)

    return final_df
