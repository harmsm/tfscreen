from tfscreen.util import (
    read_dataframe,
    check_columns,
)

from tfscreen.genetics import (
    set_categorical_genotype
)

import pandas as pd
import numpy as np

def _filter_low_observation_genotypes(df: pd.DataFrame,
                                      min_genotype_obs: int) -> pd.DataFrame:
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

def _impute_missing_genotypes(df: pd.DataFrame,
                              sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every sample in a library has a row for every genotype in that library.
    (Refactored to use merge for metadata imputation and guarantee sort order).
    """
    if df.empty:
        return df

    # Create a scaffold with every combination of sample and genotype within each library.
    lib_samples = df[['library', 'sample']].drop_duplicates()
    lib_genotypes = df[['library', 'genotype']].drop_duplicates()
    scaffold = pd.merge(lib_samples, lib_genotypes, on='library', how='outer')

    # Merge the original data onto the scaffold. This brings in 'counts' and
    # other data, leaving NaNs for missing combinations.
    complete_df = pd.merge(scaffold, df, on=['library', 'sample', 'genotype'], how='left')

    # Create a unique metadata map from the original df. This creates a small
    # DataFrame: one row for each sample with its complete metadata.
    sample_info_cols = sample_df.columns.tolist()
    sample_meta_map = df[['sample'] + sample_info_cols].drop_duplicates(subset=['sample'])

    # Impute the missing metadata using a merge. Drop the now-sparse metadata
    # columns and merge the complete map back in.
    complete_df = complete_df.drop(columns=sample_info_cols)
    complete_df = pd.merge(complete_df, sample_meta_map, on='sample', how='left')

    # Finalize the DataFrame. Fill missing counts with 0.
    complete_df['counts'] = complete_df['counts'].fillna(0).astype(int)

    # Enforce a predictable, stable row order.
    sort_keys = ['library', 'sample', 'genotype']
    complete_df = complete_df.sort_values(by=sort_keys).reset_index(drop=True)

    # Reorder columns to match the original DataFrame's structure for consistency.
    final_col_order = [col for col in df.columns if col in complete_df.columns]
    
    return complete_df[final_col_order]

def _calculate_frequencies(df: pd.DataFrame,
                           pseudocount: int) -> pd.DataFrame:
    """
    Add a pseudocount and calculate genotype frequencies for each sample.

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
    """
    Calculate the cfu/mL for each genotype and propagate the variance.
    """
    
    # Calculate the cfu/mL for each genotype
    df['cfu'] = df['frequency'] * df['sample_cfu']

    # Convert input standard deviation into variance
    sample_cfu_var = (df["sample_cfu_std"])**2

    # Variance in frequency (from binomial uncertainty)
    total_counts_per_sample = df.groupby('sample')['adjusted_counts'].transform('sum')
    var_frequency = df['frequency'] * (1 - df['frequency']) / total_counts_per_sample

    # Propagate error from multiplying frequency and cfu
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_var_freq = np.nan_to_num(var_frequency / (df['frequency']**2))
        relative_var_sample_cfu = np.nan_to_num(sample_cfu_var / (df['sample_cfu']**2))

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
        for each sample. The required columns are 'library', 'sample_cfu',
        and 'sample_cfu_std'. Any additional columns (e.g., 'replicate',
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
                                              "sample_cfu",
                                              "sample_cfu_std"])
    
    # Read counts dataframe
    counts_df = read_dataframe(counts_df)
    check_columns(counts_df,required_columns=["sample",
                                              "genotype",
                                              "counts"])
    
    # Merge on the index of sample_df and the 'sample' column of counts_df
    combined_df = pd.merge(counts_df,
                           sample_df,
                           left_on=['sample'],
                           right_index=True,
                           how='left')

    # Remove genotypes with too few observations per library
    filtered_df = _filter_low_observation_genotypes(combined_df, min_genotype_obs)

    # Impute missing genotypes so all samples in a library have all genotypes
    imputed_df = _impute_missing_genotypes(filtered_df, sample_df)

    # If imputed_df is empty after filtering, return it.
    if imputed_df.empty:
        return imputed_df

    # Add pseudocount and calculate frequency
    freq_df = _calculate_frequencies(imputed_df, pseudocount)

    # Calculate genotype cfu/mL and propagate variance
    final_df = _calculate_concentrations_and_variance(freq_df)

    # Define genotype as a categorical variable
    final_df = set_categorical_genotype(final_df,standardize=True,sort=False)
    
    # Sort final dataframe
    sort_columns = ['genotype', 'library', 'sample']
    final_df = final_df.sort_values(by=sort_columns,ignore_index=True)

    return final_df
