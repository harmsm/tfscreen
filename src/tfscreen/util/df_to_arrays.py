from tfscreen.util import read_dataframe

import numpy as np

def _count_df_to_arrays(df):
    """
    Take a dataframe and pivot it into a set of NumPy arrays.

    This function transforms a long-format DataFrame into a set of wide-format
    2D NumPy arrays suitable for numerical analysis. Each unique combination
    of (genotype, sample) becomes a single row in the output arrays.

    The implementation uses a robust pandas pivot operation, which does not
    require the input DataFrame to be sorted and correctly handles any missing
    time points by inserting `np.nan`.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to process. It must contain the columns:
        "genotype", "sample", "time", "counts", "total_counts_at_time",
        and "total_cfu_mL_at_time".

    Returns
    -------
    times : np.ndarray
        2D float array of times with shape (num_obs, num_times).
    sequence_counts : np.ndarray
        2D float array of counts for each genotype with shape
        (num_obs, num_times).
    total_counts : np.ndarray
        2D float array of total counts in the sample with shape
        (num_obs, num_times).
    total_cfu_ml : np.ndarray
        2D float array of total cfu/mL in the sample with shape
        (num_obs, num_times).
    genotypes : np.ndarray
        1D object array of genotypes with shape (num_obs,).
    samples : np.ndarray
        1D object array of sample strings with shape (num_obs,).

    Raises
    ------
    ValueError
        If any of the required columns are missing from the input DataFrame.
    """
    
    required_cols = {
        "genotype",
        "sample",
        "time",
        "counts",
        "total_counts_at_time",
        "total_cfu_mL_at_time"
    }
    missing_cols = sorted(list(required_cols - set(df.columns)))
    if missing_cols:
        err = "Not all required columns found in DataFrame. Missing columns:\n"
        for col in missing_cols:
            err += f"    {col}\n"
        raise ValueError(err)

    # Create a 'time_rank' column that numbers the time points (0, 1, 2, ...)
    # for each unique (genotype, sample) group. This provides a consistent
    # column to pivot on, regardless of the actual time values.
    df_copy = df.copy() 
    df_copy['time_rank'] = df_copy.groupby(['genotype', 'sample']).cumcount()

    # Get order of genotype/sample in the dataframe.
    genotype_sample = df[['genotype', 'sample']].drop_duplicates()
    original_order = genotype_sample.set_index(['genotype', 'sample']).index

    # Use pivot_table to reshape the data. Each (genotype, sample) pair will
    # become a row. The 'time_rank' will become the new columns.
    pivoted = df_copy.pivot_table(
        index=['genotype', 'sample'],
        columns='time_rank',
        values=[
            'time',
            'counts',
            'total_counts_at_time',
            'total_cfu_mL_at_time'
        ]
    ).reindex(original_order)

    # Extract the data matrices. The .values attribute returns the data
    # as a NumPy array.
    times = pivoted['time'].values
    sequence_counts = pivoted['counts'].values
    total_counts = pivoted['total_counts_at_time'].values
    total_cfu_ml = pivoted['total_cfu_mL_at_time'].values

    # Extract the row labels from the pivot table's index.
    genotypes = pivoted.index.get_level_values('genotype').to_numpy()
    samples = pivoted.index.get_level_values('sample').to_numpy()

    return times, sequence_counts, total_counts, total_cfu_ml, genotypes, samples



def _get_ln_cfu(sequence_counts,
                total_counts,
                total_cfu_ml,
                pseudocount=1):
    """
    Convert sequence counts to frequencies and variances. 

    Parameters
    ----------
    sequence_counts : numpy.ndarray
        Array of sequence counts for a specific genotype/sample.
    total_counts : numpy.ndarray
        Array of total sequence counts for each sample.
    total_cfu_ml : numpy.ndarray
        Array of total CFU/mL measurements for each sample.
    pseudocount : int or float, optional
        Pseudocount added to sequence counts to avoid division by zero. Default is 1.

    Returns
    -------
    ln_cfu : numpy.ndarray
        Natural logarithm of the CFU/mL for each genotype
    ln_cfu_var : numpy.ndarray
        Variance of the natural logarithm of the CFU/mL for each genotype.   
    """

    n = len(sequence_counts)

    adj_sequence_counts = sequence_counts + pseudocount
    adj_total_counts = total_counts + n*pseudocount
    
    f = (adj_sequence_counts)/(adj_total_counts)
    f_var = f*(1 - f)/(adj_total_counts)
    
    cfu = f*total_cfu_ml
    cfu_var = f_var*(total_cfu_ml**2)
        
    ln_cfu = np.log(cfu)
    ln_cfu_var = cfu_var/(cfu**2)

    return cfu, cfu_var, ln_cfu, ln_cfu_var


def df_to_arrays(combined_df,
                 sample_df,
                 pseudocount=1):
    """
    Process dataframes for fitting growth parameters.

    This function takes two dataframes, a combined dataframe containing
    sequence counts, cfu_mL, and total reads, as well as a sample dataframe
    specifying the conditions for each sample. This function then converts 
    these dataframes into a set of numpy arrays. 

    Parameters
    ----------
    combined_df : str or pandas.DataFrame
        Combined dataframe. This dataframe should have columns "genotype",
        "sample", "time", "counts", "total_counts_at_time", and
        "total_cfu_mL_at_time". If string, this is interpreted as a path to the
        file with the dataframe. 
    sample_df : str or pandas.DataFrame
        Sample dataframe. This dataframe should have columns "marker", "select"
        and "iptg" and be indexed by 'sample' (replicate|marker|select|iptg). 
        If string, this is interpreted as a path to the file with the dataframe. 
    pseudocount : int or float, optional
        Pseudocount added to sequence counts to avoid division by zero when
        calculating frequencies. Default is 1.

    Returns
    -------
    out : dict

        A dictionary containing the processed data. The dictionary has the
        following keys:

        *   "genotypes" : numpy.ndarray
            1D array of genotypes. Shape (num_genotypes*num_samples)
        *   "samples" : numpy.ndarray
            1D array of samples. Shape (num_genotypes*num_samples)
        *   "times" : numpy.ndarray
            2D array of time points
            Shape: (num_times,num_genotypes*num_samples)
        *   "cfu" : numpy.ndarray
            2D array of CFU values
            Shape: (num_times,num_genotypes*num_samples)
        *   "cfu_var" : numpy.ndarray
            2D array of CFU variances
            Shape: (num_times,num_genotypes*num_samples)
        *   "ln_cfu" : numpy.ndarray
            2D array of natural log of CFU values
            Shape: (num_times,num_genotypes*num_samples)
        *   "ln_cfu_var" : numpy.ndarray
            2D array of variances of natural log of CFU values, expanded to
            include the time zero point. 
            Shape: (num_times,num_genotypes*num_samples)
    """

    combined_df = read_dataframe(combined_df)
    sample_df = read_dataframe(sample_df)

    # Convert the dataframe into a collection of numpy arrays
    _results = _count_df_to_arrays(combined_df)
    
    # 2D arrays (num_times,num_genotypes*num_samples)
    times = _results[0]
    sequence_counts = _results[1]
    total_counts = _results[2]
    total_cfu_ml = _results[3]

    # 1D arrays (num_genotypes*num_samples). 
    genotypes = _results[4]
    samples = _results[5]

    # Extract ln_cfu and ln_cfu_variance from the count data
    cfu, cfu_var, ln_cfu, ln_cfu_var = _get_ln_cfu(
        sequence_counts=sequence_counts,
        total_counts=total_counts,
        total_cfu_ml=total_cfu_ml,
        pseudocount=pseudocount
    )
    
    out = {"genotypes":genotypes,
           "samples":samples,
           "sequence_counts":sequence_counts,
           "times":times,
           "cfu":cfu,
           "cfu_var":cfu_var,
           "ln_cfu":ln_cfu,
           "ln_cfu_var":ln_cfu_var}
    
    return out