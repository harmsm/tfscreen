import numpy as np

def count_df_to_arrays(df):
    """
    Take a dataframe with columns "genotype", "base_condition", "time",
    "counts", "total_counts_at_time", and "total_cfu_per_time" return numpy
    arrays with shape (num_samples,num_times) where num_samples iterates 
    over all combinations of genotype and and base condition. There must be
    an identical number of time points for each genotype/base_condition pair,
    though the time point values can all differ from one another. 

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe to process

    Returns
    -------
    times : np.ndarray
        2D float array of times with shape (num_samples,num_times)
    sequence_counts : np.ndarray 
        2D float array of counts of particular genotype with shape
        (num_samples,num_times)
    total_counts : np.ndarray
        2D float array of total counts in sample with shape
        (num_samples,num_times)
    total_cfu_ml : np.ndarray
        2D float array of total cfu/mL in sample with shape
        (num_samples,num_times)
    genotypes : np.ndarray
        1D object array of genotypes with shape (num_samples,)
    base_conditions : np.ndarray
        1D object array of base_condition strings with shape
        (num_samples,)
    """

    # Check for all required columns
    look_for = ["time","counts","total_counts_at_time","total_cfu_mL_at_time"]
    diff = list(set(look_for) - set(df.columns))
    diff.sort()
    if len(diff) > 0:
        err = "not all columns found in dataframe. missing columns:\n"
        for d in diff:
            err += f"    {d}\n"
        raise ValueError(err + "\n")
        
    # Get number of times per experiment
    number_of_times_per_expt = df.groupby(['genotype', 'base_condition']).size().unique()
    if len(number_of_times_per_expt) != 1:
        err = "each genotype/base_condition must have the same number of time entries\n"
        raise ValueError(err)

    num_l = len(df["time"])
    num_t = number_of_times_per_expt[0]

    # Extract relevant series as arrays
    times = np.array(df["time"])
    sequence_counts = np.array(df["counts"])
    total_counts = np.array(df["total_counts_at_time"])
    total_cfu_ml = np.array(df["total_cfu_mL_at_time"])

    # Reshape into sample x time arrays
    times = times.reshape(num_l//num_t,num_t)
    sequence_counts = sequence_counts.reshape(num_l//num_t,num_t)
    total_counts = total_counts.reshape(num_l//num_t,num_t)
    total_cfu_ml = total_cfu_ml.reshape(num_l//num_t,num_t)

    # Get genotypes and base conditions for masking arrays
    genotypes = np.array(df["genotype"])
    base_conditions = np.array(df["base_condition"])
    genotypes = genotypes[np.arange(0,num_l,num_t,dtype=int)]
    base_conditions = base_conditions[np.arange(0,num_l,num_t,dtype=int)]
    
    return times, sequence_counts, total_counts, total_cfu_ml, genotypes, base_conditions
