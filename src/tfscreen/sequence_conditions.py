"""
Functions for simulating the sequencing of conditions.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def sequence_conditions(condition_pops,
                        condition_df_with_time,
                        genotype_df,
                        bacteria,
                        total_num_reads):
    """
    Simulate sequencing of bacterial populations across different conditions.

    This function simulates the sequencing process by sampling reads from the
    final bacterial populations in each condition. It generates a dataframe
    containing the counts of each genotype observed in each condition,
    mimicking the output of a real sequencing experiment.

    Parameters
    ----------
    condition_pops : dict
        Dictionary mapping condition names to log population arrays (ln_pop).
        Each ln_pop array represents the log-transformed population size of
        each bacterium in that condition.
    condition_df_with_time : pandas.DataFrame
        DataFrame containing information about each condition, including time.
    genotype_df : pandas.DataFrame
        DataFrame containing information about each genotype, including
        genotype names and other relevant features.
    bacteria : numpy.ndarray
        Array of bacteria, where each element is either a genotype string
        (if there is one plasmid per bacterium) or a tuple of genotype strings
        (if there are multiple plasmids per bacterium).
    total_num_reads : int
        The total number of reads to simulate across all conditions.  This is
        split evenly across all conditions.

    Returns
    -------
    count_df : pandas.DataFrame
        DataFrame containing the counts of each genotype observed in each
        condition.  Columns include 'condition', 'base_condition', 'time',
        'genotype', and 'counts'.
    condition_df_with_time : pandas.DataFrame
        DataFrame containing information about each condition, updated with
        'cfu_per_mL' and 'num_reads' columns.
    """
    
    print("Sequencing final populations",flush=True)
    
    num_samples = len(condition_df_with_time)        
    condition_df_with_time["cfu_per_mL"] = 0.0
    condition_df_with_time["num_reads"] = int(total_num_reads//num_samples)

    # Make bacteria ints for faster sort/unique calls
    bacteria_as_int = np.arange(len(bacteria),dtype=int)

    # Create template dataframe to store results for each condition
    template_df = genotype_df.copy()
    columns = list(template_df.columns)
    to_drop = [c for c in columns if c not in ["genotype"]]
    template_df = template_df.drop(columns=to_drop)

    # Together genotype_number and base_condition_number will allow us to sort
    # the final dataframe in a human-readable way that will also allow us to 
    # readily pull counts for future fitting. 

    # Create a genotype_number
    template_df["genotype_number"] = np.arange(len(template_df["genotype"]),
                                               dtype=int)
    
    # Create dictionary mapping base conditions to their order in the 
    # condition_df. 
    counter = 0
    base_condition_number = {}
    for c in condition_df_with_time.index:
        base_condition = "-".join(c.split("-")[:-1])
        if base_condition not in base_condition_number:
            base_condition_number[base_condition] = counter
            counter += 1

    # Go through each condition ... 
    sequencing_dataframes = []
    for condition in tqdm(condition_pops):

        # Get information about this condition
        num_reads = condition_df_with_time.loc[condition,"num_reads"]
        time = condition_df_with_time.loc[condition,"time"]
        base_condition = "-".join(condition.split("-")[:-1])

        # Record the cfu/mL for this sample
        ln_pop = condition_pops[condition]
        condition_df_with_time.loc[condition,"cfu_per_mL"] = np.sum(np.exp(ln_pop))

        # this_df will hold the samples for all genotypes from this condition. 
        this_df = template_df.copy()
        this_df["condition"] = condition
        this_df["base_condition"] = base_condition
        this_df["base_condition_number"] = base_condition_number[base_condition]
        this_df["time"] = time
        this_df["counts"] = 0

        # Do a weighted sample from the bacteria (stored as integers)
        w = np.exp(ln_pop)
        w = w/np.sum(w)
        sample = np.random.choice(bacteria_as_int,size=num_reads,p=w)

        # Get the unique bacteria seen and their counts
        seen, counts = np.unique(sample,return_counts=True)
        bacteria_seen = bacteria[seen]

        # If this is true there is one plasmid per bacterium -- we can just 
        # grab the bacteria we see and turn into genotypes
        if issubclass(type(bacteria_seen[0]),str):
            genotypes_seen = bacteria_seen
            genotype_counts = counts
    
        # Otherwise, we have multiple plasmids per bacterium. We're going to 
        # randomly sample from each of those plasmids in our analysis.
        else:
            
            # Lists to hold broken out plasmids and counts
            genotypes_seen = []
            genotype_counts = []
            
            # Go through every bacterium we see
            for i in range(len(bacteria_seen)):
                
                # Grab random plasmids from this bacterium for sequencing
                plasmids_extracted = np.random.choice(bacteria_seen[i],
                                                      size=counts[i])
                
                # Count the unique plasmids pulled
                plasmids, p_count = np.unique(plasmids_extracted,
                                              return_counts=True)
                
                # Record these in our new bacteria/counts
                genotypes_seen.extend(plasmids)
                genotype_counts.extend(p_count)

        # Records the genotypes we saw. this_df has genotype name as its index
        this_df.loc[genotypes_seen,"counts"] = genotype_counts

        # Record the sequencing dataframe
        sequencing_dataframes.append(this_df)
        
    print("Building final dataframe",flush=True)

    # Build one huge dataframe with all sequencing results
    count_df = pd.concat(sequencing_dataframes)

    count_df = count_df.sort_values(["genotype_number",
                                     "base_condition_number",
                                     "time"])

    count_df.index = np.arange(len(count_df["genotype"]),dtype=int)

    # Return sequencing counts dataframe and condition dataframe
    return count_df, condition_df_with_time