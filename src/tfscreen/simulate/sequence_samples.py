"""
Functions for simulating the sequencing of timepoints. 
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def _sample_population(bacteria,
                       ln_pop,
                       num_reads):
    """
    Sample genotypes from a bacterial population for sequencing.

    This function simulates the process of sampling reads from a bacterial
    population, taking into account the abundance of each bacterium. It
    returns the genotypes observed in the sample and their corresponding
    counts.

    Parameters
    ----------
    bacteria : numpy.ndarray
        Array of bacteria, where each element is either a genotype string
        (if there is one plasmid per bacterium) or a tuple of genotype strings
        (if there are multiple plasmids per bacterium).
    ln_pop : numpy.ndarray
        Log-transformed population sizes for each bacterium in the population.
    num_reads : int
        Number of reads to sample from the population.

    Returns
    -------
    genotypes_seen : numpy.ndarray
        List of unique genotypes observed in the sample.
    genotype_counts : numpy.ndarray
        Array of counts for each unique genotype observed in the sample.
    """

    # Make bacteria ints for faster sort/unique calls
    bacteria_as_int = np.arange(len(bacteria),dtype=int)

    # Do a weighted sample from the bacteria (stored as integers)
    w = np.exp(ln_pop)
    w = w/np.sum(w)
    sample = np.random.choice(bacteria_as_int,
                              size=num_reads,
                              p=w,
                              replace=True)

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

    genotypes_seen = np.array(genotypes_seen)
    genotype_counts = np.array(genotype_counts,dtype=int)

    return genotypes_seen, genotype_counts


def _get_overall_plasmid_freqs(bacteria,ln_pop):
    """
    Calculate the overall frequency of each plasmid in a bacterial population.

    This function calculates the overall frequency of each plasmid in a
    bacterial population, taking into account both the frequency of each
    bacterium and the identities of the plasmids in each bacterium.

    Parameters
    ----------
    bacteria : numpy.ndarray
        Array of bacteria, where each element is either a genotype string
        (if there is one plasmid per bacterium) or a tuple of genotype strings
        (if there are multiple plasmids per bacterium).
    ln_pop : numpy.ndarray
        Log-transformed population sizes for each bacterium in the population.

    Returns
    -------
    plasmid : numpy.ndarray
        Array of unique plasmids in the population.
    overall_plasmid_freq : numpy.ndarray
        Array of overall frequencies for each unique plasmid.
    """

    # Frequency of bacteria in at each timepoint
    bacterial_freq = np.exp(ln_pop)

    # Simple case: no multi-transformants
    if issubclass(type(bacteria[0]),str):
        plasmids_to_count = bacteria
        all_plasmid_freq = bacterial_freq

    # Harder case: multi-transformant
    else:
        plasmids_to_count = []
        all_plasmid_freq = []
        for i in range(len(bacteria)):
            plasmids_to_count.extend(bacteria[i])
            all_plasmid_freq.extend(np.ones(len(bacteria[i]))*bacterial_freq[i])

    # This holds all plasmids seen with their associated bacterial frequencies
    plasmids_to_count = np.array(plasmids_to_count)
    all_plasmid_freq = np.array(all_plasmid_freq)

    # Get all unique plasmids
    plasmid, index, counts = np.unique(plasmids_to_count,
                                       return_counts=True,
                                       return_index=True)

    w = all_plasmid_freq[index]*counts
    overall_plasmid_freq = w/np.sum(w)

    return plasmid, overall_plasmid_freq
        

def _index_hopping(genotype_seen,
                   genotype_counts,
                   bacteria,
                   ln_pop,
                   index_hop_freq):
    """
    Simulate index hopping during sequencing.

    This function simulates the process of index hopping, where a small
    number of reads are incorrectly reassigned to other genotypes in the
    population.

    Parameters
    ----------
    genotype_seen : list
        List of unique genotypes observed in the initial sample.
    genotype_counts : numpy.ndarray
        Array of counts for each unique genotype observed in the initial sample.
    bacteria : numpy.ndarray
        Array of bacteria, where each element is either a genotype string
        (if there is one plasmid per bacterium) or a tuple of genotype strings
        (if there are multiple plasmids per bacterium).
    ln_pop : numpy.ndarray
        Log-transformed population sizes for each bacterium in the population.
    index_hop_freq : float
        Proportion of reads that should be reassigned randomly after the
        initial sampling.

    Returns
    -------
    genotype_seen : numpy.ndarray
        Array of unique genotypes observed after simulating index hopping.
    genotype_counts : numpy.ndarray
        Array of counts for each unique genotype observed after simulating
        index hopping.
    """

    # Expand seen/counts into a read array and shuffle order
    reads = np.repeat(genotype_seen,genotype_counts)
    np.random.shuffle(reads)

    # We're going to replace num_to_hop reads.
    num_to_hop = int(np.round(reads.shape[0]*index_hop_freq,0))
    
    # Get the overall frequency of each plasmid in the current population,
    # taking into account both the frequency of each bacterium and the
    # identities of the plasmids in each bacterium. 
    plasmids, plasmid_freq = _get_overall_plasmid_freqs(bacteria,ln_pop)
    
    # Select hopped genotypes from the plasmid frequencies
    new_genotypes = np.random.choice(plasmids,
                                     size=num_to_hop,
                                     p=plasmid_freq,
                                     replace=True)
    
    # Replace reads
    reads[:num_to_hop] = new_genotypes

    # Get new genotypes seen and counts with the index hopping contaminants 
    # added in
    genotype_seen, genotype_counts = np.unique(reads,
                                               return_counts=True)

    return genotype_seen, genotype_counts
    

def sequence_samples(sample_pops,
                     sample_df_with_time,
                     genotype_df,
                     bacteria,
                     total_num_reads,
                     index_hop_freq=0.0):
    """
    Simulate sequencing of bacterial populations from different samples. 

    This function simulates the sequencing process by randomly sampling reads
    from the bacterial populations at each time point in each sample. It
    generates a dataframe containing the counts for each genotype observed at
    each time point mimicking the output of a real sequencing experiment.

    Parameters
    ----------
    sample_pops : dict
        Dictionary mapping timepoint names to log population arrays (ln_pop).
        Each ln_pop array represents the log-transformed population size of
        each bacterium in that timepoint.
    sample_df_with_time : pandas.DataFrame
        DataFrame containing information about each sample, including time.
    genotype_df : pandas.DataFrame
        DataFrame containing information about each genotype, including
        genotype names and other relevant features.
    bacteria : numpy.ndarray
        Array of bacteria, where each element is either a genotype string
        (if there is one plasmid per bacterium) or a tuple of genotype strings
        (if there are multiple plasmids per bacterium).
    total_num_reads : int
        The total number of reads to simulate across all timepoints.  This is
        split evenly across all timempoints.
    index_hop_freq : float, default=0
        Proportion of reads that should be reassigned randomly after the 
        initial sampling. 

    Returns
    -------
    count_df : pandas.DataFrame
        DataFrame containing the counts of each genotype observed in each
        timepoint.  Columns include 'timepoint', 'sample', 'sample_number','time',
        'genotype', and 'counts'.
    sample_df_with_time : pandas.DataFrame
        DataFrame containing information about each timepoint, updated with
        'cfu_per_mL' and 'num_reads' columns.
    """
        
    num_samples = len(sample_df_with_time)        
    sample_df_with_time["cfu_per_mL"] = 0.0
    sample_df_with_time["num_reads"] = int(total_num_reads//num_samples)

    # Create template dataframe to store results for each timepoint
    template_df = genotype_df.copy()
    columns = list(template_df.columns)
    to_drop = [c for c in columns if c not in ["genotype"]]
    template_df = template_df.drop(columns=to_drop)

    # Together genotype_number and sample_number will allow us to sort
    # the final dataframe in a human-readable way that will also allow us to 
    # readily pull counts for future fitting. 

    # Create a genotype_number
    template_df["genotype_number"] = np.arange(len(template_df["genotype"]),
                                               dtype=int)
    
    # Create dictionary mapping samples to their order in the sample_df. 
    counter = 0
    sample_number = {}
    for c in sample_df_with_time.index:
        sample = "|".join(c.split("|")[:-1])
        if sample not in sample_number:
            sample_number[sample] = counter
            counter += 1

    # Go through each timepoint ... 
    sequencing_dataframes = []
    desc = "{}".format("sequencing samples")
    for timepoint in tqdm(sample_pops,desc=desc,ncols=800):

        # Get information about this timepoint
        num_reads = sample_df_with_time.loc[timepoint,"num_reads"]
        time = sample_df_with_time.loc[timepoint,"time"]
        sample = "|".join(timepoint.split("|")[:-1])

        # Record the cfu/mL for this sample
        ln_pop = sample_pops[timepoint]
        sample_df_with_time.loc[timepoint,"cfu_per_mL"] = np.sum(np.exp(ln_pop))

        # this_df will hold the samples for all genotypes from this timepoint. 
        this_df = template_df.copy()
        this_df["timepoint"] = timepoint
        this_df["sample"] = sample
        this_df["sample_number"] = sample_number[sample]
        this_df["time"] = time
        this_df["counts"] = 0

        # Sample from bacteria
        genotypes_seen, genotype_counts = _sample_population(bacteria,
                                                             ln_pop,
                                                             num_reads)


        # Simulate index hopping, where a small number of reads are incorrectly
        # replaced by other genotypes in the population
        if index_hop_freq > 0:
            genotypes_seen, genotype_counts = _index_hopping(genotypes_seen,
                                                             genotype_counts,
                                                             bacteria,
                                                             ln_pop,
                                                             index_hop_freq)

        # Records the genotypes we saw. this_df has genotype name as its index
        this_df.loc[genotypes_seen,"counts"] = genotype_counts

        # Record the sequencing dataframe
        sequencing_dataframes.append(this_df)
        
    # Build one huge dataframe with all sequencing results
    count_df = pd.concat(sequencing_dataframes)

    count_df = count_df.sort_values(["genotype_number",
                                     "sample_number",
                                     "time"])

    count_df.index = np.arange(len(count_df["genotype"]),dtype=int)

    # Return sequencing counts dataframe and sample dataframe
    
    return count_df, sample_df_with_time