"""
Functions for simulating the growth of bacteria during a screen.
"""

import numpy as np
from tqdm.auto import tqdm

from .cell_growth_moves import grow_for_time

def simulate_growth(ln_pop_array,
                    bact_sample_k,
                    sample_df,
                    sample_df_with_time):
    """
    Simulate bacterial population growth in different samples over time.

    This function simulates the growth of bacterial populations in different 
    samples, using growth rates specified for each genotype in that sample and
    and specified time  points. It iterates through each sample, calculates the
    time steps, and updates the population sizes.

    Parameters
    ----------
    ln_pop_array : numpy.ndarray
        Array of log-transformed initial population sizes for each bacterium
        across all samples (n_bacteria, n_samples).
    bact_sample_k : numpy.ndarray
        Array of growth rates for each bacterium in each sample
        (n_bacteria, n_samples).
    sample_df : pandas.DataFrame
        DataFrame containing sample information (replicate, marker, select,
        iptg) used to filter time-dependent samples.
    sample_df_with_time : pandas.DataFrame
        DataFrame containing sample information at each time point,
        including 'replicate', 'marker', 'select', 'iptg', and 'time'.

    Returns
    -------
    sample_pops : dict
        Dictionary mapping sample IDs to log-transformed population sizes
        (ln_pop) at the end of the growth simulation for that sample. 
    """

    # st will hold all timepoint-level results (cfu/mL and num reads)
    st = sample_df_with_time.copy()
    
    # Dictionary to store populations for all time points
    sample_pops = {}

    # Go through each sample
    desc = "{}".format("growing populations")
    for i in tqdm(range(ln_pop_array.shape[1]),desc=desc,ncols=800):
        
        # Grab slice of sample_df_with_time that matches the conditions in 
        # sample_df (and thus the growth rates in bact_condition_k).
        c = sample_df.loc[sample_df.index[i],["replicate",
                                              "marker",
                                              "select",
                                              "iptg"]]
        mask = (st["replicate"] == c["replicate"]) & \
               (st["marker"] == c["marker"]) & \
               (st["select"] == c["select"]) & \
               (st["iptg"] == c["iptg"])

        # Get time steps for "grow_for_time calls"
        delta_time = np.array(st.loc[mask,"time"],dtype=float)
        delta_time[1:] = delta_time[1:] - delta_time[:-1]

        # Get sample ids for each time point
        sample_ids = st.index[mask]

        # Go through each sample
        ln_pop = ln_pop_array[:,i]
        for j in range(len(sample_ids)):
            
            # Grow ln_pop according to growth rates
            ln_pop = grow_for_time(ln_pop_array=ln_pop,
                                   growth_rates=bact_sample_k[:,i],
                                   t=delta_time[j])
            
            # Record population at this sample + time
            sample_pops[sample_ids[j]] = ln_pop

    return sample_pops