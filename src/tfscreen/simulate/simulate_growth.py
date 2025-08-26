"""
Functions for simulating the growth of bacteria during a screen.
"""

import numpy as np
from tqdm.auto import tqdm

from .cell_growth_moves import grow_for_time

def simulate_growth(ln_pop_array,
                    bact_condition_k,
                    condition_df,
                    condition_df_with_time):
    """
    Simulate bacterial population growth across different conditions over time.

    This function simulates the growth of bacterial populations under various
    conditions, using growth rates specified for each condition and time point.
    It iterates through each condition, calculates the time steps, and applies
    the growth simulation to update the population sizes.

    Parameters
    ----------
    ln_pop_array : numpy.ndarray
        Array of log-transformed initial population sizes for each bacterium
        across all conditions (n_bacteria, n_conditions).
    bact_condition_k : numpy.ndarray
        Array of growth rates for each bacterium under each condition
        (n_bacteria, n_conditions).
    condition_df : pandas.DataFrame
        DataFrame containing condition information (replicate, marker, select,
        iptg) used to filter time-dependent conditions.
    condition_df_with_time : pandas.DataFrame
        DataFrame containing condition information at each time point,
        including 'replicate', 'marker', 'select', 'iptg', and 'time'.

    Returns
    -------
    condition_pops : dict
        Dictionary mapping condition IDs to log-transformed population sizes
        (ln_pop) at the end of the growth simulation for that condition.
    """

    print("Growing populations in specified conditions",flush=True)

    # ct will hold all condition-level results (cfu/mL and num reads)
    ct = condition_df_with_time.copy()
    
    # Dictionary to store populations under all conditions
    condition_pops = {}

    # Go through each condition
    for i in tqdm(range(ln_pop_array.shape[1])):
        
        # Grab slice of condition_df_with_time that matches the conditions in 
        # condition_df (and thus the growth rates in bact_condition_k).
        c = condition_df.loc[condition_df.index[i],["replicate",
                                                    "marker",
                                                    "select",
                                                    "iptg"]]
        mask = (ct["replicate"] == c["replicate"]) & \
            (ct["marker"] == c["marker"]) & \
            (ct["select"] == c["select"]) & \
            (ct["iptg"] == c["iptg"])

        # Get time steps for "grow_for_time calls"
        delta_time = np.array(ct.loc[mask,"time"],dtype=float)
        delta_time[1:] = delta_time[1:] - delta_time[:-1]

        # Get condition ids for each time point
        condition_ids = ct.index[mask]

        # Go through each condition
        ln_pop = ln_pop_array[:,i]
        for j in range(len(condition_ids)):
            
            # Grow ln_pop according to growth rates
            ln_pop = grow_for_time(ln_pop_array=ln_pop,
                                   growth_rates=bact_condition_k[:,i],
                                   t=delta_time[j])
            
            # Record population at this condition + time
            condition_pops[condition_ids[j]] = ln_pop

    return condition_pops