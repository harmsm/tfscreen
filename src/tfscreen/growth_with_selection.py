"""
Functions for simulating population growth under selection conditions in tfscreen.
"""
import numpy as np
from tqdm.auto import tqdm
from tfscreen import grow_for_time

def growth_with_selection(ln_pop_array,
                          growth_rates,
                          time_points):
    """
    Given some starting populations, grow under conditions in growth rates 
    and record populations at times in time points. 
    
    Parameters
    ----------
    ln_pop_array : numpy.ndarray
        1D numpy array with natural logs of genotype populations
    growth_rates : dict
        dictionary whose keys are selectors. values are 2D numpy arrays with
        growth rates (num_clones x num_iptg)
    time_points : list-like
        time points to sample (float, minutes from inoculation)

    Returns
    -------
    pops_vs_time : dict
        dictionary keying selector to results. values are
        num_clones x num_iptg x num_time_points arrays of ln_pop.
    
    """

    print("simulating growth under all conditions",flush=True)
    
    orig_ln_pop_array = np.copy(ln_pop_array)

    num_conditions = len(growth_rates)*len(time_points)
    with tqdm(total=num_conditions) as pbar:
    
        pops_vs_time = {}
        for selector in growth_rates:
    
            pops_vs_time[selector] = []
            
            N = growth_rates[selector].shape[1]
            ln_pop_matrix = np.stack([orig_ln_pop_array for _ in range(N)]).T
        
            for t in time_points:
            
                m = grow_for_time(ln_pop_matrix,
                                  growth_rates[selector],
                                  t=t)
                pops_vs_time[selector].append(m)

                pbar.update(1)
                
            pops_vs_time[selector] = np.array(pops_vs_time[selector])
                
    return pops_vs_time
