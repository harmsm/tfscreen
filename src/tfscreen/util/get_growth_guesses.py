from tfscreen import data

import numpy as np

def get_growth_guesses(iptg,
                       select,
                       marker):
    """
    Return a guess for the growth rate of a bacterium given iptg, selection, 
    and marker. 
    
    Parameters
    ----------
    iptg : float or list-like
        iptg values for calculation
    select : bool or list-like
        whether or not selection is applied to the sample
    marker : str or list-like   
        marker applied (must be a key in data.markers)

    Returns
    -------
    growth_rate_guess : float or np.ndarray
        growth rate guess. if single values are passed in, return a single value.
        Otherwise, return an array. 
    """

    return_single_value = False
    if not hasattr(iptg,"__iter__"):
        return_single_value = True
    
        iptg = [iptg]
        select = [int(select)]
        marker = [marker]
    
    growth_rate_guess = []
    for i in range(len(iptg)):
          
        iptg_value = float(iptg[i])
        select_value = int(select[i])
        marker_value = marker[i]
        wt_theta = data.wt_theta(iptg_value)

        wt_growth = data.wt_growth(iptg_value)
        marker_effect = data.markers[marker_value](wt_theta)
        select_effect = data.selectors[marker_value](wt_theta)

        growth_rate_guess.append(wt_growth + marker_effect + select_effect*select_value)

    growth_rate_guess = np.array(growth_rate_guess)
    if return_single_value:
        growth_rate_guess = growth_rate_guess[0]

    return growth_rate_guess
