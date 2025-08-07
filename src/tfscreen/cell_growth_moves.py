"""
Functions for simulating cell growth, dilution, and related operations in tfscreen.
"""
import numpy as np

def thaw_glycerol_stock(input_library,
                        num_thawed_colonies=1e7,
                        overnight_volume_in_mL=10):
    """
    Thaw the glycerol stock (setting initial populations).

    Parameters
    ----------
    input_library : numpy.ndarray
        array holding bacterial clones in the population
    num_thawed_colonies : int, default = 1e7
        sample this number of colonies from the input library
    overnight_volume_in_mL : float, default = 10
        overnight growth volume. (This sets cfu/mL for rest of experiemnts).

    Returns
    -------
    input_library : numpy.ndarray
        array holding bacterial clones in the population. This could be smaller
        than the input if we took a smaller number of colonies than the 
        initial input_library
    ln_pop_array : numpy.ndarray
        1D numpy array with the ln(cfu/mL) for each member of the population
    """

    # Get integer number of thawed colonies
    num_thawed_colonies = int(np.round(num_thawed_colonies,0))

    # If the number of colonies is smaller than the input library, chop off the
    # end of the library. (Random order, so this is effectively a random sample).
    if num_thawed_colonies < input_library.shape[0]:
        if len(input_library.shape) == 2:
            input_library = input_library[:num_thawed_colonies,:]
        else:
            input_library = input_library[:num_thawed_colonies]

    # Get the cfu/mL for each clone
    clone_cfu_per_mL = num_thawed_colonies/input_library.shape[0]/overnight_volume_in_mL

    # Make initial population array
    ln_pop_array = np.log(clone_cfu_per_mL)*np.ones(len(input_library),dtype=float)

    return input_library, ln_pop_array

def grow_to_target(ln_pop_array,
                   growth_rates,
                   final_cfu_mL=1e9):
    """
    Grow a population vector proportionally to a final cfu/mL.

    Parameters
    ----------
    ln_pop_array : numpy.ndarray
        array with current ln(pop) of each clone
    growth_rates : numpy.ndarray
        array with growth rate of each clone under this condition
    final_cfu_mL : float, default = 1e9
        grow to this cfu/mL

    Returns
    -------
    ln_pop_array : numpy.ndarray
        updated log population array
    """

    # Growth follows A(t) = A0 + t*k. 
    # t = [sum(A(t)) - sum(A0)]/sum(k)
    
    current = np.sum(ln_pop_array)
    target = np.log(final_cfu_mL)
    all_growth = np.sum(growth_rates)

    t = (target - current)/all_growth

    return growth_rates*t + ln_pop_array
    

def dilute(ln_pop_array,dilution_factor):
    """
    Dilute a population by a dilution factor. 
    """
    
    return np.log(np.exp(ln_pop_array)*dilution_factor)

def grow_for_time(ln_pop_array,
                  growth_rates,
                  t):
    """
    Grow a population vector for some amount of time.

    Parameters
    ----------
    ln_pop_array : numpy.ndarray
        array with current ln(pop) of each clone
    growth_rates : numpy.ndarray
        array with growth rate of each clone under this condition
    t : float
        time (in units matching growth rate; minutes, usually) to grow

    Returns
    -------
    ln_pop_array : numpy.ndarray
        updated log population array
    """

    return growth_rates*t + ln_pop_array

def get_cfu(ln_pop_array):
    """
    Get the total cfu/mL from a full ln population array.
    """

    return np.sum(np.exp(ln_pop_array))