"""
Functions for initializing bacterial populations for tfscreen simulations.
"""
from tfscreen import grow_to_target
from tfscreen import grow_for_time
from tfscreen import dilute
from tfscreen import thaw_glycerol_stock

import numpy as np
from tqdm.auto import tqdm


def _get_growth_rates(input_library,
                      all_genotypes,
                      selectors):
    """
    Get the growth rates of a population of cells that may have multiple
    plasmids per cell.

    Parameters
    ----------
    input_library : numpy.ndarray
        Array holding bacterial clones in the population.
    all_genotypes : list
        List of all genotypes with full genotype information.
    selectors : list
        List of selectors to apply (e.g., ["kan", "pheS"]). These should
        match selectors created by pheno_to_growth calls.

    Returns
    -------
    base_growth_rate : numpy.ndarray
        Growth rates of cells in the population in the absence of selection.
    growth_rates : dict
        Dictionary mapping selector to growth rate array for each clone.
    """

    print("getting growth rates of each bacterium",flush=True)
    
    growth_rates = dict([(s,[0.0 for _ in range(len(input_library))]) for s in selectors])
    base_growth_rate = np.zeros(len(input_library),dtype=float)

    # Decide if this is a single plasmid
    single_plasmid = True
    if len(input_library.shape) > 1:
        single_plasmid = False


    for i in tqdm(range(len(input_library))):
        
        if single_plasmid:

            # Get the genotype
            genotype = all_genotypes[input_library[i]]

            # Record the genotype base growth rate and growth under selection 
            # conditions
            base_growth_rate[i] = genotype["base_growth_rate"]
            for sel_name in growth_rates:
                growth_rates[sel_name][i] = genotype[sel_name]
        
            continue

        # If we get there, there are multiple plasmids. Get the indexes of the
        # genotypes for each plasmid.
        genotype_indexes = input_library[i,input_library[i,:] > -1]
        num_gentoypes = len(genotype_indexes)

        # Go through each genotype
        for j in genotype_indexes:

            # Sum the log of the growth rates for the base growth and selection
            # growth
            base_growth_rate[i] += np.log(all_genotypes[j]["base_growth_rate"])
            for sel_name in growth_rates:
                growth_rates[sel_name][i] += np.log(all_genotypes[j][sel_name])

        # Now divide by number of genotypes and take exponent (geometric mean)
        base_growth_rate[i] = np.exp(base_growth_rate[i]/num_gentoypes)
        for sel_name in growth_rates:
            growth_rates[sel_name][i] = np.exp(growth_rates[sel_name][i]/num_gentoypes)

    # Convert growth rates into a matrix
    for selector in growth_rates:
        growth_rates[selector] = np.array(growth_rates[selector])
                         
    return base_growth_rate, growth_rates


def initialize_population(input_library,
                          all_genotypes,
                          num_thawed_colonies=1e7,
                          overnight_volume_in_mL=10,
                          saturation_cfu_mL=1e9,
                          morning_dilution=1/70,
                          pre_iptg_cfu_mL=90000000,
                          iptg_dilution_factor=0.2/10.2,
                          iptg_out_growth_time=30):
    
    """
    Initialize a bacterial population for tfscreen simulations, including
    thawing, growth, dilution, and induction steps.

    Parameters
    ----------
    input_library : numpy.ndarray
        Array holding bacterial clones in the population.
    all_genotypes : list
        List of all genotypes with full genotype information.
    num_thawed_colonies : int, optional
        Number of colonies to thaw from the input library (default: 1e7).
    overnight_volume_in_mL : float, optional
        Overnight growth volume in mL (default: 10).
    saturation_cfu_mL : float, optional
        Final cfu/mL for overnight saturation (default: 1e9).
    morning_dilution : float, optional
        Dilution factor for morning culture (default: 1/70).
    pre_iptg_cfu_mL : float, optional
        Target cfu/mL before IPTG induction (default: 9e7).
    iptg_dilution_factor : float, optional
        Dilution factor for IPTG induction (default: 0.2/10.2).
    iptg_out_growth_time : float, optional
        Out-growth time in IPTG (default: 30).

    Returns
    -------
    input_library : numpy.ndarray
        Array holding bacterial clones in the population after initialization.
    ln_pop_array : numpy.ndarray
        Log population array for each clone after initialization.
    base_growth_rates : numpy.ndarray
        Growth rates of cells in the absence of selection.
    growth_rates : dict
        Dictionary mapping selector to growth rate array for each clone.
    """
    # Thaw glycerol stock, 
    input_library, ln_pop_array = thaw_glycerol_stock(input_library,
                                                      num_thawed_colonies=num_thawed_colonies,
                                                      overnight_volume_in_mL=overnight_volume_in_mL)

    # Get final growth rates for population of cells under the specified 
    # selection conditions.
    keys = list(all_genotypes[0].keys())
    selectors = [k for k in keys if k.startswith("sel_")]
    base_growth_rates, growth_rates = _get_growth_rates(input_library,
                                                        all_genotypes,
                                                        selectors=selectors)

    # Grow to saturation overnight
    ln_pop_array = grow_to_target(ln_pop_array,
                                  base_growth_rates,
                                  final_cfu_mL=saturation_cfu_mL)

    # Dilute culture in morning
    ln_pop_array = dilute(ln_pop_array,
                          dilution_factor=morning_dilution)


    # Grow to pre-induction OD600 (0.35 ~ 90,000,000 CFU/mL)
    ln_pop_array = grow_to_target(ln_pop_array,
                                  base_growth_rates,
                                  final_cfu_mL=pre_iptg_cfu_mL)

    # Dilute growing cultures into tubes with IPTG
    ln_pop_array = dilute(ln_pop_array,
                          dilution_factor=iptg_dilution_factor)

    # Out-growth in IPTG
    ln_pop_array = grow_for_time(ln_pop_array,
                                 base_growth_rates,
                                 t=iptg_out_growth_time)

    return input_library, ln_pop_array, base_growth_rates, growth_rates