"""
Functions for initializing bacterial populations for tfscreen simulations.
"""
from .cell_growth_moves import grow_to_target
from .cell_growth_moves import grow_for_time
from .cell_growth_moves import dilute
from .cell_growth_moves import thaw_glycerol_stock

import numpy as np
from tqdm.auto import tqdm

def _get_initial_pops(input_library,
                      num_thawed_colonies=1e7,
                      overnight_volume_in_mL=10):
    """
    Generate initial bacterial populations from an input library.

    Parameters
    ----------
    input_library : numpy.ndarray
        Array holding genotype(s) transformed into each bacterium in the population.
        Can be 1D (single genotype per bacterium) or 2D (multiple genotypes per bacterium).
    num_thawed_colonies : int, optional
        Number of colonies to thaw from the input library (default: 1e7).
    overnight_volume_in_mL : float, optional
        Overnight growth volume in mL (default: 10).

    Returns
    -------
    bacteria : numpy.ndarray
        Array of unique bacteria (genotype or tuple of genotypes per bacterium).
    ln_pop_array : numpy.ndarray
        Log population array for each unique bacterium.

    """


    # If input_library is a 1D array, there is one genotype per bacterium.
    if input_library.ndim == 1:

        bacteria, ln_pop_array = thaw_glycerol_stock(input_library,
                                                     num_thawed_colonies=num_thawed_colonies,
                                                     overnight_volume_in_mL=overnight_volume_in_mL)
        bacteria, counts = np.unique(bacteria,return_counts=True)
        ln_pop_array = np.log(np.exp(ln_pop_array[0])*counts)

    # If input_library is a 2D array, there are multiple genotypes per bacterium.
    else:

        # Create an array "to_thaw" that holds the indices of the bacteria to 
        # thaw. We will then use this to thaw the bacteria, remove duplicates,
        # and get the counts. The "indexer" and "reverse_indexer" dictionaries
        # are used to map the genotypes back to the original input_library.
        to_thaw = []
        indexer = {}
        for i in range(input_library.shape[0]):
            
            # Drop '-' empty slots and created a sorted tuple of genotypes in 
            # this bacterium. 
            tmp_genos = list(input_library[i][input_library[i] != "-"])
            tmp_genos.sort()
            tmp_genos = tuple(tmp_genos)
            
            # Record a previously unseen genotype
            if tmp_genos not in indexer:
                indexer[tmp_genos] = i
            
            # Add the index to the list of bacteria to thaw
            to_thaw.append(indexer[tmp_genos])

        to_thaw = np.array(to_thaw)
        reverse_indexer = dict(zip(indexer.values(),indexer.keys()))

        # Thaw and de-duplicate the bacteria just like we did for the 1D case
        bacteria, ln_pop_array = thaw_glycerol_stock(to_thaw,
                                                     num_thawed_colonies=num_thawed_colonies,
                                                     overnight_volume_in_mL=overnight_volume_in_mL)
        bacteria, counts = np.unique(bacteria,return_counts=True)
        ln_pop_array = np.log(np.exp(ln_pop_array[0])*counts)

        # Convert the bacteria back to genotype tuples
        bacteria = np.array([reverse_indexer[b] for b in bacteria],dtype=object)

    return bacteria, ln_pop_array
    

def _get_growth_rates(bacteria,
                      phenotype_df,
                      genotype_df,
                      sample_df,
                      growth_rate_noise=0):
    """
    Get growth rates for each bacterium in the population.

    Parameters
    ----------
    bacteria : numpy.ndarray
        Array of bacteria, each as a genotype string or tuple of genotype strings.
    phenotype_df : pandas.DataFrame
        DataFrame with phenotype information for all genotypes.
    genotype_df : pandas.DataFrame
        DataFrame with genotype, mutation, and site information for all clones.
    sample_df : pandas.DataFrame
        DataFrame describing all samples.
    growth_rate_noise : float
        percent noise to apply to all growth rates. (growth_noise*base_value is
        the standard deviation of the random normal distribution to sample)

    Returns
    -------
    bact_base_k : numpy.ndarray
        Base growth rates for each bacterium (no selection).
    bact_marker_k : numpy.ndarray
        Growth rates for each bacterium under marker expression (no selection).
        Shape (num_bacteria, num_samples).
    bact_sample_k : numpy.ndarray
        Growth rates for each bacterium under selection in this sample.
        Shape (num_bacteria, num_samples).
    """    

    desc = "{}".format(f"getting initial populations")
    with tqdm(total=3,desc=desc,ncols=800) as pbar:
        pbar.update()

        # Map genotype strings to indices in the genotype_df dataframe
        genotype_to_idx = dict(zip(list(genotype_df.index),
                            range(len(genotype_df.index))))

        # Number of samples and bacteria in the complete library
        num_samples = sample_df.shape[0]
        num_bacteria = len(bacteria)

        # Get base and marker growth rates for each genotype
        base_k = np.array(phenotype_df["base_growth_rate"])
        
        # Create marker_k array (num_genotypes,num_samples) with growth 
        # rates for each individual genotype when it's marker is expressed but not 
        # under selection.
        marker_k = np.array(phenotype_df["marker_growth_rate"])
        marker_k = np.reshape(marker_k,
                            (len(marker_k)//num_samples,
                            num_samples))

        # Create sample_k array (num_genotypes,num_samples) with growth 
        # rates for each individual genotype with marker and selection. 
        sample_k = np.array(phenotype_df["overall_growth_rate"])
        sample_k = np.reshape(sample_k,
                            (len(sample_k)//num_samples,num_samples))
            
        # If bacteria array is made of strings, each bacterium has a single 
        # genotype
        if issubclass(type(bacteria[0]),str):
            
            idx = np.array([genotype_to_idx[g] for g in bacteria])

            bact_base_k = base_k[idx]
            bact_marker_k = marker_k[idx]
            bact_sample_k = sample_k[idx]

        # If the bacteria array is not made of strings, bacteria have multiple 
        # genotypes. We need to calculate the average growth rate effects for each
        # bacterium.
        else:

            bact_base_k = np.zeros(num_bacteria)
            bact_marker_k = np.zeros((num_bacteria,num_samples),dtype=float)
            bact_sample_k = np.zeros((num_bacteria,num_samples),dtype=float)
            
            for i in range(len(bacteria)):
            
                idx = np.array([genotype_to_idx[g] for g in bacteria[i]])        
                
                bact_base_k[i] = np.mean(base_k[idx])
                bact_marker_k[i] = np.mean(marker_k[idx,:],axis=0)
                bact_sample_k[i] = np.mean(sample_k[idx,:],axis=0)
        
        pbar.update()

        # Add noise to the growth rates
        if growth_rate_noise > 0:

            bact_base_k = np.random.normal(loc=bact_base_k,
                                        scale=np.abs(bact_base_k)*growth_rate_noise)
            bact_marker_k = np.random.normal(loc=bact_marker_k,
                                            scale=np.abs(bact_marker_k)*growth_rate_noise)
            bact_sample_k = np.random.normal(loc=bact_sample_k,
                                            scale=np.abs(bact_sample_k)*growth_rate_noise)

        pbar.update()

    return bact_base_k, bact_marker_k, bact_sample_k


def initialize_population(input_library,
                          phenotype_df,
                          genotype_df,
                          sample_df,
                          num_thawed_colonies=1e7,
                          overnight_volume_in_mL=10,
                          pre_iptg_cfu_mL=90000000,
                          iptg_out_growth_time=30,
                          post_iptg_dilution_factor=0.2/10.2,
                          growth_rate_noise=0.0):
    """
    Initialize a bacterial population for tfscreen simulations, including
    thawing, growth, dilution, and induction steps.

    Parameters
    ----------
    input_library : numpy.ndarray
        Array holding genotype(s) transformed into each bacterium in the
        population.
    phenotype_df : pandas.DataFrame
        DataFrame with phenotype information for all genotypes and conditions.
    genotype_df : pandas.DataFrame
        DataFrame with genotype, mutation, and site information for all clones.
    sample_df : pandas.DataFrame
        DataFrame describing all samples
    num_thawed_colonies : int, optional
        Number of colonies to thaw from the input library (default: 1e7).
    overnight_volume_in_mL : float, optional
        Overnight growth volume in mL (default: 10).
    pre_iptg_cfu_mL : float, optional
        Target cfu/mL before IPTG induction (default: 9e7).
    iptg_out_growth_time : float, optional
        Out-growth time in IPTG (default: 30).
    post_iptg_dilution_factor : float, optional
        Dilution factor after IPTG induction (default: 0.2/10.2).
    growth_rate_noise : float
        percent noise to apply to all growth rates. (growth_noise*base_value is
        the standard deviation of the random normal distribution to sample)

    Returns
    -------
    bacteria : numpy.ndarray
        Array holding bacterial genotypes in the population after initialization.
    ln_pop_array : numpy.ndarray
        Log population array (n_bacteria,n_samples) for each bacterium after
        initialization.
    bact_sample_k : numpy.ndarray
        Growth rate array (n_bacteria,n_samples) of cells in each selection
        sample after initialization.
    """

    desc = "{}".format("initializing sample")
    with tqdm(total=3,desc=desc,ncols=800) as pbar:
    
        # Thaw glycerol stock and get initial populations
        bacteria, ln_pop_array = _get_initial_pops(input_library,
                                                num_thawed_colonies=num_thawed_colonies,
                                                overnight_volume_in_mL=overnight_volume_in_mL)

        pbar.update()

        # Get base growth rates and sample-specific growth rates for each
        # bacterium in the population.
        bact_base_k, bact_marker_k, bact_sample_k = _get_growth_rates(bacteria,
                                                                        phenotype_df,
                                                                        genotype_df,
                                                                        sample_df,
                                                                        growth_rate_noise=growth_rate_noise)

        # Grow to pre-induction OD600 (0.35 ~ 90,000,000 CFU/mL) using the base
        # growth rates for all bacteria.
        ln_pop_array = grow_to_target(ln_pop_array,
                                    growth_rates=bact_base_k,
                                    final_cfu_mL=pre_iptg_cfu_mL)

        pbar.update()

        # Get the IPTG concentrations for each sample
        iptg_concs = np.array(sample_df["iptg"],dtype=float)
        num_samples = len(iptg_concs)

        # Split culture into iptg matching samples, without selection, and then
        # grow for the specified out-growth time.
        ln_pop_array = np.stack([ln_pop_array]*num_samples,axis=1)
        for i in range(num_samples):
            ln_pop_array[:,i] = grow_for_time(ln_pop_array[:,i],
                                            growth_rates=bact_marker_k[:,i],
                                            t=iptg_out_growth_time)
            
        # Dilute as we drop into selection conditions 
        ln_pop_array = dilute(ln_pop_array,
                            dilution_factor=post_iptg_dilution_factor)

        pbar.update()

    return bacteria, ln_pop_array, bact_sample_k
