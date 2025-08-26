"""
Functions for simulating transformation and mixing of mutant libraries in tfscreen.
"""
import numpy as np
from scipy.stats import lognorm 
import copy

def _transform_cells(num_transformants,
                     library_vector,
                     skew_sigma=0,
                     lambda_value=None,
                     max_num_plasmids=10):
    """
    Simulate transformation of cells with a library of plasmids.

    This function simulates the process of transforming bacterial cells with a
    library of plasmids. It models the number of plasmids entering each cell
    based on a Poisson distribution (if `lambda_value` is provided) or assumes
    each cell receives one plasmid.

    Parameters
    ----------
    num_transformants : int
        Number of transformant cells to generate.
    library_vector : numpy.ndarray
        Vector of possible plasmid indexes to sample from.  This represents the
        pool of available plasmids in the library.
    skew_sigma : float, default = 0
        sigma value to give to log normal distribution for generating genotype
        frequencies in the library. 0 gives an even distribution; 1-2 is a good
        guess for a real library. 
    lambda_value : float or None, optional
        Mean number of plasmids per cell (Poisson-distributed). If None or
        <= 0, each cell gets one plasmid. Default is None.
    max_num_plasmids : int, optional
        Maximum number of plasmids a cell can contain.  This limits the number
        of plasmids assigned even if the Poisson distribution suggests more.
        Default is 10.

    Returns
    -------
    bacteria : numpy.ndarray
        Array of shape (num_transformants, max_num_plasmids) with plasmid
        names. '-' indicates no plasmid in that slot. Each row represents a
        single cell, and each column represents a plasmid slot within that cell.
    """

    # Build counts for each entry in library_vector. These are even or sampled
    # from a log normal distribution with sigma = skew_sigma
    if skew_sigma == 0 or skew_sigma is None:
        raw_counts = np.ones(len(library_vector))
    else:
        dist = lognorm(s=skew_sigma,scale=1,loc=0)
        raw_counts = dist.rvs(size=len(library_vector))

    # Get frequency of each genotype in the pre-transformation library. 
    frequencies = raw_counts/np.sum(raw_counts)
    
    # Make a pool of bacteria sampled from the library vector. 
    bacteria = np.random.choice(library_vector,
                                size=(num_transformants,max_num_plasmids),
                                p=frequencies)
    
    # The goal of this block is to create a "slice_at" index array. This selects
    # how many plasmids each bacterium has based on a Poisson distribution. The
    # reason we do it this way is because we are going to filter out all 
    # bacteria that have zero plasmids. So, we make 100*num_transformants 
    # samples then drop any samples with zero plasmids.
    if lambda_value is not None and lambda_value > 0:
    
        if lambda_value < 1:
            make_extra = int(1/lambda_value)*100*num_transformants
        else:
            make_extra = 100*num_transformants
        
        slice_at = np.random.poisson(lambda_value,make_extra)
        slice_at = slice_at[slice_at > 0]
        slice_at = slice_at[:num_transformants]
    else:
        slice_at = np.ones(num_transformants,dtype=int)
    
    slice_at[slice_at > max_num_plasmids] = max_num_plasmids
    
    col_indexes = np.arange(bacteria.shape[1])
    
    # Create a mask where True means "set to -"
    # Broadcasting creates a comparison for each row with all columns
    mask = col_indexes >= slice_at[:, np.newaxis]
    bacteria[mask] = '-'

    return bacteria

def _scale_library_mixture(library_mixture):
    """
    Scale the library mixture so the lowest entry is 1 and others are
    proportional.

    Parameters
    ----------
    library_mixture : dict
        Dictionary mapping library names to mixture sizes.

    Returns
    -------
    library_mixture : dict
        Scaled dictionary with integer mixture sizes.
    """

    library_mixture = copy.deepcopy(library_mixture)

    # Create vector from entries
    v = []
    for k in library_mixture:
        v.append(library_mixture[k])
    v = np.array(v)

    # Regularize so lowest scale has 1 entry
    v = np.array(np.round(v/np.min(v),0),dtype=int)

    # Update dictionary
    for i, k in enumerate(library_mixture):
        library_mixture[k] = v[i]

    return library_mixture
    

def transform_and_mix(libraries,
                      transform_sizes,
                      library_mixture,
                      skew_sigma=None,
                      lambda_value=0,
                      max_num_plasmids=10):
    """
    Transform and mix multiple libraries into a single input library.

    This function simulates the process of transforming multiple mutant libraries
    into bacterial cells and then mixing them to create a single, combined
    input library. It allows for specifying different transformation sizes and
    mixing ratios for each library. Each input library has a frequency 
    distribution set by skew_sigma. The function also models the number of
    plasmids entering each cell based on a Poisson distribution (if
    `lambda_value` is provided) or assumes each cell receives one plasmid.

    Parameters
    ----------
    libraries : dict
        Dictionary mapping library names to lists or arrays of genotype indexes
        to sample from. This represents the pool of available genotypes in each
        library.
    transform_sizes : dict
        Dictionary mapping library names to the number of transformants for each
        library. This determines how many cells are transformed with each
        individual library.
    library_mixture : dict
        Dictionary mapping library names to mixture sizes (relative proportions).
        These values determine the relative abundance of each library in the
        final mixed population. The function `_scale_library_mixture` ensures
        that the lowest entry is 1 and others are proportional.
    skew_sigma : float, default = 0
        sigma value to give to log normal distribution for generating genotype
        frequencies in the library. 0 gives an even distribution; 1-2 is a good
        guess for a real library. 
    lambda_value : float, default=0
        Mean number of plasmids per cell (Poisson-distributed). If None or
        <= 0, each cell gets one plasmid. Default is 0.
    max_num_plasmids : int, optional
        Maximum number of plasmids per cell. This limits the number of plasmids
        assigned even if the Poisson distribution suggests more. Default is 10.

    Returns
    -------
    input_library : numpy.ndarray
        Array holding bacterial clones in the population. If lambda_value <= 0,
        this is a 1D array (one plasmid per cell). If lambda_value > 0, this is
        a 2D array (cells x max_num_plasmids), with '-' indicating empty slots.
        The order of the input library is randomized to remove any structure
        from the separate transformations.
    """

    if lambda_value is None:
        lambda_value = 0

    # error checking
    ts_keys = set(transform_sizes.keys())
    lm_keys = set(library_mixture.keys())
    if ts_keys != lm_keys:
        err = "transform_sizes and library_mixture must have identical keys.\n"
        raise ValueError(err)

    if not ts_keys.issubset(set(libraries.keys())):
        err = "all keys in transform_sizes must be in libraries.\n"
        raise ValueError(err)

    # Set up library mixtures
    library_mixture = _scale_library_mixture(library_mixture)
    
    to_combine = []
    for k in transform_sizes:

        # Transform library, sampling appropriately
        transformed = _transform_cells(int(transform_sizes[k]),
                                       library_vector=libraries[k],
                                       skew_sigma=skew_sigma,
                                       lambda_value=lambda_value,
                                       max_num_plasmids=max_num_plasmids)
        
        # Now repeat each clone the right number of times to match the library
        # mixture
        transformed = np.repeat(transformed,repeats=library_mixture[k],axis=0)

        # Set up to combined
        to_combine.append(transformed)

    input_library = np.concatenate(to_combine)

    # Shuffle order of library to scramble any structure in the data that came 
    # from the separate transformations
    idx = np.arange(input_library.shape[0])
    np.random.shuffle(idx)
    input_library = input_library[idx,:]

    if lambda_value <= 0:
        input_library = np.array(input_library[:,0])

    return input_library

