import numpy as np

import copy

def _transform_cells(num_transformants,
                     library_vector,
                     lambda_value=None,
                     max_num_plasmids=10):
    
    raw_genotypes = np.random.choice(library_vector,
                                     size=(num_transformants,max_num_plasmids))
    
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
    
    col_indexes = np.arange(raw_genotypes.shape[1])
    
    # Create a mask where True means "set to -1"
    # Broadcasting creates a comparison for each row with all columns
    mask = col_indexes >= slice_at[:, np.newaxis]
    raw_genotypes[mask] = -1

    return raw_genotypes



def _scale_library_mixture(library_mixture):
    
    library_mixture = copy.deepcopy(library_mixture)

    # Create vector from entries
    v = []
    for k in library_mixture:
        v.append(library_mixture[k])
    v = np.array(v)

    # Regularize so lowest scale has 1 entry
    v = np.array(np.round(v/ np.min(v),0),dtype=int)

    # Update dictionary
    for i, k in enumerate(library_mixture):
        library_mixture[k] = v[i]

    return library_mixture
    

def transform_and_mix(lib_phenotypes,
                      transform_sizes,
                      library_mixture,
                      lambda_value=0,
                      max_num_plasmids=10):

    if lambda_value is None:
        lambda_value = 0

    # error checking
    ts_keys = set(transform_sizes.keys())
    lm_keys = set(library_mixture.keys())
    if ts_keys != lm_keys:
        err = "transform_sizes and library_mixture must have identical keys.\n"
        raise ValueError(err)

    if not ts_keys.issubset(set(lib_phenotypes.keys())):
        err = "all keys in transform_sizes must be in libraries.\n"
        raise ValueError(err)

    # Create a single list with all possible clones
    index_dict = {}
    mega_library = []
    for k in ts_keys:
        idx_i = len(mega_library)
        idx_j = idx_i + len(lib_phenotypes[k])
        index_dict[k] = (idx_i,idx_j)
        mega_library.extend(lib_phenotypes[k])

    # Set up library mixtures
    library_mixture = _scale_library_mixture(library_mixture)
    
    to_combine = []
    for k in transform_sizes:

        # Get indexes to sample from
        idx_i, idx_j = index_dict[k]

        # Transform library, sampling appropriately
        transformed = _transform_cells(int(transform_sizes[k]),
                                       library_vector=np.arange(idx_i,idx_j,1,dtype=int),
                                       lambda_value=lambda_value,
                                       max_num_plasmids=max_num_plasmids)
        
        # Now repeat each clone the right number of times to match the library
        # mixture
        transformed = np.repeat(transformed,repeats=library_mixture[k],axis=0)

        # Set up to combined
        to_combine.append(transformed)

    input_library = np.concatenate(to_combine)

    # Shuffle order of library
    idx = np.arange(input_library.shape[0])
    np.random.shuffle(idx)
    input_library = input_library[idx,:]

    if lambda_value <= 0:
        input_library = np.array(input_library[:,0])

    return input_library, mega_library

