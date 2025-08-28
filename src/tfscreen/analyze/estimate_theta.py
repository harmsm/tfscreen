from tfscreen.calibration import predict_growth_rate
from tfscreen.fitting import (
    matrix_wls,
    matrix_nls
)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def _chunk_by_group(arr, max_chunk_size):
    """
    Splits an array into chunks of max size N without breaking groups of
    identical values. Returns lists of numpy arrays of indexes for the chunks.
    It assumes arr values are sorted by value, that values can repeat, but
    that the number of each value is different. 
    
    For example: [0,0,0,1,1,2,2,2,2]

    Parameters
    ----------
    arr : np.ndarray
        A 1D sorted array with repeating values.
    max_chunk_size : int
        the maximum size for any chunk.

    Returns
    -------
    chunks : list
        A list of NumPy arrays with indexes representing the chunks.
    """
    
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    max_chunk_size = int(max_chunk_size)
    if max_chunk_size < 1:
        err = "max_chunk_size must be 1 or greater.\n"
        raise ValueError(err)

    # Find the start of each new group of numbers.
    group_starts = np.where(arr[:-1] != arr[1:])[0] + 1

    # Create a full list of boundaries including the start (0) and end of the
    # array. 
    boundaries = np.concatenate(([0], group_starts, [len(arr)]))

    # Iterate through boundaries to find split points.
    split_indices = []
    current_chunk_start = 0  
    for i in range(1, len(boundaries)):
        
        # If the current group's end minus the chunk's start exceeds
        # max_chunk_size...
        if boundaries[i] - boundaries[current_chunk_start] > max_chunk_size:
            
            # ...then we must split at the previous group's end.
            split_point = boundaries[i-1]
            split_indices.append(split_point)
            
            # The new chunk will start from that split point.
            current_chunk_start = i-1
            
    # Return indexes split into chunks
    indexes = np.arange(len(arr),dtype=int)
    return np.split(indexes, split_indices)

                    
def _multi_genotype_regression(y,
                               weights,
                               iptg,
                               slopes,
                               genotypes_as_idx,
                               method="nls"):
    """
    """
        
    # Get the number of observations and genotypes
    num_obs = len(y)
    num_genotypes = len(np.unique(genotypes_as_idx))

    # This holds the first column in the design matrix for the current 
    # genotype
    genotype_start_column = 0

    # mut_param_mapper holds the column index for the global mutation effect 
    # of each mutation. It will be num_obs long.
    mut_param_mapper = []

    # theta_param_mapper holds the column index for the theta (fractional 
    # saturation) parameter for for each genotype + iptg condition. It will
    # be num_obs long.
    theta_param_mapper = []

    # parameter indexes, iptg concs, and degrees of freedom for each genotype
    genotype_idx = []
    genotype_iptg = []

    guesses = []
    lower_bounds = []
    upper_bounds = []
        
    # Go through all unique genotypes...
    for i in pd.unique(genotypes_as_idx):

        # Get mask with indexes pointing to this genotype
        param_mask = genotypes_as_idx == i
        num_this_genotype = np.sum(param_mask)

        # Record the first column for this genotype (will point to the 
        # parameter capturing the global effect of the mutation). 
        mut_param_mapper.extend(np.repeat(genotype_start_column,
                                          num_this_genotype))

        # Get the iptg concentrations for this genotype
        this_iptg = iptg[param_mask]
        unique_iptg = pd.unique(iptg[param_mask])
        num_unique_iptg = len(unique_iptg)

        # Map iptg concentration of this condition to the index pointing to
        # the appropriate theta
        theta_dict = dict([(iptg_conc,j + genotype_start_column + 1)
                           for j, iptg_conc in enumerate(unique_iptg)])
        theta_param_mapper.extend([theta_dict[key] for key in this_iptg])

        # Number of new parameters we needed
        num_new_params = 1 + num_unique_iptg

        # Record parameter indexes and iptg concentrations for this genotype
        genotype_idx.append(np.arange(genotype_start_column,
                                      genotype_start_column + num_new_params,
                                      dtype=int))
        genotype_iptg.append(unique_iptg)

        guesses.append(0.01)
        lower_bounds.append(-np.inf)
        upper_bounds.append(np.inf)
        
        guesses.extend(0.5*np.ones(num_unique_iptg))
        lower_bounds.extend(np.zeros(num_unique_iptg))
        upper_bounds.extend(np.ones(num_unique_iptg))
        
        # The next genotype will start in a new block after all of the
        # parameters for the current genotype
        genotype_start_column += num_new_params
        
    
    # The final genotype_start_column will hold the total number of parameters
    num_params = genotype_start_column
    
    mut_param_mapper = np.array(mut_param_mapper,dtype=int)
    theta_param_mapper = np.array(theta_param_mapper,dtype=int)

    # Create empty design matrix X
    X = np.zeros((num_obs, num_params),dtype=float)

    # All rows indexer
    row_indexer = np.arange(num_obs,dtype=int)

    # First column for each genotype under all conditions is 1.0. This is
    # the global growth offset due to the mutation. 
    X[row_indexer,mut_param_mapper] = 1.0

    # Remaining columns for the genotype hold the slopes relating theta to 
    # observable for the specific conditions
    X[row_indexer,theta_param_mapper] = slopes

    # Do regression based on specified method
    if method == "wls":
        all_parameters, cov = matrix_wls(X,
                                         y,
                                         weights)
    elif method == "nls":
        all_parameters, cov = matrix_nls(X,
                                         y,
                                         weights,
                                         lower_bounds,
                                         upper_bounds,
                                         guesses)
    else:
        err = f"method '{method}' not recognized.\n"
        raise ValueError(err)
        
    # Standard errors are the diagonal of the covariance matrix
    all_std_errors = np.sqrt(np.diag(cov))
    
    out_genotypes = []
    out_mut_est = []
    out_mut_std = []
    out_iptg = []
    out_theta_est = []
    out_theta_std = []
    for i in range(len(genotype_idx)):

        params = all_parameters[genotype_idx[i]]
        std_errors = all_std_errors[genotype_idx[i]]

        out_genotypes.append(i)
        out_mut_est.append(params[0])
        out_mut_std.append(std_errors[0])
        out_iptg.append(genotype_iptg[i])
        out_theta_est.append(params[1:])
        out_theta_std.append(std_errors[1:])

    return out_genotypes, out_mut_est, out_mut_std, out_iptg, out_theta_est, out_theta_std


def estimate_theta(df,
                   calibration_dict,
                   block_size=100,
                   method="nls"):
    """
    This function estimates the global effect of each mutation on growth, as 
    well as the fractional occupancy of the binding site across all iptg 
    conditions. 

    Parameters
    ----------
    df : pandas.DataFrame
        genotype, k_est, k_std, iptg, marker, select
    calibration : dict
    block_size : int, default = 100
        break into blocks of block_size genotypes (each genotype is
        independent, but pooling them speeds up the array operations--to a
        point. If we have too many many genotypes, the array operations
        become prohibitively slow. 
    method : str, default="nls"
        should be "nls" (non-linear least squares) or "wls" (weighted linear 
        least squares). 

    Returns
    -------
    growth_out : pandas.DataFrame
        dataframe with columns "genotype", "mut_effect", and "mut_effect_std"
    theta_out : pandas.DataFrame
        dataframe with columns "genotype", "iptg", "theta", and "theta_std"
    """
    
    # Work on a copy of the dataframe
    df = df.copy()

    # Get rid of nan
    df = df.loc[np.logical_not(np.isnan(df["k_est"])),:]
    df = df.loc[np.logical_not(np.isnan(df["k_std"])),:]
    
    # Get estimated k and standard error on estimated k
    k_est = np.array(df["k_est"])
    k_std = np.array(df["k_std"])

    # Get iptg, marker, and selection data for each condition
    iptg = np.array(df["iptg"])
    marker = np.array(df["marker"])
    select = np.array(df["select"])

    # get wildtype growth rate under each of these conditions
    k_wt, _ = predict_growth_rate(marker=np.array(["none" for _ in range(len(marker))]),
                                  select=select,
                                  iptg=iptg,
                                  calibration_dict=calibration_dict,
                                  calc_err=False)

    param_values = calibration_dict["param_values"]
    param_idx_dict = dict([(p,i) for i,p in enumerate(calibration_dict["param_names"])])

    intercepts = []
    slopes = []
    for m, s in zip(df["marker"],df["select"]):
        b_idx = param_idx_dict[f"{m}|{s}_b"]
        intercepts.append(param_values[b_idx])
        
        m_idx = param_idx_dict[f"{m}|{s}_m"]
        slopes.append(param_values[m_idx])

    intercepts = np.array(intercepts)
    slopes = np.array(slopes)
    
    # Growth for wildtype with zero occupancy operator under these conditions
    k_wt0 = k_wt + intercepts 
    
    # y is the change in growth rate due to the global effect of the mutation
    # and the shift on occupancy away from theta = 0. 
    y = k_est - k_wt0

    # weights are 1/k_std
    weights = (1/k_std**2)
    
    # Convert genotypes to integer indexes
    genotypes = np.array(df["genotype"])
    idx_to_genotype = np.array(pd.unique(genotypes))
    genotype_to_idx = dict([(k,i) for i, k in enumerate(idx_to_genotype)])
    genotypes_as_idx = np.array([genotype_to_idx[g] for g in genotypes],dtype=int)

    # Break into chunks governed by block_size
    chunks = _chunk_by_group(genotypes_as_idx,block_size)

    # Prepare output
    growth_out = {"genotype":[],
                  "mut_effect":[],
                  "mut_effect_std":[]}
    theta_out = {"genotype":[],
                 "iptg":[],
                 "theta":[],
                 "theta_std":[]}
    
    for i, chunk in enumerate(tqdm(chunks)):

        # values to fit against in this chunk
        chunk_y = y[chunk]
        chunk_weights = weights[chunk]

        # iptg and slopes in this chunk
        chunk_iptg = iptg[chunk]
        chunk_slopes = slopes[chunk]

        # chunk_genotype_idx indexes from 0 to the number of genotypes
        # in this chunk. 
        _genotype_idx = genotypes_as_idx[chunk]
        chunk_genotypes_as_idx = _genotype_idx - np.min(_genotype_idx)
    
        result = _multi_genotype_regression(y=chunk_y,
                                            weights=chunk_weights,
                                            iptg=chunk_iptg,
                                            slopes=chunk_slopes,
                                            genotypes_as_idx=chunk_genotypes_as_idx,
                                            method=method)

        out_genotype = idx_to_genotype[result[0] + np.min(_genotype_idx)]
        
        growth_out["genotype"].extend(out_genotype)
        growth_out["mut_effect"].extend(result[1])
        growth_out["mut_effect_std"].extend(result[2])

        out_iptg = result[3]
        out_theta_est = result[4]
        out_theta_std = result[5]
        for j in range(len(out_iptg)):
            
            idx = np.argsort(out_iptg[j])
            
            theta_out["genotype"].extend(np.repeat(out_genotype[j],len(idx)))
            theta_out["iptg"].extend(out_iptg[j][idx])
            theta_out["theta"].extend(out_theta_est[j][idx])
            theta_out["theta_std"].extend(out_theta_std[j][idx])

    print("Creating final dataframes.",flush=True)
    growth_out = pd.DataFrame(growth_out)
    theta_out = pd.DataFrame(theta_out)

    return growth_out, theta_out
