import tfscreen
import jax
from jax import numpy as jnp
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

# Declare float datatype
FLOAT_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

# Set zero conc to this when taking log
ZERO_CONC_VALUE = 1e-20

def _get_posterior_samples(param_posteriors, param_name):
    """
    Get posterior samples for a parameter, handling name fallbacks and HDF5.

    Parameters
    ----------
    param_posteriors : dict
        Dictionary mapping parameter names to posterior samples.
    param_name : str
        Name of the parameter to extract.

    Returns
    -------
    val : numpy.ndarray
        Posterior samples for the requested parameter.

    Raises
    ------
    KeyError
        If the parameter is not found in `param_posteriors`.
    """

    if param_name not in param_posteriors:
        # Try suffixes for MAP/guide keys
        found = False
        for suffix in ["_auto_loc", "_mean"]:
            if f"{param_name}{suffix}" in param_posteriors:
                param_name = f"{param_name}{suffix}"
                found = True
                break

        if not found:
            # Provide more helpful error message if possible
            available_keys = list(param_posteriors.keys())
            if len(available_keys) > 10:
                keys_str = ", ".join(available_keys[:5]) + " ... " + ", ".join(available_keys[-5:])
            else:
                keys_str = ", ".join(available_keys)
            
            error_msg = f"Parameter '{param_name}' not found in posteriors. Available keys: {keys_str}"
            raise KeyError(error_msg)

    val = param_posteriors[param_name]

    return val

def _extract_param_est(input_df,
                       params_to_get,
                       map_column,
                       get_columns,
                       in_run_prefix,
                       param_posteriors,
                       q_to_get):
    """
    Extract parameter estimates and quantiles from posterior samples.

    This function creates a DataFrame for each parameter in `params_to_get`, mapping
    parameter indices to metadata columns, and fills in quantile columns for each
    requested quantile in `q_to_get` using the posterior samples in `param_posteriors`.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing metadata and mapping columns for the parameters.
    params_to_get : list of str
        List of parameter names to extract from the posterior samples.
    map_column : str
        Name of the column in `input_df` that maps rows to parameter indices.
    get_columns : list of str
        List of metadata columns to include in the output DataFrame.
    in_run_prefix : str
        Prefix to prepend to parameter names when looking them up in `param_posteriors`.
    param_posteriors : dict
        Dictionary mapping parameter names (with prefix) to posterior samples,
        where each value is a NumPy array of shape (num_samples, num_params).
    q_to_get : dict
        Dictionary mapping output column names to quantile values (between 0 and 1)
        to extract from the posterior samples.

    Returns
    -------
    out_dfs : dict
        Dictionary mapping each parameter name to a DataFrame containing the
        requested quantiles for each parameter, with metadata columns.

    Raises
    ------
    KeyError
        If a requested parameter or quantile is not found in `param_posteriors`.
    """
    # Create dataframe with unique rows for map_column that has columns
    # get_columns + map_column. Rows will be sorted by map_column.
    get_columns.append(map_column)
    df = (input_df
          .drop_duplicates(map_column)[get_columns]
          .sort_values(map_column)
          .reset_index(drop=True)
          .copy())

    # Go through all parameters requested
    out_dfs = {}
    for param in params_to_get:

        # Grab the posterior distribution of this parameters and flatten.
        model_param = f"{in_run_prefix}{param}"
        val = _get_posterior_samples(param_posteriors, model_param)
        
        # Load HDF5 into memory if needed for reshape
        if hasattr(val, "shape") and not hasattr(val, "reshape"):
            val = val[:]
            
        flat_param = val.reshape(val.shape[0], -1)

        # Create dataframe for loading the data
        to_write = df.copy()

        # Go through quantiles
        for q_name in q_to_get:

            # Calculate quantile and load into the output dataframe
            q = np.quantile(flat_param, q_to_get[q_name], axis=0)
            to_write[q_name] = q[to_write[map_column].values]

        # Record the final dataframe
        out_dfs[param] = to_write.drop(columns=[map_column])

    return out_dfs

def extract_parameters(model, posteriors, q_to_get=None):
    """
    Extract parameter quantiles from posterior samples.

    This method extracts specified quantiles for each model parameter of
    interest, returning a dictionary of DataFrames with parameter estimates
    and associated metadata.

    Parameters
    ----------
    model : ModelClass
        The model instance to extract parameters from.
    posteriors : dict or str
        Assumes this is a dictionary of posteriors keying parameters to 
        numpy arrays, a numpy.lib.npyio.NpzFile object, or a path to a 
        .npz or .h5/.hdf5 file containing posterior samples for model 
        parameters.
    q_to_get : dict, optional
        Dictionary mapping output column names to quantile values (between 0 and 1)
        to extract from the posterior samples. If None, a default set of quantiles
        is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
        upper_quartile, upper_95, max).

    Returns
    -------
    params : dict
        Dictionary mapping parameter names to DataFrames containing the requested
        quantiles and metadata columns for each parameter.

    Raises
    ------
    ValueError
        If `q_to_get` is not a dictionary.
    """

    # Load the posterior file
    if isinstance(posteriors,(dict,np.lib.npyio.NpzFile,h5py.File,h5py.Group)):
        param_posteriors = posteriors
    else:
        if posteriors.endswith(".h5") or posteriors.endswith(".hdf5"):
            param_posteriors = h5py.File(posteriors, 'r')
        else:
            param_posteriors = np.load(posteriors)
    

    # Named quantiles to pull from the posterior distribution
    if q_to_get is None:
        q_to_get = {"min":0.0,
                    "lower_95":0.025,
                    "lower_std":0.159,
                    "lower_quartile":0.25,
                    "median":0.5,
                    "upper_quartile":0.75,
                    "upper_std":0.841,
                    "upper_95":0.975,
                    "max":1.0}
        
    # make sure q_to_get is a dictionary
    if not isinstance(q_to_get,dict):
        raise ValueError(
            "q_to_get should be a dictionary keying column names to quantiles"
        )

    # Define how to go about constructing dataframes to store the parameter
    # estimates. 
    extract = []

    # theta
    if model._theta == "hill":
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["hill_n","log_hill_K","theta_high","theta_low"],
                map_column = "map_theta_group",
                get_columns = ["genotype","titrant_name"],
                in_run_prefix = "theta_"
            )
        )
    elif model._theta == "categorical":
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["theta"],
                map_column = "map_theta",
                get_columns = ["genotype","titrant_name","titrant_conc"],
                in_run_prefix = "theta_"
            )
        )
    
    # condition
    if model._condition_growth in ["linear", "linear_independent", "hierarchical", "independent"]:
        extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition'],
                params_to_get = ["growth_m","growth_k"],
                map_column = "map_condition",
                get_columns = ["replicate","condition"],
                in_run_prefix = "condition_"
            )
        )
    elif model._condition_growth == "power":
         extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition'],
                params_to_get = ["growth_k","growth_m","growth_n"],
                map_column = "map_condition",
                get_columns = ["replicate","condition"],
                in_run_prefix = "condition_"
            )
        )
    elif model._condition_growth == "saturation":
         extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition'],
                params_to_get = ["growth_min","growth_max"],
                map_column = "map_condition",
                get_columns = ["replicate","condition"],
                in_run_prefix = "condition_"
            )
        )

    # growth_transition
    if model._growth_transition == "memory":
        extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition'],
                params_to_get = ["growth_transition_tau0", 
                                 "growth_transition_k1", 
                                 "growth_transition_k2"],
                map_column = "map_condition",
                get_columns = ["replicate","condition"],
                in_run_prefix = ""
            )
        )
    elif model._growth_transition == "baranyi":
         extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition'],
                params_to_get = ["growth_transition_tau_lag", 
                                 "growth_transition_k_sharp"],
                map_column = "map_condition",
                get_columns = ["replicate","condition"],
                in_run_prefix = ""
            )
        )

    # ln_cfu0
    if model._dk_geno == "hierarchical":
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["ln_cfu0"],
                map_column = "map_ln_cfu0",
                get_columns = ["replicate","condition_pre","genotype"],
                in_run_prefix = ""
            )
        )

    # dk_geno
    if model._dk_geno == "none":
        pass
    elif model._dk_geno == "hierarchical":
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["dk_geno"],
                map_column = "map_genotype",
                get_columns = ["genotype"],
                in_run_prefix = ""
            )
        )

    # activity
    if model._activity == "fixed":
        pass
    elif model._activity in ["hierarchical","horseshoe"]:
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["activity"],
                map_column = "map_genotype",
                get_columns = ["genotype"],
                in_run_prefix = ""
            )
        )

    # transformation
    if model._transformation == "congression":
        # lam is global
        lam_df = pd.DataFrame({"parameter":["lam"], "map_all":[0]})
        extract.append(
            dict(
                input_df = lam_df,
                params_to_get = ["lam"],
                map_column = "map_all",
                get_columns = ["parameter"],
                in_run_prefix = "transformation_"
            )
        )

        # mu and sigma are (titrant_name, titrant_conc)
        # We can use the growth_tm.df to find the unique (titrant_name, titrant_conc) pairs 
        # and their indices.
        trans_df = (model.growth_tm.df[["titrant_name", "titrant_conc", 
                                       "titrant_name_idx", "titrant_conc_idx"]]
                    .drop_duplicates()
                    .copy())
        
        # num_titrant_conc
        idx = np.where(np.array(model.growth_tm.tensor_dim_names) == "titrant_conc")[0][0]
        num_titrant_conc = len(model.growth_tm.tensor_dim_labels[idx])
        
        # Map column
        trans_df["map_trans"] = (trans_df["titrant_name_idx"] * num_titrant_conc + 
                                 trans_df["titrant_conc_idx"])
        
        extract.append(
            dict(
                input_df = trans_df,
                params_to_get = ["mu","sigma"],
                map_column = "map_trans",
                get_columns = ["titrant_name","titrant_conc"],
                in_run_prefix = "transformation_"
            )
        )

    params = {}
    for kwargs in extract:
        params.update(_extract_param_est(param_posteriors=param_posteriors,
                                         q_to_get=q_to_get,
                                         **kwargs))

    return params

def extract_theta_curves(model, posteriors, q_to_get=None, manual_titrant_df=None):
    """
    Extract theta curves by sampling from the joint posterior distribution.

    This method calculates fractional occupancy (theta) across a range of
    titrant concentrations by sampling from the joint posterior of Hill
    parameters (hill_n, log_hill_K, theta_high, theta_low).

    Parameters
    ----------
    model : ModelClass
        The model instance to extract parameters from.
    posteriors : dict or str
        Assumes this is a dictionary of posteriors keying parameters to
        numpy arrays or a path to a .npz file containing posterior samples
        for model parameters.
    q_to_get : dict, optional
        Dictionary mapping output column names to quantile values (between 0 and 1)
        to extract from the posterior samples. If None, a default set of quantiles
        is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
        upper_quartile, upper_95, max).
    manual_titrant_df : pd.DataFrame, optional
        A DataFrame specifying 'titrant_name' and 'titrant_conc' values
        at which to calculate theta. If provided, it overrides the default
        calculation at the concentrations present in the input data.
        If 'genotype' is present, it will be used; otherwise, the method
        will calculate theta for all genotypes in the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: 'genotype', 'titrant_name', 'titrant_conc',
        and requested quantiles of theta.

    Raises
    ------
    ValueError
        If the model was not initialized with theta='hill'.
        If `q_to_get` is not a dictionary.
        If `manual_titrant_df` is missing required columns.
    """

    if model._theta != "hill":
        raise ValueError(
            "extract_theta_curves is only available for models where "
            "theta='hill'."
        )

    # Load the posterior file
    if isinstance(posteriors,(dict,np.lib.npyio.NpzFile,h5py.File,h5py.Group)):
        param_posteriors = posteriors
    else:
        if posteriors.endswith(".h5") or posteriors.endswith(".hdf5"):
            param_posteriors = h5py.File(posteriors, 'r')
        else:
            param_posteriors = np.load(posteriors)
    

    # Named quantiles to pull from the posterior distribution
    if q_to_get is None:
        q_to_get = {"min":0.0,
                    "lower_95":0.025,
                    "lower_std":0.159,
                    "lower_quartile":0.25,
                    "median":0.5,
                    "upper_quartile":0.75,
                    "upper_std":0.841,
                    "upper_95":0.975,
                    "max":1.0}

    # make sure q_to_get is a dictionary
    if not isinstance(q_to_get,dict):
        raise ValueError(
            "q_to_get should be a dictionary keying column names to quantiles"
        )

    # Construct calculation DataFrame
    if manual_titrant_df is None:
        # Use unique (genotype, titrant_name, titrant_conc) from input data
        calc_df = (model.growth_tm.df[["genotype", "titrant_name", "titrant_conc", "map_theta_group"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    else:
        tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                              required_columns=["titrant_name", "titrant_conc"])
        
        # If genotype is not provided, broadcast across all genotypes
        if "genotype" not in manual_titrant_df.columns:
            genotypes = model.growth_tm.df["genotype"].unique()
            dfs = []
            for g in genotypes:
                tmp = manual_titrant_df.copy()
                tmp["genotype"] = g
                dfs.append(tmp)
            calc_df = pd.concat(dfs).reset_index(drop=True)
        else:
            calc_df = manual_titrant_df.copy()

        # Map to theta groups
        # We need to reach into the GROWTH_TM to find the mapping
        # This is a bit tricky because the manual_titrant_df might have new concentrations.
        # BUT the parameters (hill_n, etc) are mapped to (genotype, titrant_name).
        # The mapper used in _extract_param_est for Hill model is "map_theta_group"
        # which maps (genotype, titrant_name) to an index.
        
        # Find the (genotype, titrant_name) -> map_theta_group mapping
        mapping = (model.growth_tm.df[["genotype", "titrant_name", "map_theta_group"]]
                   .drop_duplicates()
                   .set_index(["genotype", "titrant_name"])["map_theta_group"]
                   .to_dict())
        
        # Apply mapping. If a (genotype, titrant_name) pair is not in the model, it's an error.
        try:
            calc_df["map_theta_group"] = calc_df.set_index(["genotype", "titrant_name"]).index.map(mapping)
        except Exception as e:
            raise ValueError(
                "Some (genotype, titrant_name) pairs in manual_titrant_df "
                "were not found in the model data."
            ) from e
        
        if calc_df["map_theta_group"].isna().any():
            missing = calc_df[calc_df["map_theta_group"].isna()]
            raise ValueError(
                f"The following (genotype, titrant_name) pairs were not found in the model data: "
                f"{missing[['genotype', 'titrant_name']].drop_duplicates().values}"
            )

    # indices shape: (N_points,)
    indices = calc_df["map_theta_group"].values.astype(int)

    # log_titrant shape: (1, N_points)
    log_titrant = calc_df["titrant_conc"].values.copy()
    log_titrant[log_titrant == 0] = ZERO_CONC_VALUE
    log_titrant = np.log(log_titrant)[None, :]

    # Extract posterior parameters and flatten (num_samples, num_groups)
    hill_n = _get_posterior_samples(param_posteriors, "theta_hill_n")
    if hasattr(hill_n, "shape") and not hasattr(hill_n, "reshape"):
        hill_n = hill_n[:]
    hill_n = hill_n.reshape(hill_n.shape[0], -1)

    log_hill_K = _get_posterior_samples(param_posteriors, "theta_log_hill_K")
    if hasattr(log_hill_K, "shape") and not hasattr(log_hill_K, "reshape"):
        log_hill_K = log_hill_K[:]
    log_hill_K = log_hill_K.reshape(log_hill_K.shape[0], -1)

    theta_high = _get_posterior_samples(param_posteriors, "theta_theta_high")
    if hasattr(theta_high, "shape") and not hasattr(theta_high, "reshape"):
        theta_high = theta_high[:]
    theta_high = theta_high.reshape(theta_high.shape[0], -1)

    theta_low = _get_posterior_samples(param_posteriors, "theta_theta_low")
    if hasattr(theta_low, "shape") and not hasattr(theta_low, "reshape"):
        theta_low = theta_low[:]
    theta_low = theta_low.reshape(theta_low.shape[0], -1)
    
    # Indexed params shape: (N_samples, N_points)
    h_n = hill_n[:, indices]
    l_K = log_hill_K[:, indices]
    t_h = theta_high[:, indices]
    t_l = theta_low[:, indices]
    
    # Calculate theta using Hill equation: (N_samples, N_points)
    # occupancy = 1 / (1 + exp(-hill_n * (log(conc) - log_K)))
    occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_titrant - l_K)))
    theta_samples = t_l + (t_h - t_l) * occupancy
    
    # Calculate quantiles across samples (axis 0)
    for q_name, q_val in q_to_get.items():
        calc_df[q_name] = np.quantile(theta_samples, q_val, axis=0)

    return calc_df.drop(columns=["map_theta_group"])

def extract_growth_predictions(model,
                               posteriors,
                               q_to_get=None,
                               row_chunk_size=100,
                               max_block_elements=1_000_000_000):
    """
    Extract predicted ln_cfu values matching the input growth data.

    This method pulls the 'growth_pred' values from the posterior samples
    and maps them back to the original rows in `model.growth_df`.

    Parameters
    ----------
    model : ModelClass
        The model instance to extract parameters from.
    posteriors : dict or str
        Assumes this is a dictionary of posteriors keying parameters to
        numpy arrays or a path to a .npz file containing posterior samples
        for model parameters.
    q_to_get : dict, optional
        Dictionary mapping output column names to quantile values (between 0 and 1)
        to extract from the posterior samples. If None, a default set of quantiles
        is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
        upper_quartile, upper_95, max).
    row_chunk_size : int, optional
        Number of rows to process at a time. Defaults to 100.
    max_block_elements : int, optional
        Maximum number of elements to read in a single HDF5 block. Defaults to 1,000,000,000.

    Returns
    -------
    pd.DataFrame
        A copy of `model.growth_df` with new columns for the requested
        quantiles of 'ln_cfu_pred'.

    Raises
    ------
    ValueError
        If 'growth_pred' is not in the posterior samples.
        If `q_to_get` is not a dictionary.
    """

    # Load the posterior file
    if isinstance(posteriors,(dict,np.lib.npyio.NpzFile,h5py.File,h5py.Group)):
        param_posteriors = posteriors
    else:
        if posteriors.endswith(".h5") or posteriors.endswith(".hdf5"):
            param_posteriors = h5py.File(posteriors, 'r')
        else:
            param_posteriors = np.load(posteriors)

    if "growth_pred" not in param_posteriors:
        raise ValueError(
            "'growth_pred' not found in posterior samples. Make sure the "
            "model was run in a way that generates growth predictions."
        )

    # Named quantiles to pull from the posterior distribution
    if q_to_get is None:
        q_to_get = {"min":0.0,
                    "lower_95":0.025,
                    "lower_std":0.159,
                    "lower_quartile":0.25,
                    "median":0.5,
                    "upper_quartile":0.75,
                    "upper_std":0.841,
                    "upper_95":0.975,
                    "max":1.0}

    # make sure q_to_get is a dictionary
    if not isinstance(q_to_get,dict):
        raise ValueError(
            "q_to_get should be a dictionary keying column names to quantiles"
        )

    # Grab the growth_pred tensor
    growth_pred = _get_posterior_samples(param_posteriors, "growth_pred")

    # The tensor shape is (num_samples, replicate, time, condition_pre, 
    # condition_sel, titrant_name, titrant_conc, genotype)
    
    # Sort the dataframe by index columns to improve HDF5 access locality
    index_cols = ["replicate_idx", "time_idx", "condition_pre_idx", 
                  "condition_sel_idx", "titrant_name_idx", 
                  "titrant_conc_idx", "genotype_idx"]
    
    out_df = model.growth_df.copy()
    out_df = out_df.sort_values(by=index_cols)
    
    # Get the sorted index columns
    rep_idx = out_df["replicate_idx"].values
    time_idx = out_df["time_idx"].values
    pre_idx = out_df["condition_pre_idx"].values
    sel_idx = out_df["condition_sel_idx"].values
    name_idx = out_df["titrant_name_idx"].values
    conc_idx = out_df["titrant_conc_idx"].values
    geno_idx = out_df["genotype_idx"].values

    # Create a clean dataframe for output
    keep_columns = ["replicate", "genotype",
                    "condition_pre", "condition_sel", 
                    "titrant_name", "titrant_conc",
                    "t_pre", "t_sel",
                    "ln_cfu","ln_cfu_std"]

    out_df = out_df[keep_columns].reset_index(drop=True)

    total_rows = len(out_df)

    # Initialize quantile columns
    for q_name in q_to_get:
        out_df[q_name] = np.nan

    # Sort quantiles to ensure predictable behavior
    q_names = list(q_to_get.keys())
    q_values = np.array([q_to_get[name] for name in q_names])
    
    is_h5 = isinstance(growth_pred, h5py.Dataset)
    num_samples = growth_pred.shape[0]

    # Grab chunks of rows to avoid OOM
    for start_r in tqdm(range(0, total_rows, row_chunk_size)):

        end_r = min(start_r + row_chunk_size, total_rows)
        
        # Slices for this chunk
        r_slice = rep_idx[start_r:end_r]
        t_slice = time_idx[start_r:end_r]
        p_slice = pre_idx[start_r:end_r]
        s_slice = sel_idx[start_r:end_r]
        n_slice = name_idx[start_r:end_r]
        c_slice = conc_idx[start_r:end_r]
        g_slice = geno_idx[start_r:end_r]

        if is_h5:
            # Calculate bounding box for this chunk
            rmin, rmax = r_slice.min(), r_slice.max()
            tmin, tmax = t_slice.min(), t_slice.max()
            pmin, pmax = p_slice.min(), p_slice.max()
            smin, smax = s_slice.min(), s_slice.max()
            nmin, nmax = n_slice.min(), n_slice.max()
            cmin, cmax = c_slice.min(), c_slice.max()
            gmin, gmax = g_slice.min(), g_slice.max()

            # Calculate volume of bounding box (excluding num_samples)
            # Cast to Python int to avoid numpy fixed-width overflow warnings
            spatial_volume = (
                int(rmax - rmin + 1) * int(tmax - tmin + 1) * 
                int(pmax - pmin + 1) * int(smax - smin + 1) * 
                int(nmax - nmin + 1) * int(cmax - cmin + 1) * 
                int(gmax - gmin + 1)
            )
            
            if (spatial_volume * int(num_samples)) <= max_block_elements:
                # Read the entire block at once
                block = growth_pred[:, 
                                    rmin:rmax+1, tmin:tmax+1, 
                                    pmin:pmax+1, smin:smax+1, 
                                    nmin:nmax+1, cmin:cmax+1, 
                                    gmin:gmax+1]
                
                # Index into the block using relative indices
                preds_chunk = block[:, 
                                    r_slice - rmin, t_slice - tmin, 
                                    p_slice - pmin, s_slice - smin, 
                                    n_slice - nmin, c_slice - cmin, 
                                    g_slice - gmin]
            else:
                # Fallback to row-by-row if the block is too sparse/large
                preds_chunk_list = []
                for idx in range(len(r_slice)):
                    row_data = growth_pred[:, r_slice[idx], t_slice[idx], p_slice[idx], 
                                           s_slice[idx], n_slice[idx], c_slice[idx], g_slice[idx]]
                    preds_chunk_list.append(row_data)
                preds_chunk = np.stack(preds_chunk_list, axis=1)
        else:
            # Standard numpy indexing for non-HDF5
            preds_chunk = growth_pred[:, r_slice, t_slice, p_slice, s_slice, n_slice, c_slice, g_slice]

        # Calculate all quantiles in one go: (len(q_values), chunk_size)
        all_quantiles = np.quantile(preds_chunk, q_values, axis=0)

        # Assign to out_df
        for i, q_name in enumerate(q_names):
            out_df.loc[out_df.index[start_r:end_r], q_name] = all_quantiles[i]

    return out_df
