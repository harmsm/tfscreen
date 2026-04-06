import tfscreen
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from tfscreen.analysis.hierarchical.posteriors import load_posteriors, get_posterior_samples

# Set zero conc to this when taking log
ZERO_CONC_VALUE = 1e-20


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
        val = get_posterior_samples(param_posteriors, model_param)
        
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

    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

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
    elif model._theta == "hill_mut":
        # Assembled per-genotype parameters shaped (T, G);
        # flat index = titrant_name_idx * num_genotype + genotype_idx
        _geno_dim = model.growth_tm.tensor_dim_names.index("genotype")
        _num_genotype = len(model.growth_tm.tensor_dim_labels[_geno_dim])
        theta_mut_df = (model.growth_tm.df[["genotype", "titrant_name",
                                            "genotype_idx", "titrant_name_idx"]]
                        .drop_duplicates()
                        .copy())
        theta_mut_df["map_theta_mut"] = (theta_mut_df["titrant_name_idx"] * _num_genotype
                                         + theta_mut_df["genotype_idx"])
        extract.append(
            dict(
                input_df = theta_mut_df,
                params_to_get = ["theta_low","theta_high","log_hill_K","hill_n"],
                map_column = "map_theta_mut",
                get_columns = ["genotype","titrant_name"],
                in_run_prefix = "theta_"
            )
        )
        # Per-mutation deltas shaped (T, M);
        # flat index = titrant_name_idx * num_mutation + mutation_idx
        _titrant_dim = model.growth_tm.tensor_dim_names.index("titrant_name")
        _titrant_names = list(model.growth_tm.tensor_dim_labels[_titrant_dim])
        _num_mut = len(model.mut_labels)
        _theta_d_rows = [
            {"titrant_name": t, "mutation": m,
             "map_theta_d_mut": ti * _num_mut + mi}
            for ti, t in enumerate(_titrant_names)
            for mi, m in enumerate(model.mut_labels)
        ]
        extract.append(
            dict(
                input_df = pd.DataFrame(_theta_d_rows),
                params_to_get = ["d_logit_low","d_logit_delta","d_log_hill_K","d_log_hill_n"],
                map_column = "map_theta_d_mut",
                get_columns = ["titrant_name","mutation"],
                in_run_prefix = "theta_"
            )
        )
        if model.pair_labels:
            _num_pair = len(model.pair_labels)
            _theta_epi_rows = [
                {"titrant_name": t, "pair": p,
                 "map_theta_epi": ti * _num_pair + pi}
                for ti, t in enumerate(_titrant_names)
                for pi, p in enumerate(model.pair_labels)
            ]
            extract.append(
                dict(
                    input_df = pd.DataFrame(_theta_epi_rows),
                    params_to_get = ["epi_logit_low","epi_logit_delta",
                                     "epi_log_hill_K","epi_log_hill_n"],
                    map_column = "map_theta_epi",
                    get_columns = ["titrant_name","pair"],
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
                input_df = model.growth_tm.map_groups['condition_rep'],
                params_to_get = ["growth_m","growth_k"],
                map_column = "map_condition_rep",
                get_columns = ["replicate","condition_rep"],
                in_run_prefix = "condition_"
            )
        )
    elif model._condition_growth == "power":
         extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition_rep'],
                params_to_get = ["growth_k","growth_m","growth_n"],
                map_column = "map_condition_rep",
                get_columns = ["replicate","condition_rep"],
                in_run_prefix = "condition_"
            )
        )
    elif model._condition_growth == "saturation":
         extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition_rep'],
                params_to_get = ["growth_min","growth_max"],
                map_column = "map_condition_rep",
                get_columns = ["replicate","condition_rep"],
                in_run_prefix = "condition_"
            )
        )

    # growth_transition
    if model._growth_transition == "memory":
        extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition_rep'],
                params_to_get = ["growth_transition_tau0", 
                                 "growth_transition_k1", 
                                 "growth_transition_k2"],
                map_column = "map_condition_rep",
                get_columns = ["replicate","condition_rep"],
                in_run_prefix = ""
            )
        )
    elif model._growth_transition == "baranyi":
         extract.append(
            dict(
                input_df = model.growth_tm.map_groups['condition_rep'],
                params_to_get = ["growth_transition_tau_lag", 
                                 "growth_transition_k_sharp"],
                map_column = "map_condition_rep",
                get_columns = ["replicate","condition_rep"],
                in_run_prefix = ""
            )
        )

    # ln_cfu0
    if model._dk_geno in ["hierarchical", "hierarchical_mut"]:
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
    elif model._dk_geno in ["hierarchical", "hierarchical_mut"]:
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["dk_geno"],
                map_column = "map_genotype",
                get_columns = ["genotype"],
                in_run_prefix = ""
            )
        )
        if model._dk_geno == "hierarchical_mut":
            _mut_df = pd.DataFrame({
                "mutation": model.mut_labels,
                "map_mutation": range(len(model.mut_labels)),
            })
            extract.append(
                dict(
                    input_df = _mut_df,
                    params_to_get = ["d_dk_geno"],
                    map_column = "map_mutation",
                    get_columns = ["mutation"],
                    in_run_prefix = "dk_geno_"
                )
            )
            if model.pair_labels:
                _pair_df = pd.DataFrame({
                    "pair": model.pair_labels,
                    "map_pair": range(len(model.pair_labels)),
                })
                extract.append(
                    dict(
                        input_df = _pair_df,
                        params_to_get = ["epi_dk_geno"],
                        map_column = "map_pair",
                        get_columns = ["pair"],
                        in_run_prefix = "dk_geno_"
                    )
                )

    # activity
    if model._activity == "fixed":
        pass
    elif model._activity in ["hierarchical", "horseshoe", "hierarchical_mut"]:
        extract.append(
            dict(
                input_df = model.growth_tm.df,
                params_to_get = ["activity"],
                map_column = "map_genotype",
                get_columns = ["genotype"],
                in_run_prefix = ""
            )
        )
        if model._activity == "hierarchical_mut":
            _mut_df = pd.DataFrame({
                "mutation": model.mut_labels,
                "map_mutation": range(len(model.mut_labels)),
            })
            extract.append(
                dict(
                    input_df = _mut_df,
                    params_to_get = ["d_log_activity"],
                    map_column = "map_mutation",
                    get_columns = ["mutation"],
                    in_run_prefix = "activity_"
                )
            )
            if model.pair_labels:
                _pair_df = pd.DataFrame({
                    "pair": model.pair_labels,
                    "map_pair": range(len(model.pair_labels)),
                })
                extract.append(
                    dict(
                        input_df = _pair_df,
                        params_to_get = ["epi_log_activity"],
                        map_column = "map_pair",
                        get_columns = ["pair"],
                        in_run_prefix = "activity_"
                    )
                )

    # transformation
    if model._transformation in ["logit_norm", "empirical"]:
        
        # lam is global for both models
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

        # mu and sigma are only for logit_norm
        if model._transformation == "logit_norm":

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

def _build_manual_calc_df_hill(model, manual_titrant_df):
    """
    Expand *manual_titrant_df* to include all genotypes (if 'genotype' is absent)
    and attach the ``map_theta_group`` index used by the ``hill`` model.
    """
    tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                          required_columns=["titrant_name", "titrant_conc"])
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

    mapping = (model.growth_tm.df[["genotype", "titrant_name", "map_theta_group"]]
               .drop_duplicates()
               .set_index(["genotype", "titrant_name"])["map_theta_group"]
               .to_dict())

    try:
        calc_df["map_theta_group"] = (calc_df
                                      .set_index(["genotype", "titrant_name"])
                                      .index.map(mapping))
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
    return calc_df


def _build_manual_calc_df_hill_mut(model, manual_titrant_df):
    """
    Expand *manual_titrant_df* to include all genotypes (if 'genotype' is absent)
    and attach ``genotype_idx`` / ``titrant_name_idx`` used by the ``hill_mut`` model.
    """
    tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                          required_columns=["titrant_name", "titrant_conc"])
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

    geno_map = (model.growth_tm.df[["genotype", "genotype_idx"]]
                .drop_duplicates()
                .set_index("genotype")["genotype_idx"]
                .to_dict())
    titrant_map = (model.growth_tm.df[["titrant_name", "titrant_name_idx"]]
                   .drop_duplicates()
                   .set_index("titrant_name")["titrant_name_idx"]
                   .to_dict())

    calc_df["genotype_idx"] = calc_df["genotype"].map(geno_map)
    calc_df["titrant_name_idx"] = calc_df["titrant_name"].map(titrant_map)

    missing_geno = calc_df["genotype_idx"].isna()
    missing_titrant = calc_df["titrant_name_idx"].isna()
    if missing_geno.any() or missing_titrant.any():
        bad = calc_df[missing_geno | missing_titrant][["genotype", "titrant_name"]].drop_duplicates()
        raise ValueError(
            f"The following (genotype, titrant_name) pairs were not found in the model data: "
            f"{bad.values}"
        )
    return calc_df


def _extract_theta_curves_hill(model, posteriors, q_to_get, manual_titrant_df):
    """
    Compute theta curves for the ``hill`` model.

    Posterior parameters are indexed per ``map_theta_group`` (one group per
    (genotype, titrant_name) pair).
    """
    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    if manual_titrant_df is None:
        calc_df = (model.growth_tm.df[["genotype", "titrant_name", "titrant_conc",
                                       "map_theta_group"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    else:
        calc_df = _build_manual_calc_df_hill(model, manual_titrant_df)

    indices = calc_df["map_theta_group"].values.astype(int)

    log_titrant = calc_df["titrant_conc"].values.copy()
    log_titrant[log_titrant == 0] = ZERO_CONC_VALUE
    log_titrant = np.log(log_titrant)[None, :]   # (1, N_points)

    def _load_flat(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return v.reshape(v.shape[0], -1)          # (S, num_groups)

    hill_n    = _load_flat("theta_hill_n")
    log_hill_K = _load_flat("theta_log_hill_K")
    theta_high = _load_flat("theta_theta_high")
    theta_low  = _load_flat("theta_theta_low")

    h_n = hill_n[:, indices]
    l_K = log_hill_K[:, indices]
    t_h = theta_high[:, indices]
    t_l = theta_low[:, indices]

    occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_titrant - l_K)))
    theta_samples = t_l + (t_h - t_l) * occupancy

    for q_name, q_val in q_to_get.items():
        calc_df[q_name] = np.quantile(theta_samples, q_val, axis=0)

    return calc_df.drop(columns=["map_theta_group"])


def _extract_theta_curves_hill_mut(model, posteriors, q_to_get, manual_titrant_df):
    """
    Compute theta curves for the ``hill_mut`` model.

    Posterior parameters are shaped ``(S, T, G)`` — indexed by
    ``titrant_name_idx`` and ``genotype_idx`` independently.
    """
    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    if manual_titrant_df is None:
        calc_df = (model.growth_tm.df[["genotype", "titrant_name", "titrant_conc",
                                       "genotype_idx", "titrant_name_idx"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    else:
        calc_df = _build_manual_calc_df_hill_mut(model, manual_titrant_df)

    geno_indices    = calc_df["genotype_idx"].values.astype(int)
    titrant_indices = calc_df["titrant_name_idx"].values.astype(int)

    log_titrant = calc_df["titrant_conc"].values.copy()
    log_titrant[log_titrant == 0] = ZERO_CONC_VALUE
    log_titrant = np.log(log_titrant)   # (N_points,)

    def _load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return v   # (S, T, G)

    hill_n    = _load("theta_hill_n")
    log_hill_K = _load("theta_log_hill_K")
    theta_high = _load("theta_theta_high")
    theta_low  = _load("theta_theta_low")

    # Index per row: (S, N_points)
    h_n = hill_n[:, titrant_indices, geno_indices]
    l_K = log_hill_K[:, titrant_indices, geno_indices]
    t_h = theta_high[:, titrant_indices, geno_indices]
    t_l = theta_low[:, titrant_indices, geno_indices]

    occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_titrant[None, :] - l_K)))
    theta_samples = t_l + (t_h - t_l) * occupancy

    for q_name, q_val in q_to_get.items():
        calc_df[q_name] = np.quantile(theta_samples, q_val, axis=0)

    return calc_df.drop(columns=["genotype_idx", "titrant_name_idx"])


def extract_theta_curves(model, posteriors, q_to_get=None, manual_titrant_df=None):
    """
    Extract theta curves by sampling from the joint posterior distribution.

    Calculates fractional occupancy (theta) across a range of titrant
    concentrations from the posterior of Hill parameters.  Supports both the
    ``hill`` model (one set of parameters per (genotype, titrant_name) group)
    and the ``hill_mut`` model (per-genotype, per-titrant parameters assembled
    from WT values and per-mutation deltas).

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
        If the model was not initialized with theta='hill' or theta='hill_mut'.
        If `q_to_get` is not a dictionary.
        If `manual_titrant_df` is missing required columns.
    """
    if model._theta == "hill":
        return _extract_theta_curves_hill(model, posteriors, q_to_get, manual_titrant_df)
    elif model._theta == "hill_mut":
        return _extract_theta_curves_hill_mut(model, posteriors, q_to_get, manual_titrant_df)
    else:
        raise ValueError(
            "extract_theta_curves is only available for models where "
            "theta='hill' or theta='hill_mut'."
        )

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

    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    if "growth_pred" not in param_posteriors:
        raise ValueError(
            "'growth_pred' not found in posterior samples. Make sure the "
            "model was run in a way that generates growth predictions."
        )

    # Grab the growth_pred tensor
    growth_pred = get_posterior_samples(param_posteriors, "growth_pred")

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
