import tfscreen
import pandas as pd
import numpy as np
import h5py
from dataclasses import dataclass
from typing import List, Any
from tqdm import tqdm
from tfscreen.analysis.hierarchical.posteriors import load_posteriors, get_posterior_samples
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry


@dataclass
class ExtractionContext:
    """Narrow interface passed to each component's get_extract_specs."""
    growth_tm: Any
    mut_labels: List[str]
    pair_labels: List[str]
    growth_shares_replicates: bool


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
    all_columns = list(get_columns) + [map_column]
    df = (input_df
          .drop_duplicates(map_column)[all_columns]
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

    ctx = ExtractionContext(
        growth_tm=model.growth_tm,
        mut_labels=model.mut_labels,
        pair_labels=model.pair_labels,
        growth_shares_replicates=model._growth_shares_replicates,
    )

    component_selections = [
        ("condition_growth", model._condition_growth),
        ("growth_transition", model._growth_transition),
        ("ln_cfu0", "hierarchical"),
        ("dk_geno", model._dk_geno),
        ("activity", model._activity),
        ("theta", model._theta),
        ("transformation", model._transformation),
    ]

    extract = []
    for category, selection in component_selections:
        module = model_registry.get(category, {}).get(selection)
        if module is not None and hasattr(module, "get_extract_specs"):
            extract.extend(module.get_extract_specs(ctx))

    params = {}
    for kwargs in extract:
        params.update(_extract_param_est(param_posteriors=param_posteriors,
                                         q_to_get=q_to_get,
                                         **kwargs))

    return params

def extract_theta_curves(model, posteriors, q_to_get=None, manual_titrant_df=None,
                         num_samples=100):
    """
    Extract theta curves by sampling from the joint posterior distribution.

    Dispatches to the active theta component's ``build_calc_df`` /
    ``compute_theta_samples`` interface, then applies shared scaffolding
    (quantile extraction, optional raw-sample columns).

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
    num_samples : int or None, optional
        Randomly select this many joint posterior samples and return them as
        columns ``sample_0``, ``sample_1``, ... alongside the quantile
        columns. Sampling is with replacement when ``num_samples`` exceeds the
        total number of posterior samples. Set to ``None`` to suppress sample
        columns and return only quantiles. Default 100.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: 'genotype', 'titrant_name', 'titrant_conc',
        requested quantiles of theta, and (unless ``num_samples`` is ``None``)
        ``sample_0`` … ``sample_{num_samples-1}``.

    Raises
    ------
    ValueError
        If the active theta component does not implement the extraction interface.
        If `q_to_get` is not a dictionary.
        If `manual_titrant_df` is missing required columns.
    """
    module = model_registry.get("theta", {}).get(model._theta)
    if module is None or not (hasattr(module, "build_calc_df")
                              and hasattr(module, "compute_theta_samples")):
        raise ValueError(
            f"extract_theta_curves requires the theta component to implement "
            f"build_calc_df and compute_theta_samples. "
            f"'{model._theta}' does not support this interface."
        )

    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    calc_df, internal_cols, extra_kwargs = module.build_calc_df(model, manual_titrant_df)
    theta_samples = module.compute_theta_samples(calc_df, param_posteriors, **extra_kwargs)

    for q_name, q_val in q_to_get.items():
        calc_df[q_name] = np.quantile(theta_samples, q_val, axis=0)

    if num_samples is not None:
        S = theta_samples.shape[0]
        chosen = np.random.choice(S, size=num_samples, replace=num_samples > S)
        samples_df = pd.DataFrame(
            theta_samples[chosen].T,
            columns=[f"sample_{i}" for i in range(num_samples)],
            index=calc_df.index,
        )
        calc_df = pd.concat([calc_df, samples_df], axis=1)

    return calc_df.drop(columns=internal_cols)

def extract_growth_predictions(model,
                               posteriors,
                               q_to_get=None,
                               num_samples=100):
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
    num_samples : int or None, optional
        Randomly select this many joint posterior samples and return them as
        columns ``sample_0``, ``sample_1``, ... alongside the quantile
        columns. The same sample indices are used for every row so that
        trajectories are jointly consistent. Sampling is with replacement when
        ``num_samples`` exceeds the total number of posterior samples. Set to
        ``None`` to suppress sample columns and return only quantiles.
        Default 100.

    Returns
    -------
    pd.DataFrame
        A copy of `model.growth_df` with new columns for the requested
        quantiles of 'ln_cfu_pred' and (unless ``num_samples`` is ``None``)
        ``sample_0`` … ``sample_{num_samples-1}``.

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

    group_cols = ["replicate_idx", "time_idx", "condition_pre_idx",
                  "condition_sel_idx", "titrant_name_idx", "titrant_conc_idx"]

    out_df = model.growth_df.copy()
    out_df = out_df.sort_values(by=group_cols + ["genotype_idx"])

    keep_columns = ["replicate", "genotype",
                    "condition_pre", "condition_sel",
                    "titrant_name", "titrant_conc",
                    "t_pre", "t_sel",
                    "ln_cfu", "ln_cfu_std"]

    # Carry index columns through reset so groupby positions match out_df rows
    out_df = out_df[keep_columns + group_cols + ["genotype_idx"]].reset_index(drop=True)

    total_rows = len(out_df)

    q_names = list(q_to_get.keys())
    q_values = np.array([q_to_get[name] for name in q_names])
    for q_name in q_names:
        out_df[q_name] = np.nan

    is_h5 = isinstance(growth_pred, h5py.Dataset)
    num_posterior_samples = growth_pred.shape[0]

    # Pre-select joint sample indices so all rows share the same trajectories
    if num_samples is not None:
        chosen_sample_idxs = np.random.choice(num_posterior_samples,
                                              size=num_samples,
                                              replace=num_samples > num_posterior_samples)
        sample_arr = np.empty((total_rows, num_samples))

    if is_h5:
        # For HDF5: iterate by (rep, time, condition) group and read one full
        # genotype slice per group. This avoids bounding-box explosion when
        # consecutive chunks straddle group boundaries (which makes gmin=0,
        # gmax=num_genotypes-1 for every boundary chunk).
        num_groups = out_df[group_cols].drop_duplicates().shape[0]
        for keys, grp in tqdm(out_df.groupby(group_cols, sort=False),
                               total=num_groups):
            r, t, p, s, n, c = (int(k) for k in keys)
            g_idx = grp["genotype_idx"].values
            row_pos = grp.index.values

            # Single contiguous read: all posterior samples × all genotypes
            # for this fixed (rep, time, condition) combination.
            geno_slice = growth_pred[:, r, t, p, s, n, c, :]  # (S, G)
            preds = geno_slice[:, g_idx]                        # (S, len(grp))

            all_quantiles = np.quantile(preds, q_values, axis=0)
            for i, q_name in enumerate(q_names):
                out_df.loc[row_pos, q_name] = all_quantiles[i]

            if num_samples is not None:
                sample_arr[row_pos] = preds[chosen_sample_idxs].T
    else:
        # numpy: the full tensor is in memory; one vectorized fancy-index
        # across all rows is fastest.
        rep_idx  = out_df["replicate_idx"].values
        time_idx = out_df["time_idx"].values
        pre_idx  = out_df["condition_pre_idx"].values
        sel_idx  = out_df["condition_sel_idx"].values
        name_idx = out_df["titrant_name_idx"].values
        conc_idx = out_df["titrant_conc_idx"].values
        geno_idx = out_df["genotype_idx"].values

        preds_all = growth_pred[:, rep_idx, time_idx, pre_idx,
                                sel_idx, name_idx, conc_idx, geno_idx]

        all_quantiles = np.quantile(preds_all, q_values, axis=0)
        for i, q_name in enumerate(q_names):
            out_df[q_name] = all_quantiles[i]

        if num_samples is not None:
            sample_arr = preds_all[chosen_sample_idxs].T

    if num_samples is not None:
        samples_df = pd.DataFrame(
            sample_arr,
            columns=[f"sample_{i}" for i in range(num_samples)],
            index=out_df.index,
        )
        out_df = pd.concat([out_df, samples_df], axis=1)

    return out_df.drop(columns=group_cols + ["genotype_idx"])
