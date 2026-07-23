import warnings
import pandas as pd
import numpy as np
import h5py
from dataclasses import dataclass
from typing import List, Any
from tqdm import tqdm
from tfscreen.tfmodel.inference.posteriors import load_posteriors, get_posterior_samples
from tfscreen.tfmodel.generative.registry import model_registry


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

def extract_parameters(orchestrator, posteriors, q_to_get=None):
    """
    Extract parameter quantiles from posterior samples.

    This method extracts specified quantiles for each model parameter of
    interest, returning a dictionary of DataFrames with parameter estimates
    and associated metadata.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
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
        growth_tm=orchestrator.training_tm,
        mut_labels=orchestrator.mut_labels,
        pair_labels=orchestrator.pair_labels,
        growth_shares_replicates=orchestrator._growth_shares_replicates,
    )

    component_selections = [
        ("condition_growth", orchestrator._condition_growth),
        ("growth_transition", orchestrator._growth_transition),
        ("ln_cfu0", "hierarchical"),
        ("dk_geno", orchestrator._dk_geno),
        ("activity", orchestrator._activity),
        ("theta", orchestrator._theta),
        ("transformation", orchestrator._transformation),
    ]

    extract = []
    for category, selection in component_selections:
        module = model_registry.get(category, {}).get(selection)
        if module is not None and hasattr(module, "get_extract_specs"):
            extract.extend(module.get_extract_specs(ctx))

    # base_growth is not a swappable registry component (see
    # model_orchestrator._read_base_growth_df / generative/model.py's
    # base_growth_obs block), so its single global k_ref scalar is extracted
    # here directly rather than via a component's get_extract_specs.
    if getattr(orchestrator, "_base_growth_df", None) is not None:
        k_ref_df = pd.DataFrame({"parameter": ["k_ref"], "map_all": [0]})
        extract.append(dict(
            input_df=k_ref_df,
            params_to_get=["k_ref"],
            map_column="map_all",
            get_columns=["parameter"],
            in_run_prefix="base_growth_",
        ))

    params = {}
    for kwargs in extract:
        try:
            params.update(_extract_param_est(param_posteriors=param_posteriors,
                                             q_to_get=q_to_get,
                                             **kwargs))
        except KeyError as exc:
            param_names = kwargs.get("params_to_get", [])
            warnings.warn(
                f"Skipping extraction of {param_names}: {exc}. "
                "This can happen with MAP checkpoints because computed "
                "(deterministic) sites like 'ln_cfu0' are absent. "
                "Run tfs-sample-posterior first to get a full posterior file, "
                "or use tfs-param-quantiles on the posterior .h5 file.",
                stacklevel=2,
            )

    return params

def extract_theta_curves(orchestrator, posteriors, q_to_get=None, manual_titrant_df=None,
                         num_samples=100):
    """
    Extract theta curves by sampling from the joint posterior distribution.

    Dispatches to the active theta component's ``build_calc_df`` /
    ``compute_theta_samples`` interface, then applies shared scaffolding
    (quantile extraction, optional raw-sample columns).

    Parameters
    ----------
    orchestrator : ModelOrchestrator
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
        will calculate theta for all genotypes in the orchestrator.
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
    module = model_registry.get("theta", {}).get(orchestrator._theta)
    if module is None or not (hasattr(module, "build_calc_df")
                              and hasattr(module, "compute_theta_samples")):
        raise ValueError(
            f"extract_theta_curves requires the theta component to implement "
            f"build_calc_df and compute_theta_samples. "
            f"'{orchestrator._theta}' does not support this interface."
        )

    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    calc_df, internal_cols, extra_kwargs = module.build_calc_df(orchestrator, manual_titrant_df)
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

def extract_theta_epistasis(orchestrator, posteriors, q_to_get=None,
                            manual_titrant_df=None, scale="logit",
                            scale_constant=1.0,
                            group_by=("titrant_name", "titrant_conc"),
                            regime_eps=0.01, regime_ci=0.95):
    """
    Extract second-order epistasis quantiles from the joint theta posterior.

    Unlike calculating epistasis from per-genotype marginal theta estimates
    (which treats the four corners of each mutant cycle as independent and
    propagates ``sqrt(sum std**2)``), this function draws theta for every
    genotype from the *same* posterior sample, computes epistasis within each
    draw, and then quantiles across draws.  The resulting uncertainty therefore
    reflects the true posterior covariance between the wildtype, single, and
    double mutants of each cycle.

    Only genotypes seen during training are supported: the joint sample matrix
    comes from the theta component's ``build_calc_df`` / ``compute_theta_samples``
    interface (the same one used by ``extract_theta_curves``), which returns
    training genotypes.  Out-of-training genotypes have no joint sample matrix
    and are not handled here.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        The fitted model instance.
    posteriors : dict or str
        Posterior samples (dict, NpzFile, or path to a ``.npz``/``.h5`` file).
        A MAP checkpoint provides a single "draw"; the returned quantile columns
        then all collapse to that point estimate (no uncertainty).
    q_to_get : dict or array-like, optional
        Quantile levels to extract.  Defaults to the standard dense set used by
        the other extraction functions.
    manual_titrant_df : pd.DataFrame, optional
        A DataFrame specifying ``'titrant_name'`` and ``'titrant_conc'`` values
        at which to calculate theta before building cycles.  If it also has a
        ``'genotype'`` column it selects the genotypes; otherwise all training
        genotypes are used.
    scale : {"logit", "add", "mult"}, default "logit"
        The epistatic scale.  ``"logit"`` is the natural choice for theta (an
        occupancy in ``[0, 1]``); see
        ``tfscreen.analysis.extract_epistasis`` for the definitions.
    scale_constant : float, default 1.0
        Constant applied to the transform before the difference-of-differences
        (e.g. ``-RT`` to report logit epistasis as an interaction free energy).
        Rejected for ``scale="mult"`` (it cancels in the ratio-of-ratios).
    group_by : sequence of str, default ("titrant_name", "titrant_conc")
        Columns defining a unique condition; epistasis is computed independently
        within each.  For theta, each concentration is its own condition.
    regime_eps : float, default 0.01
        Theta resolution floor for the ``in_regime`` flag.  A cycle corner is
        "in the resolvable band" when its theta posterior sits inside
        ``[regime_eps, 1 - regime_eps]``; outside that band logit(theta) is
        saturated and its uncertainty is dominated by the theta-model
        extrapolation rather than the data.  Must satisfy ``0 <= regime_eps <
        0.5``.
    regime_ci : float, default 0.95
        Central posterior-mass fraction that must lie inside the band for a
        corner to count as in-regime (e.g. 0.95 requires the theta 2.5-97.5%
        interval within ``[regime_eps, 1 - regime_eps]``).  Must be in (0, 1).

    Returns
    -------
    pd.DataFrame
        One row per (double-mutant genotype x condition) with the ``group_by``
        columns, ``genotype``, one ``q<level>`` column per quantile (the
        library-wide quantile-output convention, e.g. ``q0.5``, ``q0.025``), and
        a trailing ``in_regime`` column (int 0/1).  ``in_regime == 1`` means all
        four cycle corners (wt, both singles, double) have their theta posterior
        (central ``regime_ci`` interval) within ``[regime_eps, 1 - regime_eps]``,
        so the epistasis is backed by in-band posterior mass; ``0`` means at
        least one corner is near saturation, so the estimate leans on the
        theta-model's extrapolation / posterior covariance and should be treated
        as model-conditional.  (This is the posterior-mass analogue of a
        measurement-window check; it does *not* separately test whether the
        growth signal exceeds the growth noise.)  Empty if no complete mutant
        cycles (wt + both singles + double) exist.

    Raises
    ------
    ValueError
        If the theta component does not implement the sample interface, if
        ``scale_constant`` is non-trivial for ``scale="mult"``, or if
        ``regime_eps``/``regime_ci`` are out of range.
    """
    from tfscreen.analysis.extract_epistasis import (
        mutant_cycle_pivot,
        _epistasis_from_corners,
    )

    if scale == "mult" and scale_constant != 1.0:
        raise ValueError(
            "scale_constant has no effect when scale='mult' (it cancels in the "
            "ratio-of-ratios). Use scale='add' or 'logit', or leave "
            "scale_constant at its default of 1.0."
        )

    if not (0.0 <= regime_eps < 0.5):
        raise ValueError(
            f"regime_eps must be in [0, 0.5); got {regime_eps}."
        )
    if not (0.0 < regime_ci < 1.0):
        raise ValueError(
            f"regime_ci must be in (0, 1); got {regime_ci}."
        )

    module = model_registry.get("theta", {}).get(orchestrator._theta)
    if module is None or not (hasattr(module, "build_calc_df")
                              and hasattr(module, "compute_theta_samples")):
        raise ValueError(
            f"extract_theta_epistasis requires the theta component to implement "
            f"build_calc_df and compute_theta_samples. "
            f"'{orchestrator._theta}' does not support this interface."
        )

    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    # Joint sample matrix: (num_sample, num_row), rows aligned to calc_df.
    calc_df, internal_cols, extra_kwargs = module.build_calc_df(
        orchestrator, manual_titrant_df)
    theta_samples = module.compute_theta_samples(
        calc_df, param_posteriors, **extra_kwargs)

    group_by = list(group_by)

    # Pivot on the row index (not the observable): mutant_cycle_pivot matches
    # wt/single/single/double per condition and returns, for each cycle, the
    # calc_df row index of each of the four corners.  Those indices then gather
    # the corresponding columns out of the joint sample matrix.
    pivot_df = calc_df[["genotype"] + group_by].copy()
    pivot_df["_row_idx"] = np.arange(len(calc_df))
    cycles = mutant_cycle_pivot(pivot_df,
                                extract_columns=["_row_idx"],
                                group_by=group_by)

    # Drop any cycle missing a corner (e.g. a double whose single parent has no
    # theta row): without all four joint samples epistasis is undefined. The
    # missing corner surfaces as a NaN row index from the pivot's reindex.
    idx_cols = [f"{c}__row_idx" for c in ("00", "10", "01", "11")]
    if not cycles.empty:
        cycles = cycles.dropna(subset=idx_cols)

    if cycles.empty:
        return pd.DataFrame(columns=["genotype"] + group_by
                            + list(q_to_get) + ["in_regime"])

    idx_00 = cycles["00__row_idx"].values.astype(int)
    idx_10 = cycles["10__row_idx"].values.astype(int)
    idx_01 = cycles["01__row_idx"].values.astype(int)
    idx_11 = cycles["11__row_idx"].values.astype(int)

    # Each of these is (num_sample, num_cycle); epistasis is computed per draw.
    ep_samples = _epistasis_from_corners(
        theta_samples[:, idx_00],
        theta_samples[:, idx_10],
        theta_samples[:, idx_01],
        theta_samples[:, idx_11],
        scale=scale,
        scale_constant=scale_constant,
    )

    out = cycles[["genotype"] + group_by].copy()
    for q_name, q_val in q_to_get.items():
        out[q_name] = np.quantile(ep_samples, q_val, axis=0)

    # in_regime: are all four cycle corners' theta posteriors inside the
    # resolvable band [regime_eps, 1 - regime_eps]?  Outside it logit(theta)
    # saturates and the epistasis leans on the theta-model extrapolation, so the
    # flag marks whether the estimate is backed by in-band posterior mass.  A
    # MAP checkpoint has a single "draw", so the interval collapses to the point
    # estimate and this reduces to a point-value band check.
    lo_q = (1.0 - regime_ci) / 2.0
    hi_q = 1.0 - lo_q

    def _corner_in_band(idx):
        th = theta_samples[:, idx]                     # (num_sample, num_cycle)
        lo = np.quantile(th, lo_q, axis=0)
        hi = np.quantile(th, hi_q, axis=0)
        return (lo >= regime_eps) & (hi <= 1.0 - regime_eps)

    in_regime = (_corner_in_band(idx_00) & _corner_in_band(idx_10)
                 & _corner_in_band(idx_01) & _corner_in_band(idx_11))
    out["in_regime"] = in_regime.astype(int)

    return out.sort_values(["genotype"] + group_by).reset_index(drop=True)


def extract_theta_unmeasured(orchestrator, posteriors, target_genotypes,
                            manual_titrant_df, q_to_get=None,
                            genotype_batch_size=2000):
    """
    Predict theta for unmeasured genotypes using per-mutation additive assembly.

    Dispatches to the active theta component's ``predict_unmeasured`` function.
    Genotypes that contain any mutation not seen during training are returned
    with NaN quantiles.

    Processes genotypes in batches of ``genotype_batch_size`` to avoid OOM
    when the epistasis pair matrix (N_genotype × N_pair) would be too large
    to materialise all at once.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Fitted model instance (must carry ``mut_labels``, ``pair_labels``,
        ``growth_tm``, and ``priors``).
    posteriors : dict or str
        Posterior samples (dict, NpzFile, or path to .npz/.h5 file).
    target_genotypes : list[str]
        Genotype strings to predict.  Format: slash-separated mutations
        (e.g. ``"M42I/K84L"``) or ``"wt"`` for wild-type.
    manual_titrant_df : pd.DataFrame
        Must have ``'titrant_name'`` and ``'titrant_conc'`` columns specifying
        the concentration grid.  All ``titrant_name`` values must be present in
        the orchestrator.
    q_to_get : dict, optional
        Quantiles to extract.  Defaults to the standard set used by other
        extraction functions.
    genotype_batch_size : int, optional
        Number of genotypes to process per batch.  Smaller values reduce peak
        memory (pair_mat is batch_size × N_pair) at the cost of more iterations.
        Default 2000.

    Returns
    -------
    pd.DataFrame
        Columns: ``'genotype'``, ``'titrant_name'``, ``'titrant_conc'``, plus
        one column per quantile key.  Rows for genotypes with unrecognised
        mutations have NaN quantile values.

    Raises
    ------
    ValueError
        If the active theta component does not implement ``predict_unmeasured``.
    """
    module = model_registry.get("theta", {}).get(orchestrator._theta)
    if module is None or not hasattr(module, "predict_unmeasured"):
        raise ValueError(
            f"extract_theta_unmeasured requires the theta component to implement "
            f"predict_unmeasured.  '{orchestrator._theta}' does not support this interface."
        )

    q_to_get, param_posteriors = load_posteriors(posteriors, q_to_get)

    titrant_dim   = orchestrator.training_tm.tensor_dim_names.index("titrant_name")
    titrant_names = list(orchestrator.training_tm.tensor_dim_labels[titrant_dim])

    extra_kwargs = {}
    theta_priors = orchestrator.priors.theta
    if hasattr(theta_priors, "theta_tf_total_M"):
        extra_kwargs["tf_total"] = float(theta_priors.theta_tf_total_M)
    if hasattr(theta_priors, "theta_op_total_M"):
        extra_kwargs["op_total"] = float(theta_priors.theta_op_total_M)
    if hasattr(theta_priors, "theta_conc_unit_scale"):
        extra_kwargs["conc_unit_scale"] = float(theta_priors.theta_conc_unit_scale)

    target_genotypes = list(target_genotypes)
    n_total = len(target_genotypes)

    if n_total <= genotype_batch_size:
        return module.predict_unmeasured(
            target_genotypes=target_genotypes,
            titrant_names=titrant_names,
            manual_titrant_df=manual_titrant_df,
            mut_labels=orchestrator.mut_labels,
            pair_labels=orchestrator.pair_labels,
            param_posteriors=param_posteriors,
            q_to_get=q_to_get,
            **extra_kwargs,
        )

    # Batched path: process genotype_batch_size genotypes at a time so the
    # pair indicator matrix (batch × N_pair) fits in memory.
    n_batches = (n_total + genotype_batch_size - 1) // genotype_batch_size
    print(f"  Processing {n_total} genotypes in {n_batches} batches "
          f"of {genotype_batch_size}...", flush=True)

    result_dfs = []
    for batch_start in tqdm(range(0, n_total, genotype_batch_size),
                            total=n_batches, desc="theta batches"):
        batch = target_genotypes[batch_start:batch_start + genotype_batch_size]
        chunk_df = module.predict_unmeasured(
            target_genotypes=batch,
            titrant_names=titrant_names,
            manual_titrant_df=manual_titrant_df,
            mut_labels=orchestrator.mut_labels,
            pair_labels=orchestrator.pair_labels,
            param_posteriors=param_posteriors,
            q_to_get=q_to_get,
            **extra_kwargs,
        )
        result_dfs.append(chunk_df)

    return pd.concat(result_dfs, ignore_index=True)


def extract_growth_predictions(orchestrator,
                               posteriors,
                               q_to_get=None,
                               num_samples=100):
    """
    Extract predicted ln_cfu values matching the input growth data.

    This method pulls the 'growth_pred' values from the posterior samples
    and maps them back to the original rows in `orchestrator.growth_df`.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
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
        A copy of `orchestrator.growth_df` with new columns for the requested
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

    out_df = orchestrator.growth_df.copy()
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
