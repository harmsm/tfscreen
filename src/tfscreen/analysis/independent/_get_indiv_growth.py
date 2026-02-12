
from tfscreen.util.dataframe import (
    df_to_arrays,
    get_scaled_cfu,
    check_columns,
)
from tfscreen.analysis.independent.model_pre_growth import model_pre_growth
from tfscreen.calibration import get_wt_k
from tfscreen.models.growth import MODEL_LIBRARY

import pandas as pd
import numpy as np

def _prepare_and_validate_growth_data(df,
                                      series_selector,
                                      fit_method,
                                      dk_geno_selector,
                                      lnA0_selector):
    """
    Validates inputs, prepares CFU columns, sorts, and extracts
    per-series metadata.
    """

    # 1. Validate fit_method and get required columns
    if fit_method not in MODEL_LIBRARY:
        raise ValueError(f"fit method '{fit_method}' not recognized.")
    
    needs_columns = MODEL_LIBRARY[fit_method]["args"]
    needed_cfu_columns = [n for n in needs_columns if n not in ["t_sel"]]
    
    # 2. Prepare a clean, sorted copy of the DataFrame
    df = df.copy()
    df = get_scaled_cfu(df, needed_cfu_columns)

    # 3. Check for presence of all required metadata columns
    required_columns = ["t_pre", "t_sel", "condition_sel", "titrant_name", "titrant_conc"]
    required_columns.extend(series_selector)
    if dk_geno_selector:
        required_columns.extend(dk_geno_selector)
    if lnA0_selector:
        required_columns.extend(lnA0_selector)
    check_columns(df, required_columns=list(set(required_columns)))

    # 4. Sort to ensure consistent processing order
    df = df.sort_values(by=series_selector + ["t_sel"])

    # 5. Extract a single row for each unique series to get its metadata
    series_metadata_df = df.loc[df.groupby(series_selector,observed=True)["t_sel"].idxmin()].copy()
    series_metadata_df.reset_index(drop=True, inplace=True)

    return df, series_metadata_df, needs_columns


def _run_batch_fits(df,
                    series_selector,
                    fit_fcn,
                    needs_columns,
                    fitter_kwargs):
    """
    Groups data by timepoint count and runs the fitter on each batch.
    """
    param_dfs = []
    pred_dfs = []

    df['_timepoint_count'] = df.groupby(series_selector,observed=True)["t_sel"].transform('size')
    
    for _, sub_df in df.groupby('_timepoint_count'):
        
        # Create a column for pivoting that represents timepoint order (0, 1, 2...)
        sub_df['_t_sel_row_number'] = (sub_df.groupby(series_selector,observed=True)["t_sel"]
                                       .rank(method='first').astype(int) - 1)
        
        # Reshape from long to wide format for the fitter
        row_ids, arrays = df_to_arrays(sub_df, pivot_on=series_selector)

        # Assemble arguments and run the fit
        kwargs = {k: arrays[k] for k in needs_columns}
        if fitter_kwargs:
            kwargs.update(fitter_kwargs)

        param_df_batch, pred_df_batch = fit_fcn(**kwargs)

        # The fitter returns one row of parameters per series.
        # Add the series identifiers to align with metadata later.
        if len(row_ids.names) > 1:
            param_df_batch[list(row_ids.names)] = row_ids.to_frame().values
        else:
            param_df_batch[row_ids.name] = row_ids.values
        param_dfs.append(param_df_batch)

        # Append predictions
        sub_df_sorted = sub_df.sort_values(by=series_selector + ["_t_sel_row_number"])
        
        # pred_df_batch is flattened from (N_series, Max_obs). 
        # We need to filter it to only include rows that match actual observations
        # in sub_df_sorted. 
        # To do this safely, we'll create a full temporary frame and merge.
        num_series = len(row_ids)
        max_obs = (sub_df['_t_sel_row_number'].max() + 1)
        
        # Reconstruct identifiers for EACH flattened row
        full_ids = row_ids.to_frame(index=False).iloc[np.repeat(np.arange(num_series), max_obs)].reset_index(drop=True)
        full_ids["_t_sel_row_number"] = np.tile(np.arange(max_obs), num_series)
        
        # Combine with predictions
        pred_df_batch_full = pd.concat([full_ids, pred_df_batch.reset_index(drop=True)], axis=1)
        
        # Drop columns in sub_df_sorted that will be replaced by pred_df_batch_full
        # (avoid suffixes in merge)
        to_drop = [c for c in pred_df_batch_full.columns if c in sub_df_sorted.columns and c not in (series_selector + ["_t_sel_row_number"])]
        sub_df_sorted = sub_df_sorted.drop(columns=to_drop)

        # Merge back into sub_df_sorted to ensure perfect alignment
        sub_df_with_pred = pd.merge(sub_df_sorted,
                                    pred_df_batch_full,
                                    on=series_selector + ["_t_sel_row_number"],
                                    how="left")
        
        pred_dfs.append(sub_df_with_pred)
            
    # Combine results from all batches
    final_param_df = pd.concat(param_dfs, ignore_index=True)
    final_pred_df = pd.concat(pred_dfs, ignore_index=True)

    return final_param_df, final_pred_df


def _apply_pre_growth_correction(param_df,
                                 series_metadata_df,
                                 calibration_data,
                                 dk_geno_selector,
                                 dk_geno_mask_col,
                                 lnA0_selector):
    """
    Applies the pre-growth model to correct initial parameter estimates.
    """
    # If fitter returns absolute A0, convert to log-space (lnA0)
    if "A0_est" in param_df.columns:
        to_lnA0 = param_df[["A0_est", "A0_std"]].rename(
            columns={"A0_est": "cfu", "A0_std": "cfu_std"}
        )
        to_lnA0 = get_scaled_cfu(to_lnA0, need_columns=["ln_cfu", "ln_cfu_std"])
        param_df["lnA0_est"] = to_lnA0["ln_cfu"].to_numpy()
        param_df["lnA0_std"] = to_lnA0["ln_cfu_std"].to_numpy()
        param_df = param_df.drop(columns=["A0_est", "A0_std"])
    
    # Get parameters needed for the model from the metadata frame
    k_wt = get_wt_k(
        condition=series_metadata_df["condition_sel"],
        titrant_name=series_metadata_df["titrant_name"],
        titrant_conc=series_metadata_df["titrant_conc"],
        calibration_data=calibration_data
    )
    t_pre = series_metadata_df["t_pre"].to_numpy()

    # Create grouping arrays for the model
    dk_geno_groups = None
    dk_geno_mask = None
    if dk_geno_selector:
        dk_geno_groups = (series_metadata_df.groupby(dk_geno_selector,observed=True)
                          .ngroup().to_numpy(dtype=int))
        dk_geno_mask = series_metadata_df[dk_geno_mask_col].to_numpy(dtype=bool)
        
    lnA0_groups = None
    if lnA0_selector:
        lnA0_groups = (series_metadata_df.groupby(lnA0_selector,observed=True)
                       .ngroup().to_numpy(dtype=int))

    # Apply the pre-growth correction model
    dk_geno, lnA0_pre, lnA0_est, lnA0_std = model_pre_growth(
        k_est=param_df["k_est"].to_numpy(),
        lnA0_est=param_df["lnA0_est"].to_numpy(),
        lnA0_std=param_df["lnA0_std"].to_numpy(),
        k_wt=k_wt,
        t_pre=t_pre,
        dk_geno_groups=dk_geno_groups,
        dk_geno_mask=dk_geno_mask,
        lnA0_groups=lnA0_groups
    )

    # Add corrected parameters back to the parameter dataframe
    param_df["dk_geno"] = dk_geno
    param_df["lnA0_pre"] = lnA0_pre
    param_df["lnA0_est"] = lnA0_est
    param_df["lnA0_std"] = lnA0_std
    
    return param_df


def _get_indiv_growth(df,
                      series_selector,
                      calibration_data,
                      fit_method="wls",
                      dk_geno_selector=None,
                      dk_geno_mask_col=None,
                      lnA0_selector=None,
                      fitter_kwargs=None):
    """
    Estimate individual growth parameters from timecourse data.

    This function orchestrates a multi-step process to determine growth
    parameters (growth rate `k`, initial population size `lnA0`) for
    multiple time series. It first performs an initial fit to the raw data
    and then applies a post-processing correction to account for a
    pre-growth phase, adjusting `lnA0` and calculating the change in growth
    rate relative to wildtype (`dk_geno`).

    Parameters
    ----------
    df : pandas.DataFrame
        A long-format DataFrame containing the timecourse data. It must
        contain columns specified in `series_selector` as well as `t_sel`,
        `t_pre`, `condition_sel`, `titrant_name`, and `titrant_conc`.
    series_selector : list[str]
        A list of column names that together uniquely identify each time
        series (e.g., `["replicate", "genotype"]`).
    calibration_data : dict
        A dictionary containing calibration data required to calculate the
        expected wild-type growth rate (`k_wt`) under various conditions.
    fit_method : str, optional
        The name of the fitting model to use for the initial parameter
        estimation, by default "wls". Must be a key in `MODEL_LIBRARY`.
    dk_geno_selector : list[str], optional
        A list of column names to group by for estimating a shared growth
        rate effect (`dk_geno`). For example, `["genotype"]` would calculate
        the average `dk_geno` for each unique genotype.
    dk_geno_mask_col : str, optional
        The name of a boolean column in `df`. If specified, only rows where
        this column is `True` will be used to calculate the average `dk_geno`
        for a group. Requires `dk_geno_selector` to also be specified.
    lnA0_selector : list[str], optional
        A list of column names to group by for estimating a shared initial,
        pre-growth population size (`lnA0_pre`). For example, `["replicate"]`
        would assume all series within a replicate started from the same
        initial population.
    fitter_kwargs : dict, optional
        A dictionary of additional keyword arguments to pass directly to the
        underlying fitting function specified by `fit_method`.

    Returns
    -------
    param_df : pandas.DataFrame
        A DataFrame with one row per unique series, containing all identifying
        metadata from `series_selector` plus the final, corrected growth
        parameters. Key columns include `k_est`, `lnA0_est`, `lnA0_std`,
        `dk_geno`, and `lnA0_pre`.
    pred_df : pandas.DataFrame
        A long-format DataFrame with the same dimensions as the input `df`,
        but with additional columns (e.g., `pred`, `obs`) containing the
        model's predictions for each timepoint.

    Raises
    ------
    ValueError
        If `fit_method` is not recognized, or if required columns are
        missing from the input DataFrame.

    See Also
    --------
    model_pre_growth : The function used for post-fit parameter correction.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     "replicate": [1, 1, 1, 2, 2, 2],
    ...     "genotype": ["wt", "wt", "wt", "mutant", "mutant", "mutant"],
    ...     "condition_sel": ["A", "A", "A", "A", "A", "A"],
    ...     "titrant_name": ["-", "-", "-", "-", "-", "-"],
    ...     "titrant_conc": [0, 0, 0, 0, 0, 0],
    ...     "t_sel": [0, 1, 2, 0, 1, 2],
    ...     "t_pre": [4, 4, 4, 4, 4, 4],
    ...     "ln_cfu": [10.1, 11.2, 12.0, 10.0, 10.5, 11.1],
    ...     "is_control": [True, True, True, False, False, False]
    ... }
    >>> df = pd.DataFrame(data)
    >>> series_selector = ["replicate", "genotype"]
    >>> # Assume calibration_data is a pre-loaded dictionary
    >>> param_df, pred_df = get_indiv_growth(
    ...     df=df,
    ...     series_selector=series_selector,
    ...     calibration_data=calibration_data,
    ...     dk_geno_selector=["genotype"],
    ...     dk_geno_mask_col="is_control",
    ...     lnA0_selector=["replicate"]
    ... )
    >>> print(param_df[["replicate", "genotype", "k_est", "dk_geno"]].round(2))
       replicate genotype  k_est  dk_geno
    0          1       wt   1.05     0.05
    1          2   mutant   0.55    -0.45

    """
    
    # Validate inputs, prepare data, and extract per-series metadata.
    df, series_metadata_df, needs_columns = _prepare_and_validate_growth_data(
        df=df,
        series_selector=series_selector,
        fit_method=fit_method,
        dk_geno_selector=dk_geno_selector,
        lnA0_selector=lnA0_selector
    )

    # Run the fitting function in batches based on timepoint counts.
    fit_fcn = MODEL_LIBRARY[fit_method]["fcn"]
    param_df, pred_df = _run_batch_fits(
        df=df,
        series_selector=series_selector,
        fit_fcn=fit_fcn,
        needs_columns=needs_columns,
        fitter_kwargs=fitter_kwargs if fitter_kwargs else {}
    )

    # Apply post-fit corrections for pre-growth phase.
    corrected_param_df = _apply_pre_growth_correction(
        param_df=param_df,
        series_metadata_df=series_metadata_df,
        calibration_data=calibration_data,
        dk_geno_selector=dk_geno_selector,
        dk_geno_mask_col=dk_geno_mask_col,
        lnA0_selector=lnA0_selector
    )
    
    overlapping_cols = series_metadata_df.columns.intersection(corrected_param_df.columns)
    series_metadata_df = series_metadata_df.drop(columns=overlapping_cols)

    # Combine the corrected parameters with the series metadata.
    # The order is guaranteed by sorting and consistent processing.
    final_param_df = pd.concat(
        [series_metadata_df.reset_index(drop=True),
         corrected_param_df.reset_index(drop=True)],
        axis=1
    )

    # Clean up temporary columns from the final dataframes.
    pred_df = pred_df.drop(columns=["_timepoint_count", "_t_sel_row_number"])
    final_param_df = final_param_df.drop(columns=["_timepoint_count"], errors="ignore")

    return final_param_df, pred_df
