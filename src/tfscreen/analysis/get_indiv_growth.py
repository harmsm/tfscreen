from ._get_indiv_growth import _get_indiv_growth
from tfscreen.util import get_scaled_cfu

import pandas as pd

def get_indiv_growth(df,
                     series_selector,
                     calibration_data,
                     fit_method="wls",
                     dk_geno_selector=None,
                     dk_geno_mask_col=None,
                     lnA0_selector=None,
                     fitter_kwargs=None,
                     num_iterations=3):
    """
    Estimate individual growth parameters from timecourse data.

    This function orchestrates a multi-step process to determine growth
    parameters (growth rate `k`, initial population size `lnA0`) for
    multiple time series. It first performs an initial fit to the raw data
    and then applies a post-processing correction to account for a
    pre-growth phase, adjusting `lnA0` and calculating the change in growth
    rate relative to wildtype (`dk_geno`). If num_iterations > 1, the function
    runs the inference iteratively, taking the values of lnA0 from the last fit
    and putting them in as a fake t = 0 time point. In tests, num_iterations = 3
    leads to a converged estimate. 

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
    num_iterations : int, optional
        Run this many iterative fits. Default is to run 3 iterations. 

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
    
    pred_df = df.copy()

    for i in range(num_iterations):

        # fit a growth model to the data
        param_df, pred_df = _get_indiv_growth(pred_df,
                                              series_selector,
                                              calibration_data,
                                              fit_method=fit_method,
                                              dk_geno_selector=dk_geno_selector,
                                              dk_geno_mask_col=dk_geno_mask_col,
                                              lnA0_selector=lnA0_selector,
                                              fitter_kwargs=fitter_kwargs)
        
        # If we're only doing one iteration, don't go to the trouble of adding 
        # fake t = 0 data. 
        if num_iterations == 1:
            break

        # Standardize pred_df on ln_cfu and ln_cfu_std, dropping other columns
        # that might be present.
        pred_df = get_scaled_cfu(pred_df,["ln_cfu","ln_cfu_std"])
        pred_df = pred_df.drop(columns=["cfu","cfu_std","cfu_var","ln_cfu_var"],
                               errors='ignore')

        # If this is the first loop, add a fake set of rows--one for each 
        # series--that has t = 0. These points will have our estimate of lnA0 as
        # the observed ln_cfu. 
        if i == 0:

            # First loop, add a fake point marker
            pred_df["_is_fake_point"] = False

            # Grab the index corresponding to the minimum time for each 
            # series. 
            min_t_idx = pred_df.groupby(series_selector)["t_sel"].idxmin() 

            # Create a new dataframe from this minimum time point, copying all 
            # information over. Set time to 0, copy in the fit data, and declare
            # it fake. 
            new_df = pred_df.loc[min_t_idx,:].reset_index(drop=True)
            new_df["t_sel"] = 0
            new_df["ln_cfu"] = param_df["lnA0_est"].values
            new_df["ln_cfu_std"] = param_df["lnA0_std"].values
            new_df["_is_fake_point"] = True

            # Build new prediction data with the fake rows. 
            pred_df = pd.concat([pred_df,new_df],ignore_index=True)

            # Sort by series then t_sel. 
            sort_on = series_selector[:]
            sort_on.append("t_sel")
            pred_df = pred_df.sort_values(sort_on)
        
        else:

            # Later loops. Just jam in fake data from the last estimate. 
            pred_df.loc[pred_df["t_sel"] == 0,"ln_cfu"] = param_df["lnA0_est"].values
            pred_df.loc[pred_df["t_sel"] == 0,"ln_cfu_std"] = param_df["lnA0_std"].values

    # Drop the fake points from the prediction dataframe if they were made. 
    if "_is_fake_point" in pred_df.columns:
        pred_df = pred_df[~pred_df["_is_fake_point"]].reset_index()
        pred_df = pred_df.drop(columns=["_is_fake_point"])

    return param_df, pred_df
        
    