
from tfscreen.calibration import get_background
from tfscreen.analysis.get_growth_rates import get_growth_rates_wls
from tfscreen.analysis import get_time0

import pandas as pd
import numpy as np


def manual_fit(df: pd.DataFrame, calibration_data: dict) -> pd.DataFrame:
    """
    Calculate growth rates by fitting individual experimental groups.

    This function provides an alternative fitting approach to the global model,
    allowing us to test/visualize the global fit quality. It works by grouping
    the data by replicate and experimental condition, then performing a separate
    weighted least-squares regression of `ln(CFU)` vs. `time` for each group. 
    Each replicate/condition gets its own growth rate, with all conditions 
    within a replicate sharing a single initial population ln(A0). The
    resulting growth rates (`k_est`) are then aggregated across replicates to
    calculate the mean and standard error (`k_std`).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the time-course data. Must include the
        columns 'replicate', 'pre_condition', 'condition', 'titrant_name',
        'titrant_conc', 'time', 'cfu_per_mL', and 'cfu_per_mL_std'.
    calibration_data: dict or str,
        calibration_data : dict or str
        A pre-loaded calibration dictionary or the file path to the
        calibration JSON file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the fit results. It contains the mean growth
        rate (`k_est`) and the standard error of the mean (`k_std`) for each
        unique experimental condition (i.e., averaged across replicates).

    Notes
    -----
    - This function assumes that within each group (defined by replicate,
      condition, etc.), the data forms a perfect grid of `titrant_conc` vs.
      `time`. If data points are missing, the array reshaping step will fail.
    """
    
    # Work on a copy
    df = df.copy()

    # Get ln_cfu and ln_cfu_var
    cfu = df['cfu_per_mL'].to_numpy()
    cfu_var = (df['cfu_per_mL_std'].to_numpy())**2
    df["ln_cfu"] = np.log(cfu)
    df["ln_cfu_var"] = cfu_var / (cfu**2)

    # Sort in a stereotyped way so we can turn each replicate into a 2D matrix
    sort_on = ["replicate",
               "pre_condition",
               "condition",
               "titrant_name",
               "titrant_conc",
               "time"]
    df = df.sort_values(sort_on)

    # Go through each replicate with same pre_condition, condition, and 
    # titrant and build results vs. titrant_conc
    
    unique_groups = ["replicate","pre_condition","condition","titrant_name"]
    final_columns = unique_groups[:]
    final_columns.append("titrant_conc")

    fit_results = []
    for _, sub_df in df.groupby(unique_groups):

        # Grab relevant info from this sub datafrae
        times = sub_df["time"].to_numpy()
        ln_cfu = sub_df["ln_cfu"].to_numpy()
        ln_cfu_var = sub_df["ln_cfu_var"].to_numpy()
        titrant_conc = pd.unique(sub_df["titrant_conc"])

        # Get data to reshape into a 2D matrix
        num_samples = len(titrant_conc)
        new_shape = (num_samples,times.shape[0]//num_samples)

        # 2D matrices
        times = times.reshape(new_shape)
        ln_cfu = ln_cfu.reshape(new_shape)
        ln_cfu_var = ln_cfu.reshape(new_shape)

        # dummy value. function expects this to calculate k_shift, which we're
        # going to ignore. 
        no_select_mask = np.zeros(new_shape[0],dtype=bool)

        # Calculate wildtype background growth rate for pre-growth
        no_time_sub_df = sub_df[sort_on[:-1]].drop_duplicates()
        k_wt = get_background(no_time_sub_df["titrant_name"],
                              no_time_sub_df["titrant_conc"],
                              calibration_data)

        # Build arrays with a pseudo time 0 point because we know everyone 
        # started with the same A0. 
        _, times, ln_cfu, ln_cfu_var = get_time0(times,
                                                 ln_cfu,
                                                 ln_cfu_var,
                                                 no_select_mask,
                                                 k_wt=k_wt)

        # Run the regression 
        param_df, _ = get_growth_rates_wls(times,
                                           ln_cfu,
                                           ln_cfu_var)

        # Dataframe holding results for this replicate
        out_df = sub_df[final_columns].drop_duplicates()
        out_df["k_est"] = param_df["k_est"].to_numpy()
    
        fit_results.append(out_df)

    # Build single mega dataframe with all fit results
    fit_df = pd.concat(fit_results)

    # Take mean and sem of k_est across replicates
    group = ["pre_condition","condition","titrant_name","titrant_conc"]
    fit_df = (
        fit_df.groupby(group)["k_est"]
        .agg(k_est="mean", k_std="sem") 
        .reset_index()
    )

    return fit_df