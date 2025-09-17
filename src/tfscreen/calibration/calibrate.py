
from ._models import (
    hill_model,
    simple_poly
)

from tfscreen.calibration import (
    write_calibration,
    read_calibration
)

from tfscreen.util import read_dataframe
from tfscreen.fitting import (
    run_least_squares,
    predict_with_error
)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


def _fit_theta(df):
    """
    Fit a Hill model to theta vs. titrant for each titrant.

    This function iterates through each unique `titrant_name` present in the
    input DataFrame, extracts the corresponding concentration and theta values,
    generates initial parameter guesses, and performs a non-linear
    least-squares fit using the Hill model.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the experimental data. It must contain the
        following columns:
        - 'titrant_name': An identifier for the titrant being measured.
        - 'titrant_conc': The concentration of the titrant (x-values).
        - 'theta': The measured fractional saturation or response (y-values).
        - 'theta_std': The standard deviation of 'theta' (y-errors).

    Returns
    -------
    dict
        A dictionary where keys are the unique `titrant_name` strings from
        the input DataFrame. The corresponding values are NumPy arrays
        containing the four fitted Hill parameters in the order:
        [baseline, amplitude, K, n].

    Notes
    -----
    The initial parameter guesses are tailored for a repressor system where
    the signal (`theta`) starts high and decreases with increasing titrant
    concentration. The guesses may not be suitable for activator-like systems.
    """

    out_dict = {}
    for titr, titr_df in df.groupby("titrant_name"):

        # Grab sub dataframe with only titrant_name
        titr_df = df[df["titrant_name"] == titr]

        # Pull out x, y and y_std
        x = titr_df["titrant_conc"].to_numpy()
        y = titr_df["theta"].to_numpy()
        y_std = titr_df["theta_std"].to_numpy()

        # Some simple guesses. (Note, this was built for a repressor and
        # assumes the baseline starts high, then drops with increasing 
        # titrant). 
        baseline_guess = np.max(y)
        amplitude_guess = np.min(y) - baseline_guess
        y_half = baseline_guess + 0.5 * amplitude_guess
        K_guess = x[np.argmin(np.abs(y - y_half))]
        n_guess = 2

        # Set up guess array 
        guesses = np.array([baseline_guess,
                            amplitude_guess,
                            K_guess,
                            n_guess])

        # Run fit (nonlinear least squares) 
        params, _, _, _ = run_least_squares(
            hill_model,
            y,
            y_std,
            guesses=guesses,
            args=(x,)
        )

        # record parameters
        out_dict[titr] = params

    return out_dict

def _calculate_log_population(params,
                              theta,
                              titrant_conc,
                              pre_time,
                              time,
                              not_bg,
                              b_pre_idx,
                              m_pre_idx,
                              b_idx,
                              m_idx,
                              A0_idx,
                              bg_param_idx):
    """
    Calculate the log-transformed population size (lnA) from model parameters.

    This function implements the forward model for predicting an observable (lnA)
    based on a global parameter vector and the specific experimental conditions
    for each data point. It combines background growth with a perturbation term
    that depends on transcription factor occupancy (`theta`).

    Parameters
    ----------
    params : numpy.ndarray
        A 1D array of float values representing all global model parameters.
    theta : numpy.ndarray
        A 1D array containing the fractional occupancy of the transcription
        factor for each observation. Same length as `time`.
    titrant_conc : numpy.ndarray
        A 1D array of titrant concentrations for each observation.
    pre_time : numpy.ndarray
        A 1D array of pre-incubation times for each observation.
    time : numpy.ndarray
        A 1D array of main incubation times for each observation.
    not_bg : numpy.ndarray
        A 1D boolean array. `True` for observations that are not background
        controls, `False` for those that are. This acts as a mask to
        include the perturbation terms.
    b_pre_idx : numpy.ndarray
        A 1D integer array mapping each observation to its corresponding
        'b_pre' (pre-incubation intercept) parameter in `params`.
    m_pre_idx : numpy.ndarray
        A 1D integer array mapping each observation to its corresponding
        'm_pre' (pre-incubation slope) parameter in `params`.
    b_idx : numpy.ndarray
        A 1D integer array mapping each observation to its corresponding
        'b' (main incubation intercept) parameter in `params`.
    m_idx : numpy.ndarray
        A 1D integer array mapping each observation to its corresponding
        'm' (main incubation slope) parameter in `params`.
    A0_idx : numpy.ndarray
        A 1D integer array mapping each observation to its corresponding
        'lnA0' (initial log population) parameter in `params`.
    bg_param_idx : numpy.ndarray
        A 2D integer array mapping each observation to its corresponding
        background growth polynomial coefficients in `params`. Shape is (M, D)
        where M is the number of observations and D is the polynomial degree + 1.

    Returns
    -------
    numpy.ndarray
        A 1D array of the calculated lnA values for each observation.

    Notes
    -----
    The various `_idx` arrays and the `not_bg` mask collectively function as a
    design matrix, selecting the appropriate parameters from the global `params`
    vector for the calculation of each data point.
    """

    # starting population
    lnA0 = params[A0_idx]

    # Background growth over pre_time + time
    k_bg_t = simple_poly(params[bg_param_idx.T],titrant_conc)*(pre_time + time)

    # Pertubation to background growth due to occupancy of the transcription
    # factor (theta) and current conditions. 
    k_pre_t = (params[b_pre_idx] + params[m_pre_idx]*theta)*pre_time
    k_t = (params[b_idx] + params[m_idx]*theta)*time

    # Final population. Note we only apply k_pre_t and k_t if this is not a
    # 'background' sample (meaning no perturbation)
    lnA = lnA0 + k_bg_t + not_bg*(k_pre_t + k_t)

    return lnA

def _get_linear_model_df(df):
    """
    Create a mapping DataFrame for linear model parameters.

    This function identifies all unique (condition, titrant_name) pairs from
    the `pre_condition` and `condition` columns of the input DataFrame. It then
    assigns a unique parameter index for an intercept (`b_idx`) and a slope
    (`m_idx`) to each non-"background" pair. Pairs where the condition is
    "background" are assigned a dummy index of 0 for both parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing experimental conditions. Must include the
        columns 'pre_condition', 'condition', and 'titrant_name'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with a `MultiIndex` of ('condition', 'titrant_name').
        The columns are:
        - 'b_idx': Integer index for the intercept parameter.
        - 'm_idx': Integer index for the slope parameter.

    Notes
    -----
    - The slope indices (`m_idx`) are generated as a block that is offset from
      the intercept indices (`b_idx`). For N non-background pairs, `b_idx`
      will be `0..N-1` and `m_idx` will be `N..2N-1`.
    - "background" conditions are not fit with this linear model and are
      assigned a placeholder index of 0, which is ignored in downstream
      fitting steps due to a boolean mask.
    """

    # 1. Use pd.melt to unpivot 'pre_condition' and 'condition' columns
    id_vars = ["titrant_name"]
    value_vars = ["pre_condition", "condition"]
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars,
                        value_name="condition_unified")

    # 2. Find all unique pairs and rename the column
    unique_pairs = melted_df[["condition_unified", "titrant_name"]] \
                        .drop_duplicates().dropna() \
                        .rename(columns={"condition_unified": "condition"})

    # Create the correctly structured empty DataFrame that we will return
    # in all edge cases or concatenate onto later.
    empty_indexed_df = pd.DataFrame(
        {'b_idx': pd.Series(dtype=int), 'm_idx': pd.Series(dtype=int)}
    ).set_index(pd.MultiIndex(levels=[[], []], codes=[[], []],
                             names=['condition', 'titrant_name']))

    if unique_pairs.empty:
        return empty_indexed_df

    # 3. Separate non-background from background pairs
    is_bg = unique_pairs["condition"] == "background"
    non_bg_pairs = unique_pairs[~is_bg]
    bg_pairs = unique_pairs[is_bg]

    # 4. Assign parameter indices to non-background pairs
    num_non_bg = len(non_bg_pairs)
    if num_non_bg > 0:
        b_idx = np.arange(num_non_bg, dtype=int)
        m_idx = b_idx[-1] + 1 + np.arange(num_non_bg, dtype=int)
        non_bg_df = pd.DataFrame({
            "b_idx": b_idx,
            "m_idx": m_idx
        }, index=pd.MultiIndex.from_frame(non_bg_pairs))
    else:
        # If no non-background pairs, use the empty template
        non_bg_df = empty_indexed_df

    # 5. Assign dummy index (0) to background pairs
    if not bg_pairs.empty:
        bg_df = pd.DataFrame({
            "b_idx": 0,
            "m_idx": 0
        }, index=pd.MultiIndex.from_frame(bg_pairs))
    else:
        # If no background pairs, use the empty template
        bg_df = empty_indexed_df

    # 6. Return the combined dataframe
    return pd.concat([non_bg_df, bg_df])


@dataclass
class FitSetup:
    """
    A container for the setup needed for the linear model fit.
    """

    # DataFrames needed for parsing results
    linear_model_df: pd.DataFrame
    A0_df: pd.DataFrame

    # Initial parameter guesses
    initial_guesses: np.ndarray

    # Index mapping arrays (the "design matrix")
    b_pre_idx: np.ndarray
    m_pre_idx: np.ndarray
    b_idx: np.ndarray
    m_idx: np.ndarray
    A0_idx: np.ndarray
    bg_param_idx: np.ndarray
    not_bg: np.ndarray

    # Helper lookup for background parameters
    bg_results_lookup: dict

    def get_args_tuple(self, theta, titrant_conc, pre_time, time) -> tuple:
        """
        Assembles the arguments tuple required by the prediction model.
        """
        return (
            theta, titrant_conc, pre_time, time, self.not_bg,
            self.b_pre_idx, self.m_pre_idx, self.b_idx, self.m_idx,
            self.A0_idx, self.bg_param_idx
        )

@dataclass
class FitResult:
    """
    A container for the final, parsed results of the fit.
    """

    linear_model_df: pd.DataFrame
    bg_model_param: dict
    pred_df: pd.DataFrame
    A0_df: pd.DataFrame

def _build_fit_setup(df: pd.DataFrame, bg_model_guesses: list, lnA0_guess: float) -> FitSetup:
    """
    Builds the design matrix, index arrays, and initial parameter guesses.

    This function prepares all inputs required for the least-squares fit,
    encapsulating the complex logic of mapping data rows to a global
    parameter vector.
    """

    # Build dataframe for b/m parameter lookups
    linear_model_df = _get_linear_model_df(df)

    # --- Create Index Arrays for Linear Model ---
    pre_keys = list(zip(df['pre_condition'], df['titrant_name']))
    b_pre_idx = linear_model_df.loc[pre_keys, "b_idx"].to_numpy()
    m_pre_idx = linear_model_df.loc[pre_keys, "m_idx"].to_numpy()

    keys = list(zip(df['condition'], df['titrant_name']))
    b_idx = linear_model_df.loc[keys, "b_idx"].to_numpy()
    m_idx = linear_model_df.loc[keys, "m_idx"].to_numpy()

    # --- Create Index Array for lnA0 ---
    A0_df = df[["replicate"]].drop_duplicates().sort_values("replicate").reset_index(drop=True)
    A0_start = np.max(m_idx) + 1 if len(m_idx) > 0 else 0
    replicate_codes = pd.factorize(df['replicate'])[0]
    A0_idx = replicate_codes + A0_start
    A0_df["A0_idx"] = np.arange(A0_start, len(A0_df) + A0_start, dtype=int)

    # --- Create Index Array for Background Model (Vectorized) ---
    num_bg_model_params = len(bg_model_guesses)
    titr_codes, titr_uniques = pd.factorize(df["titrant_name"])
    
    bg_start = np.max(A0_idx) + 1 if len(A0_idx) > 0 else 0
    start_indices = bg_start + (titr_codes * num_bg_model_params)
    bg_param_idx = start_indices[:, np.newaxis] + np.arange(num_bg_model_params)
    
    # --- Create Background Parameter Lookup ---
    bg_results_lookup = {}
    for i, titr in enumerate(titr_uniques):
        start = bg_start + i * num_bg_model_params
        bg_results_lookup[titr] = np.arange(start, start + num_bg_model_params)

    # --- Initialize Parameter Vector ---
    num_bg_params = len(titr_uniques) * num_bg_model_params
    num_params = bg_start + num_bg_params
    initial_guesses = np.zeros(num_params, dtype=float)
    initial_guesses[A0_idx] = lnA0_guess
    # b/m and pre_b/pre_m guesses default to 0.

    return FitSetup(
        linear_model_df=linear_model_df,
        A0_df=A0_df,
        initial_guesses=initial_guesses,
        b_pre_idx=b_pre_idx, m_pre_idx=m_pre_idx,
        b_idx=b_idx, m_idx=m_idx,
        A0_idx=A0_idx, bg_param_idx=bg_param_idx,
        not_bg=(df["condition"] != "background").to_numpy(),
        bg_results_lookup=bg_results_lookup
    )

def _prepare_fit_data(df: pd.DataFrame) -> dict:
    """
    Extracts and transforms observable data from the DataFrame for fitting.
    """

    cfu = df['cfu_per_mL'].to_numpy()
    cfu_var = df['cfu_per_mL_std'].to_numpy()**2

    # Prevent division by zero or log(0) errors for bad data points
    cfu[cfu <= 0] = 1
    cfu_var[cfu_var <= 0] = 1e-9

    ln_cfu = np.log(cfu)
    ln_cfu_var = cfu_var / (cfu**2)

    return {
        "y": ln_cfu,
        "y_var": ln_cfu_var,
        "theta": df["theta"].to_numpy(),
        "titrant_conc": df["titrant_conc"].to_numpy(),
        "pre_time": df["pre_time"].to_numpy(),
        "time": df["time"].to_numpy(),
    }

def _parse_fit_results(
    params: np.ndarray,
    std_errors: np.ndarray,
    cov_matrix: np.ndarray,
    fit_setup: FitSetup,
    fit_data: dict
) -> FitResult:
    """
    Parses raw fitter output into structured DataFrames and dictionaries.
    """

    # Populate linear model results
    lm_df = fit_setup.linear_model_df.copy()
    lm_df["b_est"] = params[lm_df["b_idx"]]
    lm_df["b_std"] = std_errors[lm_df["b_idx"]]
    lm_df["m_est"] = params[lm_df["m_idx"]]
    lm_df["m_std"] = std_errors[lm_df["m_idx"]]

    # Drop the dummy "background" condition 
    keep_idx = lm_df.index.get_level_values('condition') != 'background'
    lm_df = lm_df.loc[keep_idx, :]

    # Populate A0 results
    a0_df = fit_setup.A0_df.copy()
    a0_df["A0_est"] = params[a0_df["A0_idx"]]
    a0_df["A0_std"] = std_errors[a0_df["A0_idx"]]

    # Build prediction dataframe with uncertainty
    args = fit_setup.get_args_tuple(
        fit_data["theta"], fit_data["titrant_conc"],
        fit_data["pre_time"], fit_data["time"]
    )
    calc_values, calc_std = predict_with_error(
        _calculate_log_population, params, cov_matrix, args=args
    )
    pred_df = pd.DataFrame({
        "calc_est": calc_values, "calc_std": calc_std,
        "obs_est": fit_data["y"], "obs_std": np.sqrt(fit_data["y_var"])
    })

    # Record the background model parameters
    bg_model_param = {}
    for titr, idx in fit_setup.bg_results_lookup.items():
        bg_model_param[titr] = params[idx]

    return FitResult(
        linear_model_df=lm_df,
        bg_model_param=bg_model_param,
        pred_df=pred_df,
        A0_df=a0_df
    )


def _fit_linear_model(
    df: pd.DataFrame,
    bg_model_guesses: list,
    lnA0_guess: float = 16
) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """
    Fits the core calibration model by orchestrating helper functions.

    This function serves as a high-level orchestrator for the entire fitting
    process. It delegates complex tasks to specialized helper functions,
    making the workflow more modular and readable. The process involves:
    1. Building the design matrix and initial parameter guesses.
    2. Preparing the observational data for fitting.
    3. Executing the non-linear least-squares optimization.
    4. Parsing the raw fitter output into user-friendly results.

    Parameters
    ----------
    df : pandas.DataFrame
        The compiled experimental data. Must contain columns such as 'replicate',
        'condition', 'pre_condition', 'titrant_name', 'theta', 'cfu_per_mL',
        and 'cfu_per_mL_std'.
    bg_model_guesses : list
        A list or array of initial guesses for the parameters of the background
        growth model. The length of this list determines the model's complexity
        (e.g., number of polynomial terms).
    lnA0_guess : float, optional
        An initial guess for the log-transformed initial population size (lnA0),
        by default 16.

    Returns
    -------
    tuple
        A tuple containing four structured result objects:
        - linear_model_df (pandas.DataFrame): Contains the estimated slope (`m_est`)
          and intercept (`b_est`) parameters and their standard errors, indexed
          by (condition, titrant_name).
        - bg_model_param (dict): A dictionary mapping each titrant name to its
          fitted background growth model parameters.
        - pred_df (pandas.DataFrame): Contains the observed and calculated
          `ln(cfu)` values along with their standard deviations for diagnostics.
        - A0_df (pandas.DataFrame): Contains the estimated `lnA0` value and its
          standard error for each replicate.
    """

    # 1. Build the design matrix and parameter guesses
    fit_setup = _build_fit_setup(df, bg_model_guesses, lnA0_guess)

    # 2. Prepare the observable data for fitting
    fit_data = _prepare_fit_data(df)

    # 3. Assemble args 
    args = fit_setup.get_args_tuple(
        fit_data["theta"], fit_data["titrant_conc"],
        fit_data["pre_time"], fit_data["time"]
    )

    # run the least-squares fit
    params, std_errors, cov_matrix, fit = run_least_squares(
        _calculate_log_population,
        fit_data["y"],
        fit_data["y_var"],
        guesses=fit_setup.initial_guesses,
        args=args
    )

    # 4. Parse the raw results into final, clean outputs
    results = _parse_fit_results(params, std_errors, cov_matrix, fit_setup, fit_data)

    return (
        results.linear_model_df,
        results.bg_model_param,
        results.pred_df,
        results.A0_df
    )


def calibrate(
    df: pd.DataFrame or str,
    output_file: str,
    bg_order: int = 0,
    bg_k_guess: float = 0.025,
    lnA0_guess: float = 16,
) -> dict:
    """
    Run the full calibration workflow on experimental data.

    Takes experimental data, performs validation, runs a global non-linear
    fit to determine growth parameters (`b` and `m` vs. theta for each titrant),
    fits a Hill model to the `theta` vs. titrant data, and saves the resulting
    calibration parameters to a JSON file.

    Parameters
    ----------
    df : pandas.DataFrame or str
        A pre-loaded DataFrame or a file path (e.g., to an Excel file)
        containing the compiled experimental data.
    output_file : str
        The file path where the output JSON calibration data will be written.
    bg_order : int, optional
        The polynomial order for the background growth model (`k` vs. titrant
        concentration). `bg_order=0` (the default) fits a constant rate. 
    bg_k_guess : float, optional
        An initial guess for the constant term (c_0) of the background
        growth model, by default 0.025.
    lnA0_guess : float, optional
        An initial guess for the log-transformed initial population size (lnA0),
        by default 16.

    Returns
    -------
    dict
        A dictionary containing the final calibration parameters. The primary keys
        are 'm', 'b', 'theta_param', and 'bg_model_param'.
    pd.DataFrame
        A dataframe holding the predicted and observed ln(A) for all data points
        used for the calibration
    pd.DataFrame
        A dataframe holding the fit ln(A0) for all replicates used for the 
        calibration

    Raises
    ------
    ValueError
        - If `df` is missing any of the required columns.
        - If any `titrant_name` in the data does not have a corresponding
          row with the condition 'background'.
        - If `bg_order` is negative.
    """
    
    # Read dataframe
    df = read_dataframe(df)

    # Check for required columns
    required_columns = ["replicate",
                        "pre_condition",
                        "pre_time",
                        "condition",
                        "time",
                        "titrant_name",
                        "titrant_conc",
                        "cfu_per_mL",
                        "cfu_per_mL_std",
                        "theta",
                        "theta_std"]
    
    required_set = set(required_columns)
    seen_set = set(df.columns)
    if not required_set.issubset(seen_set):
        missing = required_set - seen_set
        err = "Not all required columns seen. Missing columns:\n"
        for c in missing:
            err += f"    {c}\n"
        err += "\n"
        raise ValueError(err)

    # Make sure we see a value of 'background' in the conditions for each 
    # titrant_name
    unique_ct = df[["condition","titrant_name"]].drop_duplicates()
    num_unique_tn = len(pd.unique(unique_ct["titrant_name"]))
    num_with_bg = len(unique_ct[unique_ct["condition"] == "background"].drop_duplicates())
    if num_unique_tn != num_with_bg:
        err = "All unique titrant_name values must have a 'background' condition.\n"
        raise ValueError(err)

    # Make sure the order of the model for the background selection is sane. 
    bg_order = int(bg_order)
    if bg_order < 0:
        err = "bg_order must be >= 0\n"
        raise ValueError(err)

    # Build guesses for model order (length specifies order)
    bg_model_guesses = np.zeros(bg_order + 1,dtype=float)
    bg_model_guesses[0] = bg_k_guess

    # Fit the k vs. theta model to the whole dataset 
    param_df, bg_model_param, pred_df, A0_df = _fit_linear_model(df,
                                                                 bg_model_guesses,
                                                                 lnA0_guess=lnA0_guess)

    # Fit a hill model to the observed theta values so we can calculate
    # approximate wildtype theta on the fly for any titrant conc
    theta_param = _fit_theta(df)
    
    # Build output dictionary with fit results
    calibration_dict = {}

    calibration_dict["m"] = param_df["m_est"].to_dict()
    calibration_dict["b"] = param_df["b_est"].to_dict()

    calibration_dict["theta_param"] = theta_param
    calibration_dict["bg_model_param"] = bg_model_param

    # Write output as json file
    write_calibration(calibration_dict=calibration_dict,
                      json_file=output_file)
    
    # Read back in calibration. (This does a bit of processing on the 
    # dictionary like internal dataframe creation so the returned calibration
    # dictionary is usable). 
    calibration_dict = read_calibration(json_file=output_file)

    return calibration_dict, pred_df, A0_df

    
