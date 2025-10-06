
from tfscreen.calibration import (
    read_calibration,
    get_wt_theta
)

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error,
    FitManager,
)

from tfscreen.util import (
    read_dataframe,
    check_columns,
    get_scaled_cfu,
    chunk_by_group,
    set_categorical_genotype
)

from tfscreen.analysis import get_indiv_growth

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _prep_inference_df(df,
                       calibration_data,
                       max_batch_size):
    """
    Prepare a DataFrame for regression analysis.

    This function takes an experimental dataframe and performs several
    preprocessing steps: it ensures required columns are present, sets
    appropriate data types, creates an ordered categorical for genotypes to
    ensure consistent sorting, verifies that all combinations of
    genotype/condition are present in each replicate, divides the data into
    batches, and then merges in pre-calculated calibration constants for growth
    rate models.

    Parameters
    ----------
    df : pandas.DataFrame or str
        The input experimental data, either as a DataFrame object or a path
        to a file readable by ``read_dataframe``.
    calibration_data : dict or str
        Calibration data, either as a loaded dictionary or a path to a file
        readable by ``read_calibration``. It should contain at least the
        keys 'k_bg_df' and 'dk_cond_df'.
    max_batch_size : int
        The maximum number of experiments to group into a single regression
        batch. Genotypes are kept together within batches.

    Returns
    -------
    pandas.DataFrame
        The processed and enriched DataFrame, ready for parameter guessing and
        inference setup.

    Notes
    -----
    The function expects the input dataframe ``df`` to contain the following
    columns: 'ln_cfu', 'ln_cfu_std', 't_pre', 't_sel', 'genotype', 'library',
    'replicate', 'titrant_name', 'titrant_conc', 'condition_pre', and
    'condition_sel'.

    The function adds the following columns to the returned dataframe:
    - ``_batch_idx``: An integer identifying the regression batch.
    - ``k_bg_m``, ``k_bg_b``: Slope and intercept for background growth rate.
    - ``dk_m_pre``, ``dk_b_pre``: Slope and intercept for pre-selection condition.
    - ``dk_m_sel``, ``dk_b_sel``: Slope and intercept for selection condition.
    """

    # Reads file or copies datafame
    df = read_dataframe(df)

    # Build ln_cfu and ln_cfu_std
    df = get_scaled_cfu(df,need_columns=["ln_cfu","ln_cfu_std"])

    condition_cols = ["library","replicate",
                      "titrant_name","titrant_conc",
                      "condition_pre","t_pre",
                      "condition_sel","t_sel"]
    value_cols = ["ln_cfu","ln_cfu_std"]
    
    # Make sure all required columns are present
    required_columns = ["genotype"] + condition_cols + value_cols
    check_columns(df,required_columns)

    # Nuke extra columns to avoid unexpected behavior. 
    df = df.loc[:,required_columns]

    # Clean up on replicate column. It must be Int64 because it can have 
    # missing values later. 
    df["replicate"] = df["replicate"].astype('Int64')
    
    # Clean up genotypes. This will make sure things like A57A --> 'wt' and 
    # that things like A2Q/M1V --> M1V/A2Q. It also makes the genotype column
    # into an ordered categorical.
    df = set_categorical_genotype(df,sort=True,standardize=True)

    # Make sure the genotype/condition columns are dense within each replicate
    for rep, rep_df in df.groupby(["replicate","library"]):

        # Make sure that every genotype/condition is unique
        must_be_unique = ["genotype"] + condition_cols
        if rep_df.duplicated(subset=must_be_unique).any():
            num_duplicates = rep_df.duplicated(subset=must_be_unique).sum()
            raise ValueError(
                f"DataFrame conditions must be unique. Found {num_duplicates} "
                f"duplicate combinations of the key columns: {must_be_unique} "
                f"in replicate {rep}."
            )
        
        # Make sure the dataframe has the right number of rows. Since we already 
        # checked for duplicates, this is a rigorous check for a dense dataframe. 
        num_genotypes = rep_df["genotype"].drop_duplicates().shape[0]
        num_conditions = rep_df[condition_cols].drop_duplicates().shape[0]
        expected_nrows = num_genotypes*num_conditions
        rep_df_nrows = rep_df.shape[0]
        if expected_nrows != rep_df_nrows:
            raise ValueError(
                f"DataFrame for replicate {rep} is missing values. It has "
                f"{rep_df_nrows} rows but a dense grid requires {expected_nrows} "
                f"rows ({num_genotypes}) unique genotypes x {num_conditions} "
                "conditions)."
            )

    # Coerce these columns to be floats
    float_columns = ["ln_cfu","ln_cfu_std","t_pre","t_sel"]
    try:
        df[float_columns] = df[float_columns].astype(float)
    except Exception as e:
        raise ValueError(
            f"Could not coerce {float_columns} into floats"
        ) from e

    # Make sure the are no nan values in the float columns
    nan_values = np.isnan(df[float_columns])
    if np.any(nan_values):
        raise ValueError(
            f"nan values are not allowed in columns {float_columns}. "
            f"Found {np.sum(nan_values,axis=0)} nan values in these columns."
        )

    # Break the dataframe into batches that keep genotypes together. 
    df_genotype_idx = df.groupby(["genotype"],observed=False).ngroup()
    batches = chunk_by_group(df_genotype_idx,max_batch_size)
    batch_idx = np.concatenate([np.full(c.shape,i,dtype=int)
                                for i, c in enumerate(batches)])
    df["_batch_idx"] = batch_idx

    # Read calibration data
    calibration_data = read_calibration(calibration_data)
    
    # Background slope and titrant vs titrant
    bg_df = calibration_data["k_bg_df"]
    k_bg_m = bg_df.loc[df["titrant_name"],"m"].to_numpy()
    k_bg_b = bg_df.loc[df["titrant_name"],"b"].to_numpy()

    # Condition slope and intercept vs. theta
    dk_df = calibration_data["dk_cond_df"]
    dk_m_pre = dk_df.loc[df["condition_pre"],"m"].to_numpy()
    dk_b_pre = dk_df.loc[df["condition_pre"],"b"].to_numpy()
    dk_m_sel = dk_df.loc[df["condition_sel"],"m"].to_numpy()
    dk_b_sel = dk_df.loc[df["condition_sel"],"b"].to_numpy()

    # Record these values as entries in the dataframe. 
    df["k_bg_m"] = k_bg_m
    df["k_bg_b"] = k_bg_b
    df["dk_m_pre"] = dk_m_pre
    df["dk_b_pre"] = dk_b_pre
    df["dk_m_sel"] = dk_m_sel
    df["dk_b_sel"] = dk_b_sel

    return df

def _prep_param_guesses(df,
                        non_sel_conditions,
                        calibration_data):
    """
    Generate and attach initial guesses for regression parameters.

    This function runs preliminary, simplified fits on individual experimental
    series to estimate initial values for the main regression parameters. The
    results of these fits (`lnA0_est`, `dk_geno`) and a calculated wild-type
    theta (`wt_theta`) are added as new columns to the dataframe. These serve
    as starting points for the global least-squares optimization.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed DataFrame, typically the output of
        `_prep_inference_df`.
    non_sel_conditions : list of str
        A list of strings identifying experimental conditions considered
        non-selective. Data from these conditions are used to get an initial
        estimate of the `dk_geno` parameter.
    calibration_data : dict
        The dictionary of calibration data, required by the fitting and
        theta calculation functions.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with new columns containing parameter guesses.

    Notes
    -----
    This function adds the following columns to the returned dataframe:
    - ``lnA0_est``: Initial guess for the log of the initial cell count for
      each unique `(genotype, library, replicate)` series.
    - ``dk_geno``: Initial guess for the global fitness effect of each
      genotype, derived from non-selective conditions.
    - ``wt_theta``: Initial guess for the fractional occupancy of the binding
      site, calculated assuming a wild-type response to the titrant.
    """

    # _dk_geno_mask is a boolean mask holding whether a condition has no
    # selection and thus is a good sample to try to get a guess of 
    # dk_geno (the global effect of the genotype on growth independent of
    # theta). 
    df["_dk_geno_mask"] = df["condition_sel"].isin(non_sel_conditions)

    # genotype/library/replicate uniquely specifies a series that will 
    # have its own lnA0 that must be estimated. 
    series_selector = ["genotype","library","replicate"]

    # Run individual fits on all genotype/library/replicate growth rates. 
    indiv_param_df, _ = get_indiv_growth(df,
                                         series_selector=series_selector,
                                         calibration_data=calibration_data,
                                         dk_geno_selector=["genotype"],
                                         dk_geno_mask_col="_dk_geno_mask",
                                         lnA0_selector=series_selector)

    # make sure the genotype column is preserved as a category (which preserves
    # it's pretty sorting). 
    indiv_param_df["genotype"] = indiv_param_df["genotype"].astype(df["genotype"].dtype)
    
    # Grab guess values from the individual fits. This will bring lnA0_est 
    # and dk_geno into the parameter dataframe.
    df = df.merge(indiv_param_df,
                  on=series_selector,
                  how='left')
    
    # Get the theta for wildtype under these conditions
    df["wt_theta"] = get_wt_theta(df["titrant_name"].to_numpy(),
                                  df["titrant_conc"].to_numpy(),
                                  calibration_data=calibration_data)
    
    return df

def _build_param_df(df,
                    series_selector,
                    base_name,
                    guess_column,
                    transform,
                    offset):
    """
    Build a parameter specification DataFrame for a single parameter class.

    This helper function identifies a set of unique model parameters based on
    the columns in ``series_selector``. It creates a DataFrame where each row
    defines one parameter, populating it with a unique name, an initial guess,
    and transformation properties. It also generates a mapping array that links
    each row in the input ``df`` back to its corresponding parameter index.

    Parameters
    ----------
    df : pandas.DataFrame
        The main, fully prepped DataFrame containing experimental data and
        initial guess columns.
    series_selector : list of str
        A list of column names that together uniquely define a parameter of
        this class. For example, ``['genotype', 'library', 'replicate']``.
    base_name : str
        The class name for this block of parameters (e.g., 'lnA0', 'theta').
        Used for naming and classification.
    guess_column : str
        The name of the column in ``df`` that holds the initial guess for
        each parameter.
    transform : str
        The name of the transformation to apply to this parameter class during
        regression (e.g., 'scale', 'logistic', 'none').
    offset : int
        An integer to add to all generated parameter indices. This is used to
        ensure that indices from different parameter classes do not overlap.

    Returns
    -------
    df_idx : pandas.Series
        An array with the same length as ``df``. Each value is the global index
        of the parameter corresponding to that row in ``df``.
    final_df : pandas.DataFrame
        A DataFrame where each row defines one parameter, containing its class,
        name, guess, transform info, and unique global index ('idx').
    """

    # Group by unique combinations in series_selector
    grouper = df.groupby(series_selector, observed=True)
    
    # Get indexes pointing to unique combinations in series_selector
    # (same length as df)
    df_idx = grouper.ngroup() + offset
    
    # Get parameters (one for each unique combination)
    param_df = grouper.agg('first').reset_index()

    # Get index of parameter. This is a global counter that will cover all 
    # parameter classes. This is enforced by offset. Note that offset is added
    # both here and to the df_idx above, keepting them in sync
    param_df["idx"] = param_df.index + offset
    
    # Get parameter class and name
    joined_series = param_df[series_selector].astype(str).agg('_'.join, axis=1)
    param_df["class"] = base_name
    param_df["name"] = f"{base_name}_" + joined_series
    
    # Get parameter guess
    param_df["guess"] = param_df[guess_column]
    
    # Figure out transform if requested. 
    param_df["transform"] = transform
    param_df["scale_mu"] = 0
    param_df["scale_sigma"] = 1

    # Deal with scale transform parameters
    if transform == "scale":
        
        mu = np.mean(param_df["guess"])
        sig = np.std(param_df["guess"])

        param_df["scale_mu"] = mu

        # Only overwrite the default scale_sigma of 1 if sig or mu is non-zero
        if sig > 0:
            param_df["scale_sigma"] = sig
        elif mu != 0:
            # If std dev is 0 but mean is not, use the mean
            param_df["scale_sigma"] = np.abs(mu)
        else:
            param_df["scale_sigma"] = 1.0

    # Build final clean dataframe with only required columns
    final_df = param_df[["class","name","guess",
                         "transform","scale_mu","scale_sigma","idx"]]

    return df_idx, final_df

    
def _setup_inference(df):
    """
    Assemble the final design matrix and response vector for regression.

    This function serves as the final step in preparing data for a
    least-squares fit. It orchestrates the creation of the complete parameter
    specification dataframe (`param_df`) by calling the `_build_param_df`
    helper for each parameter class (`lnA0`, `dk_geno`, `theta`). It then uses
    the parameter definitions and indices to construct the final design matrix
    `X` and the response vector `y_obs`.

    Parameters
    ----------
    df : pandas.DataFrame
        A fully prepared dataframe, typically from `_prep_param_guesses`. It
        must contain all columns needed for parameter assignment (e.g.,
        'genotype', 'library') and for predictor value calculation (e.g.,
        't_pre', 't_sel', 'dk_m_pre').

    Returns
    -------
    y_obs : numpy.ndarray
        A 1D array of shape (n_obs,) representing the response variable. This
        is the observed log(cfu) with all known constant growth terms
        subtracted out.
    y_std : numpy.ndarray
        A 1D array of shape (n_obs,) holding the standard error for each
        observation in ``y_obs``.
    X : numpy.ndarray
        A 2D design matrix of shape (n_obs, n_params). Each row is an
        observation and each column corresponds to a model parameter.
    param_df : pandas.DataFrame
        A dataframe where each row corresponds to a column in ``X``, defining
        the parameter's name, class, initial guess, and transformation.

    Notes
    -----
    The function constructs `X` and `y_obs` to represent the linear model:
    `y_obs ≈ X @ p`, where `p` is the vector of parameters to be fit.
    This corresponds to the underlying model for log-growth:
    `ln(cfu) - constants ≈ lnA0 + dk_geno*t_total + theta*mtp_mts`
    """

    # -----------------------------------------------------------------------------
    # Infer column/value combination to parameter mapping. 

    # Each of these calls builds a dataframe with parameters for all unique 
    # combinations of the selector (lnA0_selector, etc.) and then populates that
    # dataframe with guesses and parameter transformation info. The function
    # also returns an index (lnA0_idx, etc.) that indicates which rows in the main
    # dataframe correspond to which parameters in the parameter dataframe. 

    offset = 0
    lnA0_selector = ["genotype","library","replicate"]
    lnA0_idx, lnA0_df = _build_param_df(
        df=df,
        base_name="lnA0",
        series_selector=lnA0_selector,
        guess_column="lnA0_est",
        transform="scale",
        offset=offset
    )
    
    offset = lnA0_df["idx"].max() + 1
    
    dk_geno_selector = ["genotype"]
    dk_geno_idx, dk_geno_df = _build_param_df(
        df=df,
        base_name="dk_geno",
        series_selector=dk_geno_selector,
        guess_column="dk_geno",
        transform="scale",
        offset=offset
    )
    
    offset = dk_geno_df["idx"].max() + 1
    
    theta_selector = ["genotype","titrant_name","titrant_conc"]
    theta_idx, theta_df = _build_param_df(
        df=df,
        base_name="theta",
        series_selector=theta_selector,
        guess_column="wt_theta",
        transform="logistic",
        offset=offset
    )

    # -----------------------------------------------------------------------------
    # build final parameter dataframe
    
    param_df = pd.concat([lnA0_df,dk_geno_df,theta_df],ignore_index=True)
    
    # -----------------------------------------------------------------------------
    # Construct design matrix
    
    # Initialize design matrix
    X = np.zeros((df.shape[0],param_df.shape[0]),dtype=float)
    
    # Populate design matrix
    row_indexer = np.arange(X.shape[0],dtype=int)
    X[row_indexer,lnA0_idx] = np.ones(len(df),dtype=float)
    X[row_indexer,dk_geno_idx] = df["t_pre"] + df["t_sel"]
    X[row_indexer,theta_idx] = df["dk_m_pre"]*df["t_pre"] + df["dk_m_sel"]*df["t_sel"]

    # -----------------------------------------------------------------------------
    # build y_obs and y_std
    
    # Build final y_obs  (ln_cfu - constant terms)
    k_bg = df["k_bg_b"] + df["k_bg_m"]*df["titrant_conc"]
    y_offset = (k_bg + df["dk_b_pre"])*df["t_pre"] + (k_bg + df["dk_b_sel"])*df["t_sel"]
    y_obs = (df["ln_cfu"] - y_offset).to_numpy()

    # y_std is unchanged
    y_std = df["ln_cfu_std"].to_numpy()

    return y_obs, y_std, X, param_df


def _run_inference(fm):
    """
    Execute a weighted least-squares regression for one batch.

    This function takes a fully prepared FitManager object, which encapsulates
    the model for a single batch of data. It clips the initial parameter
    guesses to ensure they are within the specified bounds, runs the weighted
    least-squares optimization, calculates the model's predictions with error
    propagation, and returns the results in formatted dataframes.

    Parameters
    ----------
    fm : FitManager
        A FitManager object containing all necessary components for the fit:
        the design matrix (X), observations (y_obs), standard errors (y_std),
        a parameter dataframe, and methods for parameter transformation.

    Returns
    -------
    param_df : pandas.DataFrame
        A dataframe containing the final parameter estimates and standard errors
        for the current batch, along with all other parameter metadata from the
        input FitManager.
    pred_df : pandas.DataFrame
        The input dataframe for the batch, now containing new columns with the
        model's prediction (`calc_est`) and its standard error (`calc_std`)
        for each observation.
    """

    # Clip the guesses to keep them within the bounds. (This mainly applies to
    # logistic transformed data, because guesses of 0.0 and 1.0 will be outside
    # the bounds of 1e-7 and 0.9999999 we set for numerical stability). 
    clipped_guesses = fm.guesses_transformed.copy()
    too_low = clipped_guesses < fm.lower_bounds_transformed
    clipped_guesses[too_low] = fm.lower_bounds_transformed[too_low] 
    too_high = clipped_guesses > fm.upper_bounds_transformed
    clipped_guesses[too_high] = fm.upper_bounds_transformed[too_high] 

    is_nan = np.isnan(fm.y_obs)
    if np.any(is_nan):
        raise ValueError(
            f"y_obs contains {np.sum(is_nan)} nan values. Cannot run fit."
        )
    
    is_nan = np.isnan(fm.y_std)
    if np.any(is_nan):
        raise ValueError(
            f"y_std contains {np.sum(is_nan)} nan values. Cannot run fit."
        )

    # Run least squares fit
    params, std_errors, cov_matrix, _ = run_least_squares(
        fm.predict_from_transformed,
        obs=fm.y_obs,
        obs_std=fm.y_std,
        guesses=clipped_guesses,
        lower_bounds=fm.lower_bounds_transformed,
        upper_bounds=fm.upper_bounds_transformed
    )
    
    # Make predictions at each data point and store
    pred, pred_std = predict_with_error(fm.predict_from_transformed,
                                        params,
                                        cov_matrix)

    # Prediction dataframe
    pred_df = pd.DataFrame({"y_obs":fm.y_obs,
                            "y_std":fm.y_std,
                            "calc_est":pred,
                            "calc_std":pred_std})

    # Extract parameter estimates
    param_df = fm.param_df.copy()
    param_df["est"] = fm.back_transform(params)
    param_df["std"] = fm.back_transform_std_err(params,std_errors)

    return param_df, pred_df


def cfu_to_theta(df,
                 non_sel_conditions,
                 calibration_data,
                 max_batch_size=250):
    """
    Estimate transcription factor occupancy (theta) from cell growth data.

    This function implements a global regression model to infer the fractional
    occupancy of a transcription factor binding site (`theta`) from time-course
    cell fitness data. It orchestrates a complete analysis pipeline:

    1. Preprocesses the input data, adding calibration constants.
    2. Generates reasonable initial guesses for model parameters.
    3. Splits the data into manageable batches.
    4. For each batch, constructs a design matrix and runs a weighted
       least-squares regression to fit the model parameters.
    5. Aggregates and returns the results from all batches.

    Parameters
    ----------
    df : pandas.DataFrame or str
        Input experimental data, containing columns for CFU counts, times,
        conditions, genotypes, titrant concentrations, etc.
    non_sel_conditions : list of str
        A list of strings identifying experimental conditions considered
        "non-selective". This data is used to get an initial estimate for the
        baseline fitness effect of each genotype (`dk_geno`).
    calibration_data : dict or str
        A dictionary or path to a file containing pre-calculated calibration
        constants, such as background growth rates.
    max_batch_size : int, optional
        The maximum number of experiments to include in a single regression
        batch. This helps manage memory usage for large datasets.
        (default is 250).

    Returns
    -------
    param_df : pandas.DataFrame
        A dataframe containing the final estimated value (`est`) and standard
        error (`std`) for every model parameter (`lnA0`, `dk_geno`, `theta`)
        across all fitted batches.
    pred_df : pandas.DataFrame
        A dataframe containing the original experimental data for all batches,
        augmented with the model's prediction (`calc_est`) and its standard
        error (`calc_std`) for each observation.
    """
    
    
    # Prepare dataframe, creating all necessary columns etc. 
    df = _prep_inference_df(df,
                            max_batch_size=max_batch_size,
                            calibration_data=calibration_data)
    
    # get parameter guesses
    df = _prep_param_guesses(df,
                             non_sel_conditions=non_sel_conditions,
                             calibration_data=calibration_data)

    
    # Lists to store batch-wise results
    batch_param_dfs = []
    batch_pred_dfs = []

    # For batches of genotypes...
    for batch_idx, batch_df in tqdm(df.groupby(["_batch_idx"])):

        # Build the design matrix
        y_obs, y_std, X, fit_param_df = _setup_inference(batch_df)

        # Construct a FitManager object to manage the fit. 
        fm = FitManager(y_obs,y_std,X,fit_param_df)

        # Run the inference
        batch_param_df, batch_pred_df = _run_inference(fm)

        # Combine the original batch data with its new predictions
        batch_pred_df = pd.concat([batch_df.reset_index(drop=True), 
                                   batch_pred_df],axis=1)
        
        batch_param_dfs.append(batch_param_df)
        batch_pred_dfs.append(batch_pred_df)

    param_df = pd.concat(batch_param_dfs,ignore_index=True)
    pred_df = pd.concat(batch_pred_dfs,ignore_index=True)

    return param_df, pred_df