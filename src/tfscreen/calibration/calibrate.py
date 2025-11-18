
from tfscreen.models.generic import MODEL_LIBRARY

from tfscreen.calibration import (
    write_calibration,
    read_calibration
)

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error,
    FitManager,
    parse_patsy
)

from tfscreen.util import (
    read_dataframe,
    check_columns,
    get_scaled_cfu
)

import numpy as np
import pandas as pd
import patsy

import warnings

from tfscreen.analysis.independent.cfu_to_theta import _build_param_df

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

        hill_model = MODEL_LIBRARY["hill_repressor"]["model_func"]
        guess_fcn = MODEL_LIBRARY["hill_repressor"]["guess_func"]
        guesses = guess_fcn(x,y)

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

def _prep_calibration_df(df):

    # Reads file or copies datafame
    df = read_dataframe(df)

    # Build ln_cfu and ln_cfu_std columns
    df = get_scaled_cfu(df,need_columns=["ln_cfu","ln_cfu_std"])

    # add genotype if not present
    if "genotype" not in df.columns:
        warnings.warn("No genotype column. Assuming all calibration genotypes are 'wt'")
        df["genotype"] = "wt"

    # Add censored if not present
    if "censored" not in df.columns:
        df["censored"] = False
    
    required_columns = ["ln_cfu",
                        "ln_cfu_std",
                        "t_pre",
                        "condition_pre",
                        "t_sel",
                        "condition_sel",
                        "replicate",
                        "genotype",
                        "titrant_name",
                        "titrant_conc",
                        "theta",
                        "censored"]
    
    # Check for required columns
    check_columns(df,required_columns)

    # We do some regex-y matching with patsy. Nuke extra columns to avoid 
    # unexpected behavior. 
    df = df.loc[:,required_columns]

    return df


def _build_calibration_X_new(df):
    """
    Assemble the design matrix and response vector for calibration.

    This function constructs the necessary inputs for a least-squares fit to
    calibrate the model's growth rate parameters. It defines parameters for
    initial cell counts (lnA0), global background growth (k_bg), and
    condition-specific growth effects (dk), then builds the corresponding
    design matrix `X` and response vector `y_obs`.

    Parameters
    ----------
    df : pandas.DataFrame
        A fully prepared dataframe containing all necessary columns for the
        calibration model, including 'ln_cfu', 'ln_cfu_std', 'genotype',
        'library', 'replicate', 't_pre', 't_sel', 'condition_pre',
        'condition_sel', 'titrant_conc', and 'theta'.

    Returns
    -------
    y_obs : numpy.ndarray
        A 1D array of shape (n_obs,) representing the response variable,
        which is the observed log(cfu).
    y_std : numpy.ndarray
        A 1D array of shape (n_obs,) holding the standard error for each
        observation in ``y_obs``.
    X : numpy.ndarray
        A 2D design matrix of shape (n_obs, n_params).
    param_df : pandas.DataFrame
        A dataframe defining each parameter's name, class, initial guess,
        and transformation properties.

    Notes
    -----
    A key step in this function is handling the condition parameters. The
    model uses the same `dk_b` and `dk_m` parameters for a given condition
    (e.g., 'kanR-kan') regardless of whether it appears in the 'pre' or 'sel'
    phase. To achieve this, the dataframe is internally reshaped into a "long"
    format where each growth phase is its own row.
    """
    # -------------------------------------------------------------------------
    # 1. Reshape data to handle shared condition parameters
    # -------------------------------------------------------------------------
    
    # Create a unique index for each original row to map back to later
    df['_original_row_idx'] = np.arange(len(df))

    # Create a "long" dataframe where each row is one growth phase ('pre' or 'sel')
    pre_df = df[['_original_row_idx', 'condition_pre', 't_pre', 'theta']].copy()
    pre_df.rename(columns={'condition_pre': 'condition', 't_pre': 't'}, inplace=True)
    
    sel_df = df[['_original_row_idx', 'condition_sel', 't_sel', 'theta']].copy()
    sel_df.rename(columns={'condition_sel': 'condition', 't_sel': 't'}, inplace=True)

    long_df = pd.concat([pre_df, sel_df], ignore_index=True)
    long_df = long_df[long_df['t'] > 0].copy() # Drop phases with no duration

    # -------------------------------------------------------------------------
    # 2. Define all parameter blocks using the helper function
    # -------------------------------------------------------------------------

    offset = 0
    # lnA0 parameters (one per unique experimental series)
    lnA0_idx, lnA0_df = _build_param_df(
        df=df, base_name="lnA0", series_selector=["genotype", "replicate"], #"library", "replicate"],
        guess_column="ln_cfu", transform="none", offset=offset
    )
    offset = lnA0_df["idx"].max() + 1

    # Global background growth parameters (k_bg_b and k_bg_m)
    # Use a dummy column to group all rows together.
    df['_global_selector'] = 1
    _, k_bg_df = _build_param_df(
        df=df, base_name="k_bg", series_selector=["_global_selector"],
        guess_column="none", transform="none", offset=offset # Guesses added manually
    )
    # Manually create the two k_bg parameters from the single group
    k_bg_b_df = k_bg_df.copy(); k_bg_b_df['name'] = 'k_bg_b'
    k_bg_m_df = k_bg_df.copy(); k_bg_m_df['name'] = 'k_bg_m'; k_bg_m_df['idx'] += 1
    k_bg_df = pd.concat([k_bg_b_df, k_bg_m_df], ignore_index=True)
    offset = k_bg_df["idx"].max() + 1
    
    # Condition-specific parameters (dk_b and dk_m)
    # Defined from the unified 'condition' column in the long dataframe
    dk_idx, dk_df = _build_param_df(
        df=long_df, base_name="dk", series_selector=["condition"],
        guess_column="none", transform="none", offset=offset # Guesses added manually
    )
    # Manually create b and m parameters for each condition
    dk_b_df = dk_df.copy(); dk_b_df['name'] += '_b'
    dk_m_df = dk_df.copy(); dk_m_df['name'] += '_m'; dk_m_df['idx'] += len(dk_df)
    dk_df = pd.concat([dk_b_df, dk_m_df], ignore_index=True)

    # Combine all parameter definitions
    param_df = pd.concat([lnA0_df, k_bg_df, dk_df], ignore_index=True)

    # -------------------------------------------------------------------------
    # 3. Construct the final design matrix X
    # -------------------------------------------------------------------------

    X = np.zeros((len(df), len(param_df)))

    # Populate lnA0 columns (predictor is 1)
    X[df['_original_row_idx'], lnA0_idx] = 1.0

    # Populate global k_bg columns
    k_bg_b_idx = k_bg_df[k_bg_df['name'] == 'k_bg_b']['idx'].iloc[0]
    k_bg_m_idx = k_bg_df[k_bg_df['name'] == 'k_bg_m']['idx'].iloc[0]
    total_time = (df['t_pre'] + df['t_sel']).to_numpy()
    X[:, k_bg_b_idx] = total_time
    X[:, k_bg_m_idx] = total_time * df['titrant_conc'].to_numpy()

    # Populate condition-specific dk columns using the long_df
    # This loop adds the predictor values for each growth phase back to the
    # appropriate row and parameter column in the main design matrix.
    dk_b_map = pd.Series(dk_b_df['idx'].values, index=dk_b_df['condition'])
    dk_m_map = pd.Series(dk_m_df['idx'].values, index=dk_m_df['condition'])
    
    long_df['dk_b_idx'] = long_df['condition'].map(dk_b_map)
    long_df['dk_m_idx'] = long_df['condition'].map(dk_m_map)
    
    # Vectorized update using np.add.at for efficiency
    np.add.at(X, (long_df['_original_row_idx'], long_df['dk_b_idx']), long_df['t'])
    np.add.at(X, (long_df['_original_row_idx'], long_df['dk_m_idx']), long_df['t'] * long_df['theta'])
    
    # -------------------------------------------------------------------------
    # 4. Construct response vectors y_obs and y_std
    # -------------------------------------------------------------------------
    
    y_obs = df["ln_cfu"].to_numpy()
    y_std = df["ln_cfu_std"].to_numpy()

    # Clean up temporary columns from the original df before returning
    df.drop(columns=['_original_row_idx', '_global_selector'], inplace=True)
    
    return y_obs, y_std, X, param_df

def _build_calibration_X(df):
    """
    Build a design matrix that allows fitting of k and dk parameters given known
    values for theta. 
    
    ln_cfu ~ ln_cfu_0 + pre_growth + sel_growth
    pre_growth: (k_bg_b + k_bg_m*titrant + dk_geno + dk_pre_b + dk_pre_m*theta)*t_pre
    sel_growth: (k_bg_b + k_bg_m*titrant + dk_geno + dk_sel_b + dk_sel_m*theta)*t_sel
    
    We are going to describe k_bg as a linear model in titrant_conc,
    accounting for any changes in growth rate of background cells due to change
    in titrant but not theta. If we distribute terms to isolate individual 
    variables, we end up with the following terms, which are shared among 
    experiments by the following scheme.
    
    1. ln_cfu_0  * (           1                ): genotype:replicate
    2. k_bg_b    * ( t_pre + t_sel              ): titrant_name
    3. k_bg_m    * ((t_pre + t_sel)*titrant_conc): titrant_name
    4. dk_geno   * ( t_pre + t_sel              ): genotype, ref = wt
    5. dk_cond_b * ( t_pre OR t_sel             ): condition, ref = background
    6. dk_cond_m * ( t_pre OR t_sel             ): condition, ref = background
    
    The OR on dk_cond_b comes about because a condition can either be a pre-
    or sel-condition. We have to take care of that with a wide-to-long move 
    when constructing X. 
    """

    # Grab y_obs and y_std from the wide dataframe.
    y_obs = df["ln_cfu"].to_numpy()
    y_std = df["ln_cfu_std"].to_numpy()
    
    # Make aggregate variables involving t_pre and t_sel for the linear regression
    # before the wide to long transform. 
    # genotype:replicate, t_pre + t_sel, (t_pre + t_sel)*titrant_conc
    df["geno_rep"] = list(zip(df["genotype"],df["replicate"]))
    df["t_tot"] = df["t_pre"] + df["t_sel"]
    df["t_tot_titr"] = df["t_tot"]*df["titrant_conc"]
    
    # The dataframe has t_pre, t_sel, condition_pre and condition_sel. Convert
    # to a long dataframe that doubles the number of rows. The pre and sel 
    # phases each get their own row with t_pre and t_sel -> t and 
    # condition_pre and condition_sel -> condition. 
    df_for_reshape = df.reset_index()
    df_long = pd.wide_to_long(
        df_for_reshape,
        stubnames=['t', 'condition'], # The prefixes of columns to melt
        i='index',                    # The identifier for each row
        j='phase',                    # The name for the new phase column
        sep='_',                      # The separator between stubname and phase
        suffix='(pre|sel)'            # The suffixes to look for
    ).reset_index()
    
    # Create aggregate t_theta variable 
    df_long["theta_t"] = df_long["theta"]*df_long["t"]
    
    model_terms = {}
    factor_terms = {}

    # Use a different ln_cfu0 for every genotype/replicate
    model_terms["ln_cfu_0"] = "C(geno_rep)"
    factor_terms["ln_cfu_0"] = ("genotype","replicate")
    
    # background growth vs titrant depends on titrant_name. The intercept depends
    # on total time. The slope depends on total time multipled by titrant conc. 
    model_terms["k_bg_b"] = "C(titrant_name):t_tot"
    factor_terms["k_bg_b"] = "titrant_name"
    
    model_terms["k_bg_m"] = "C(titrant_name):t_tot_titr"
    factor_terms["k_bg_m"] = "titrant_name"
    
    # The global effect of the genotype depends on genotype. Define wt parameter
    # as 0 with Treatment('wt') syntax. 
    model_terms["dk_geno"] = "C(genotype, Treatment('wt'))"
    factor_terms["dk_geno"] = "genotype"

    # The relationship between the known theta values and the growth rate 
    # perturbation. Note we set the 'background' treatment has have no effect
    # with the Treatment('background') syntax. 
    model_terms["dk_cond_b"] = "C(condition, Treatment('background')):t"
    factor_terms["dk_cond_b"] = "condition"
    
    model_terms["dk_cond_m"] = "C(condition, Treatment('background')):theta_t"
    factor_terms["dk_cond_m"] = "condition"

    # Build final formula. We're predicting ln_cfu with our model. Drop
    # default intercept (0)
    formula = " + ".join(model_terms.values())
    formula = f"ln_cfu ~ 0 + {formula}"

    _, X_long = patsy.dmatrices(formula,df_long,)

    param_names = X_long.design_info.column_names
    X_long_df = pd.DataFrame(X_long, columns=param_names)
    
    X_long_df['index'] = df_long['index']
    
    # Group by the original experiment and sum the contributions
    X_final_df = X_long_df.groupby('index').sum()

    # The sum just doubled the values for ln_cfu_0, k_bg*, and dk_geno. Divide
    # by two to get the values back. 
    cols_to_fix = [c for c in X_final_df.columns if 
                   "geno_rep" in c or 
                   c.endswith(':t_tot') or 
                   c.endswith(':t_tot_titr')]

    X_final_df[cols_to_fix] /= 2.0

    # Get the final design matrix as a numpy array, ready for fitting
    X = X_final_df.values

    # Get param names
    patsy_param_names = X_final_df.columns.to_numpy()

    # patsy is finicky with intercepts attached continuous variables. This 
    # drops columns that use [background] or [wt] -- both reference 
    # conditions that should be dropped. 
    good_mask = ~X_final_df.columns.str.contains(r"\[background\]|\[wt\]")

    # Clean up X and parameter names
    X = X[:,good_mask]
    patsy_param_names = patsy_param_names[good_mask]

    # Construct initial parameter dataframe with parameter names indexed
    # by parameter order, linked back to the factors of the initial 
    # dataframe. 
    param_df = parse_patsy(df,
                           patsy_param_names,
                           model_terms,
                           factor_terms)
    
    return y_obs, y_std, X, param_df 


def setup_calibration(df,
                      ln_cfu_0_guess=16,
                      k_bg_guess=0.02,
                      no_bg_slope=False):

    # Prepare dataframe, creating all necessary columns etc. 
    df = _prep_calibration_df(df)

    # Build the design matrix
    y_obs, y_std, X, param_df = _build_calibration_X(df)

    # Build guesses
    guesses = np.zeros(len(param_df))

    # ln_cfu_0
    ln_cfu_0_mask = param_df["param_class"] == "ln_cfu_0"
    guesses[ln_cfu_0_mask] = ln_cfu_0_guess

    # k_bg_b
    k_bg_b_mask = param_df["param_class"] == "k_bg_b"
    guesses[k_bg_b_mask] = k_bg_guess

    # Record the guesses
    param_df["guess"] = guesses
    
    param_df["fixed"] = False
    if no_bg_slope:
        param_df.loc[param_df["param_class"] == "k_bg_m","guess"] = 0.0
        param_df.loc[param_df["param_class"] == "k_bg_m","fixed"] = True

    # Build fit manager
    fm = FitManager(y_obs,y_std,X,param_df)

    return fm

def calibrate(df,
              output_file,
              ln_cfu_0_guess=16,
              k_bg_guess=0.02,
              no_bg_slope=False):
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
    ln_cfu_0_guess : float, optional
        An initial guess for the log-transformed initial population size (lnA0),
        by default 16.
    bg_k_guess : float, optional
        An initial guess for the constant term (c_0) of the background
        growth model, by default 0.025.
    no_bg_slope : bool, optional
        whether or not to fix the background slope to 0. defaults to False. 

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
    """

    # Construct a FitManager object to manage the fit. 
    fm = setup_calibration(df,
                           ln_cfu_0_guess=ln_cfu_0_guess,
                           k_bg_guess=k_bg_guess,
                           no_bg_slope=no_bg_slope)

    # Run least squares fit
    params, std_errors, cov_matrix, _ = run_least_squares(
        fm.predict_from_transformed,
        obs=fm.y_obs,
        obs_std=fm.y_std,
        guesses=fm.guesses_transformed,
        lower_bounds=fm.lower_bounds_transformed,
        upper_bounds=fm.upper_bounds_transformed
    )
    
    
    # Make predictions at each data point and store
    pred, pred_std = predict_with_error(fm.predict_from_transformed,
                                        params,
                                        cov_matrix)
    
    pred_df = pd.DataFrame({"y_obs":fm.y_obs,
                            "y_std":fm.y_std,
                            "calc_est":pred,
                            "calc_std":pred_std})

    pred_df = pd.concat([df,pred_df],axis=1)

    # Extract parameter estimates
    param_df = fm.param_df.copy()
    param_df["est"] = fm.back_transform(params)
    param_df["std"] = fm.back_transform_std_err(params,std_errors)

    # Fit a hill model to the observed theta values so we can calculate
    # approximate wildtype theta on the fly for any titrant conc
    theta_param = _fit_theta(df)
        
    # # Build output dictionary with fit results
    calibration_dict = {}

    # Extract k_bg vs. titrant slopes and interecepts
    calibration_dict["k_bg"] = {}
    tmp_df = param_df.loc[param_df["param_class"] == "k_bg_b",["titrant_name","est"]]
    calibration_dict["k_bg"]["b"] = tmp_df.set_index('titrant_name')['est'].to_dict()
    tmp_df = param_df.loc[param_df["param_class"] == "k_bg_m",["titrant_name","est"]]
    calibration_dict["k_bg"]["m"] = tmp_df.set_index('titrant_name')['est'].to_dict()

    # Extract dk_cond vs. theta slopes and interecepts
    calibration_dict["dk_cond"] = {}
    tmp_df = param_df.loc[param_df["param_class"] == "dk_cond_m",["condition","est"]] 
    calibration_dict["dk_cond"]["m"] = tmp_df.set_index('condition')['est'].to_dict()
    tmp_df = param_df.loc[param_df["param_class"] == "dk_cond_b",["condition","est"]] 
    calibration_dict["dk_cond"]["b"] = tmp_df.set_index('condition')['est'].to_dict()
    
    # Put background values (defined as as 0) into dk_cond
    calibration_dict["dk_cond"]["m"]["background"] = 0.0
    calibration_dict["dk_cond"]["b"]["background"] = 0.0

    # Write out theta fit
    calibration_dict["theta_param"] = theta_param

    write_calibration(calibration_dict=calibration_dict,
                    json_file=output_file)

    calibration_dict = read_calibration(output_file)


    return calibration_dict, pred_df, param_df

    
