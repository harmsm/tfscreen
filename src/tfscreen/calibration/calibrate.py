
from tfscreen.models.generic import MODEL_LIBRARY

from tfscreen.calibration import (
    write_calibration,
    read_calibration
)

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error,
    FitManager
)

import numpy as np
import pandas as pd

import warnings

from tfscreen.util.io import read_dataframe
from tfscreen.util.dataframe import (
    check_columns,
    get_scaled_cfu
)
from tfscreen.util.design import build_param_df
from tfscreen.models.growth_linkage import get_model
from tfscreen.models.occupancy_growth_model import OccupancyGrowthModel

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
                        "theta_std",
                        "censored"]
    
    # Check for required columns
    check_columns(df,required_columns)

    # We do some regex-y matching with patsy. Nuke extra columns to avoid 
    # unexpected behavior. 
    df = df.loc[:,required_columns]

    return df


class CalibrationPredictor:
    """Wrapper to handle non-linear calibration predictions."""
    def __init__(self, model, num_model_params, dilution):
        self.model = model
        self.num_model_params = num_model_params
        self.dilution = dilution
        self._growth_model = OccupancyGrowthModel()

    def predict(self, params, X):
        """
        Predict ln_cfu based on parameters and design matrix X.
        
        X layout:
        0: lnA0_idx
        1: k_bg_b_idx
        2: k_bg_m_idx
        3: titrant_conc
        4: t_total (t_pre + t_sel)
        5: dk_geno_idx
        6: t_pre
        7: t_sel
        8: theta
        9: dk_cond_pre_start_idx
        10: dk_cond_sel_start_idx
        11: lag_idx (deprecated, now part of dk_cond block)
        """
        # Unpack indices (cast to int for indexing)
        lnA0_idx = X[:, 0].astype(int)
        k_bg_b_idx = X[:, 1].astype(int)
        k_bg_m_idx = X[:, 2].astype(int)
        dk_geno_idx = X[:, 5].astype(int)
        dk_cond_pre_idx = X[:, 9].astype(int)
        dk_cond_sel_idx = X[:, 10].astype(int)
        
        # Unpack floats
        titrant_conc = X[:, 3]
        # t_total = X[:, 4] # Unused now
        t_pre = X[:, 6]
        t_sel = X[:, 7]
        theta_sel = X[:, 8]
        theta_pre = X[:, 11]
        
        # 1. Base components
        # ln_cfu0
        ln_cfu0 = params[lnA0_idx]
        
        # Genotype effect: dk_geno
        dk_geno = params[dk_geno_idx]
        
        # Background growth: 
        # k_bg_sel uses selection titrant
        k_bg_sel = params[k_bg_b_idx] + params[k_bg_m_idx] * titrant_conc
        # k_bg_pre uses 0 titrant
        k_bg_pre = params[k_bg_b_idx]
        
        # 2. Condition-specific growth and shift (tau, k_sharp)
        # We need to gather parameters for the model. 
        # The last two parameters in each dk_cond block are now 'tau' and 'k_sharp'
        
        # Helper to extract (N, num_params) array of parameters
        def get_model_params(start_indices):
            # Create a 2D index array: row i contains [start[i], start[i]+1, ...]
            # specific_indices shape: (N, num_model_params + 2)
            offsets = np.arange(self.num_model_params + 2)
            specific_indices = start_indices[:, None] + offsets[None, :]
            return params[specific_indices]

        params_pre_all = get_model_params(dk_cond_pre_idx)
        params_sel_all = get_model_params(dk_cond_sel_idx)
        
        # Split into model params and shift parameters
        params_pre = params_pre_all[:, :-2]
        
        params_sel = params_sel_all[:, :-2]
        tau_sel = params_sel_all[:, -2]
        k_sel = params_sel_all[:, -1]
        
        # Predict growth rate perturbation from theta and model params
        # dk_pre uses theta at 0 titrant (theta_pre)
        # dk_sel uses selection theta
        dk_pre = self.model.predict(theta_pre, params_pre.T) 
        dk_sel = self.model.predict(theta_sel, params_sel.T) 
        
        # Calculate full growth rates
        mu1 = k_bg_pre + dk_geno + dk_pre
        mu2 = k_bg_sel + dk_geno + dk_sel
        
        # Use centralized model
        return self._growth_model.predict_trajectory(
            t_pre=t_pre,
            t_sel=t_sel,
            ln_cfu0=ln_cfu0,
            mu1=mu1,
            mu2=mu2,
            dilution=self.dilution,
            tau=tau_sel,
            k_sharp=k_sel
        )


def _build_calibration_data(df, model_name="linear", dilution=0.0196, per_titrant_tau=False):
    """
    Assemble the design matrix and response vector for calibration.
    """
    
    # Get model definition
    model = get_model(model_name)
    model_param_defs = model.get_param_defs()
    num_model_params = len(model_param_defs)
    
    # -------------------------------------------------------------------------
    # 1. Define all parameter blocks
    # -------------------------------------------------------------------------

    offset = 0
    
    # lnA0 parameters (one per unique experimental series)
    lnA0_idx, lnA0_df = build_param_df(
        df=df, base_name="ln_cfu_0", series_selector=["genotype", "replicate", "condition_pre"],
        guess_column="ln_cfu", transform="none", offset=offset
    )
    offset = lnA0_df["idx"].max() + 1

    # Global background growth parameters (k_bg_b and k_bg_m)
    # k_bg = b + m * titrant_conc
    # fitted per titrant
    k_bg_b_idx, k_bg_b_df = build_param_df(
        df=df, base_name="k_bg_b", series_selector=["titrant_name"],
        guess_column="none", transform="none", offset=offset 
    )
    # Set guess for b
    k_bg_b_df["guess"] = 0.02
    
    offset = k_bg_b_df["idx"].max() + 1
    
    k_bg_m_idx, k_bg_m_df = build_param_df(
        df=df, base_name="k_bg_m", series_selector=["titrant_name"],
        guess_column="none", transform="none", offset=offset 
    )
    # Set guess for m
    k_bg_m_df["guess"] = 0.0
    
    k_bg_df = pd.concat([k_bg_b_df, k_bg_m_df], ignore_index=True)
    
    offset = k_bg_df["idx"].max() + 1
    
    # Genotype effect parameters (dk_geno)
    dk_geno_idx, dk_geno_df = build_param_df(
        df=df, base_name="dk_geno", series_selector=["genotype"],
        guess_column="none", transform="none", offset=offset
    )
    
    # Fix 'wt' genotype parameter to 0 for identifiability
    # This prevents collinearity between global background growth and genotype effect
    dk_geno_df.loc[dk_geno_df["name"] == "dk_geno_wt", "fixed"] = True
    dk_geno_df.loc[dk_geno_df["name"] == "dk_geno_wt", "guess"] = 0.0
    
    offset = dk_geno_df["idx"].max() + 1
    
    # Condition-specific parameters (dk_cond)
    # We need to generate N parameters per condition, where N = num_model_params
    
    if per_titrant_tau:
        # Group by condition, titrant_name, titrant_conc
        # We only care about condition_sel here, as that's what we usually fit
        # specifically if we want per-concentration shifts.
        # But we need to handle condition_pre as well if it's different.
        # Actually, for calibration, condition_pre is usually "background".
        
        # Let's find all unique (condition, titrant_name, titrant_conc) combos.
        # For condition_pre, titrant is usually 0, so it will just be "cond:titrant:0"
        pre_groups = df.apply(lambda r: f"{r['condition_pre']}:{r['titrant_name']}:0.0", axis=1)
        sel_groups = df.apply(lambda r: f"{r['condition_sel']}:{r['titrant_name']}:{r['titrant_conc']}", axis=1)
        unique_groups = pd.concat([pre_groups, sel_groups]).unique()
        unique_groups = np.sort(unique_groups)
        
        group_to_idx_col = "dk_group"
        df["dk_group_pre"] = pre_groups
        df["dk_group_sel"] = sel_groups
    else:
        unique_groups = pd.concat([df['condition_pre'], df['condition_sel']]).unique()
        unique_groups = np.sort(unique_groups)
        group_to_idx_col = "condition"

    dk_cond_rows = []
    current_idx = offset
    
    group_to_start_idx = {}
    
    for group in unique_groups:
        group_to_start_idx[group] = current_idx
        
        # Is this background?
        # If per_titrant_tau, group is "cond:titrant:conc"
        if per_titrant_tau:
            is_bg = group.startswith("background:")
        else:
            is_bg = (group == "background")
        
        for suffix, guess, transform, desc in model_param_defs:
            row = {
                "class": f"dk_cond",
                "name": f"dk_cond_{group}_{suffix}",
                "guess": 0.0 if is_bg else guess, # Force background guesses to 0
                "transform": transform,
                "scale_mu": 0,
                "scale_sigma": 1,
                "idx": current_idx,
                "fixed": is_bg # Fix background parameters to 0
            }
            dk_cond_rows.append(row)
            current_idx += 1
            
        # Add shift parameters (tau, k_sharp) for this condition
        row = {
            "class": "dk_cond",
            "name": f"dk_cond_{group}_tau",
            "guess": 0.0,
            "transform": "none",
            "scale_mu": 0,
            "scale_sigma": 1,
            "idx": current_idx,
            "fixed": is_bg
        }
        dk_cond_rows.append(row)
        current_idx += 1

        row = {
            "class": "dk_cond",
            "name": f"dk_cond_{group}_k_sharp",
            "guess": 1.0,
            "transform": "none",
            "scale_mu": 0,
            "scale_sigma": 1,
            "idx": current_idx,
            "fixed": is_bg
        }
        dk_cond_rows.append(row)
        current_idx += 1
            
    dk_cond_df = pd.DataFrame(dk_cond_rows)
    
    # Combine all parameter definitions
    param_df = pd.concat([lnA0_df, k_bg_df, dk_geno_df, dk_cond_df], ignore_index=True)
    
    # Ensure 'fixed' column is boolean and No NaNs (concatenation can introduce them)
    if "fixed" not in param_df.columns:
        param_df["fixed"] = False
    param_df.loc[pd.isna(param_df["fixed"]), "fixed"] = False
    param_df["fixed"] = param_df["fixed"].astype(bool)

    # -------------------------------------------------------------------------
    # 2. Construct the data matrix X
    # -------------------------------------------------------------------------

    # Initialize X with zeros
    # Columns:
    # 0: lnA0_idx, 1: k_bg_b, 2: k_bg_m, 3: titr, 4: t_tot, 5: dk_geno, 
    # 6: t_pre, 7: t_sel, 8: theta, 9: pre_idx, 10: sel_idx, 11: theta_pre
    X = np.zeros((len(df), 12))
    
    # Calculate theta_pre (baseline theta for each titrant)
    theta_param = _fit_theta(df)
    theta_pre_map = {k: v[0] for k, v in theta_param.items()}
    
    X[:, 0] = lnA0_idx
    X[:, 1] = k_bg_b_idx
    X[:, 2] = k_bg_m_idx
    X[:, 3] = df['titrant_conc']
    X[:, 4] = df['t_pre'] + df['t_sel']
    X[:, 5] = dk_geno_idx
    X[:, 6] = df['t_pre']
    X[:, 7] = df['t_sel']
    X[:, 8] = df['theta']
    X[:, 9] = df["dk_group_pre" if per_titrant_tau else "condition_pre"].map(group_to_start_idx).astype(int)
    X[:, 10] = df["dk_group_sel" if per_titrant_tau else "condition_sel"].map(group_to_start_idx).astype(int)
    X[:, 11] = df['titrant_name'].map(theta_pre_map).fillna(1.0).to_numpy() # Default to 1.0 if not found
    
    # -------------------------------------------------------------------------
    # 3. Construct response vectors y_obs and y_std
    # -------------------------------------------------------------------------
    
    y_obs = df["ln_cfu"].to_numpy().copy()
    y_std = df["ln_cfu_std"].to_numpy()

    # Clean up temporary columns from the original df before returning
    df.drop(columns=['_global_selector', 'dk_group_pre', 'dk_group_sel'], inplace=True, errors='ignore')
    
    predictor = CalibrationPredictor(model, num_model_params, dilution)
    
    return y_obs, y_std, X, param_df, predictor


def setup_calibration(df,
                      ln_cfu_0_guess=16,
                      k_bg_guess=0.02,
                      no_bg_slope=False,
                      dilution=0.0196,
                      model_name="linear",
                      per_titrant_tau=False):
    """
    Prepare the calibration fit.

    Parameters
    ----------
    df : pandas.DataFrame or str
        Raw calibration data or path to a data file.
    ln_cfu_0_guess : float, optional
        Initial guess for log initial population, by default 16.
    k_bg_guess : float, optional
        Initial guess for background growth rate, by default 0.02.
    no_bg_slope : bool, optional
        Fix the background slope to 0, by default False.
    dilution : float, optional
        Dilution factor between pre-growth and selection
        (e.g. 200 µL into 10 mL = 0.0196). Default is 0.0196.
    model_name : str, optional
        Name of the growth linkage model to use ('linear', 'power_law', 'saturation'),
        by default "linear".

    Returns
    -------
    FitManager
        Configured fit manager ready for least-squares optimization.
    """

    # Prepare dataframe, creating all necessary columns etc. 
    df = _prep_calibration_df(df)

    # Build the design matrix
    y_obs, y_std, X, param_df, predictor = _build_calibration_data(
        df, 
        model_name=model_name,
        dilution=dilution,
        per_titrant_tau=per_titrant_tau
    )

    # Build guesses
    guesses = np.zeros(len(param_df))
    
    # We populate guesses from param_df if they exist (which they do from build_param_df)
    guesses = param_df["guess"].to_numpy().copy()

    # Apply overrides for ln_cfu_0 and k_bg_b if simple guesses are provided 
    # and not already set by more specific logic (though _build_calibration_data uses 
    # calibration guesses logic).
    
    # ln_cfu_0
    ln_cfu_0_mask = param_df["class"] == "ln_cfu_0"
    if ln_cfu_0_guess is not None:
        guesses[ln_cfu_0_mask] = ln_cfu_0_guess

    # k_bg_b
    k_bg_b_mask = param_df["class"] == "k_bg_b"
    if k_bg_guess is not None:
        guesses[k_bg_b_mask] = k_bg_guess

    # Record the guesses
    param_df["guess"] = guesses
    
    # Handle fixed background slope
    if no_bg_slope:
        param_df.loc[param_df["class"] == "k_bg_m","guess"] = 0.0
        param_df.loc[param_df["class"] == "k_bg_m","fixed"] = True

    # Build fit manager
    fm = FitManager(y_obs,y_std,X,param_df)
    
    # Set the prediction model
    fm.set_model_func(predictor.predict)

    return fm

def calibrate(df,
              output_file,
              ln_cfu_0_guess=16,
              k_bg_guess=0.02,
              no_bg_slope=False,
              dilution=0.0196,
              model_name="linear",
              per_titrant_tau=False):
    """
    Run the full calibration workflow on experimental data.

    Takes experimental data, performs validation, runs a global non-linear
    fit to determine growth parameters (`b` and `m` vs. theta for each
    titrant), fits a Hill model to the `theta` vs. titrant data, and saves
    the resulting calibration parameters to a JSON file.

    Parameters
    ----------
    df : pandas.DataFrame or str
        A pre-loaded DataFrame or a file path (e.g., to an Excel file)
        containing the compiled experimental data.
    output_file : str
        The file path where the output JSON calibration data will be
        written.
    ln_cfu_0_guess : float, optional
        An initial guess for the log-transformed initial population size
        (lnA0), by default 16.
    k_bg_guess : float, optional
        An initial guess for the constant term of the background growth
        model, by default 0.02.
    no_bg_slope : bool, optional
        Whether or not to fix the background slope to 0. Default is False.
    dilution : float, optional
        Dilution factor between pre-growth and selection
        (e.g. 200 µL into 10 mL = 0.0196). Default is 0.0196.
    model_name : str, optional
        Name of the growth linkage model to use ('linear', 'power_law', 'saturation'),
        by default "linear".

    Returns
    -------
    dict
        A dictionary containing the final calibration parameters.
    pandas.DataFrame
        A dataframe holding the predicted and observed ln(A) for all data
        points used for the calibration.
    pandas.DataFrame
        A dataframe holding the fit ln(A0) for all replicates used for the
        calibration.
    """

    # Construct a FitManager object to manage the fit. 
    fm = setup_calibration(df,
                           ln_cfu_0_guess=ln_cfu_0_guess,
                           k_bg_guess=k_bg_guess,
                           no_bg_slope=no_bg_slope,
                           dilution=dilution,
                           model_name=model_name,
                           per_titrant_tau=per_titrant_tau)

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

    # Avoid duplicate columns during concat by dropping them from the original df
    # if they already exist.
    to_drop = ["y_obs", "y_std", "calc_est", "calc_std"]
    df_clean = df.drop(columns=[c for c in to_drop if c in df.columns])
    pred_df = pd.concat([df_clean,pred_df],axis=1)

    # Extract parameter estimates
    param_df = fm.param_df.copy()
    param_df["est"] = fm.back_transform(params)
    param_df["std"] = fm.back_transform_std_err(params,std_errors)

    # -------------------------------------------------------------------------
    # Add -t_pre points to pred_df for plotting. These points have y_obs = NaN
    # and calc_est = ln_cfu_0. 
    
    # Grab unique series
    series_cols = ["genotype","replicate","condition_pre","condition_sel","titrant_name","titrant_conc"]
    series_metadata = pred_df.groupby(series_cols,observed=True).min(numeric_only=False).reset_index()

    # Create new rows at -t_pre
    new_rows = series_metadata.copy()
    new_rows["t_sel"] = -new_rows["t_pre"]
    new_rows["y_obs"] = np.nan
    new_rows["y_std"] = np.nan
    
    # Map ln_cfu_0 from param_df to these new rows
    lnA0_map_df = param_df.loc[param_df["class"] == "ln_cfu_0"].copy()
    lnA0_map_df["genotype"] = lnA0_map_df["genotype"].astype(str)
    lnA0_map_df["replicate"] = lnA0_map_df["replicate"].astype(str)
    lnA0_map_df["condition_pre"] = lnA0_map_df["condition_pre"].astype(str)
    
    lnA0_map = lnA0_map_df.set_index(["genotype","replicate","condition_pre"])["est"]
    lnA0_std_map = lnA0_map_df.set_index(["genotype","replicate","condition_pre"])["std"]
    
    # Get estimates and stds
    lookup_df = new_rows[["genotype","replicate","condition_pre"]].astype(str)
    lookup_idx = pd.MultiIndex.from_frame(lookup_df)
    
    new_rows["calc_est"] = lnA0_map.reindex(lookup_idx).values
    new_rows["calc_std"] = lnA0_std_map.reindex(lookup_idx).values
    
    # Append new rows and sort
    pred_df = pd.concat([pred_df,new_rows],ignore_index=True)
    pred_df = pred_df.sort_values(series_cols + ["t_sel"])

    # Fit a hill model to the observed theta values so we can calculate
    # approximate wildtype theta on the fly for any titrant conc
    theta_param = _fit_theta(df)
        
    # # Build output dictionary with fit results
    calibration_dict = {}

    # Extract ln_cfu_0
    tmp_df = param_df.loc[param_df["class"] == "ln_cfu_0",["genotype","replicate","condition_pre","est"]]
    # Create a colon-separated index string: genotype:replicate:condition_pre
    index_str = tmp_df["genotype"].astype(str) + ":" + \
                tmp_df["replicate"].astype(str) + ":" + \
                tmp_df["condition_pre"].astype(str)
    calibration_dict["ln_cfu_0"] = pd.Series(tmp_df["est"].values, index=index_str).to_dict()

    # Extract k_bg vs. titrant slopes and interecepts
    calibration_dict["k_bg"] = {}
    tmp_df = param_df.loc[param_df["class"] == "k_bg_b",["titrant_name","est"]]
    calibration_dict["k_bg"]["b"] = tmp_df.set_index('titrant_name')['est'].to_dict()
    tmp_df = param_df.loc[param_df["class"] == "k_bg_m",["titrant_name","est"]]
    calibration_dict["k_bg"]["m"] = tmp_df.set_index('titrant_name')['est'].to_dict()

    # Extract dk_cond vs. theta parameters
    # The structure depends on the model.
    # We will just dump all dk_cond parameters into a sub-dictionary.
    calibration_dict["dk_cond"] = {}
    calibration_dict["dk_cond_params"] = {} # Store raw parameter values by condition
    
    # Get the suffix list from the model
    # (re-get model to be safe, though fm has predictor.model)
    # predictor is hidden in fm._model_func closure/class, but we can assume model_name
    model = get_model(model_name)
    suffixes = [p[0] for p in model.get_param_defs()]
    suffixes.extend(["tau", "k_sharp"])
    
    # Iterate over unique conditions in param_df
    # dk_cond parameters are named like "dk_cond_{condition}_{suffix}"
    
    # Extract all dk_cond parameters
    dk_cond_df = param_df[param_df["class"] == "dk_cond"].copy()
    
    # We need to map back to condition. 
    # Name format: dk_cond_{condition}_{suffix}
    # This is slightly fragile if condition has underscores and matches suffix. 
    # But suffix is from fixed set.
    
    # Safer way: earlier construction used name = f"dk_cond_{cond}_{suffix}"
    # regex extract?
    
    for suffix in suffixes:
        # Create a dictionary for this parameter type (e.g. "m", "b", "n", "min", "max") 
        calibration_dict["dk_cond"][suffix] = {}
        
        # Filter for names ending in _{suffix}
        mask = dk_cond_df["name"].str.endswith(f"_{suffix}")
        sub_df = dk_cond_df[mask]
        
        for _, row in sub_df.iterrows():
            # extract condition from name: dk_cond_{cond}_{suffix}
            # Remove prefix and suffix
            cond = row["name"][len("dk_cond_") : -len(f"_{suffix}")]
            calibration_dict["dk_cond"][suffix][cond] = row["est"]
            
        # If we have granular keys, also store a "base" condition average for
        # fallback/plotting (e.g., in summary plots).
        if per_titrant_tau:
            bases = {}
            for k, v in calibration_dict["dk_cond"][suffix].items():
                if ":" in k:
                    b = k.split(":")[0]
                    if b not in bases: bases[b] = []
                    bases[b].append(v)
            
            for b, vals in bases.items():
                if b not in calibration_dict["dk_cond"][suffix]:
                    calibration_dict["dk_cond"][suffix][b] = np.mean(vals)

        # Ensure background is 0 (it was fixed, so est should be 0/guess)
        calibration_dict["dk_cond"][suffix]["background"] = 0.0

    # Extract dk_geno
    calibration_dict["dk_geno"] = {}
    tmp_df = param_df.loc[param_df["class"] == "dk_geno", ["name", "est"]]
    for i, r in tmp_df.iterrows():
        # name is dk_geno_{genotype}
        if r["name"].startswith("dk_geno_"):
            geno = r["name"][8:] 
            calibration_dict["dk_geno"][geno] = r["est"]

    calibration_dict["model_name"] = model_name
    calibration_dict["per_titrant_tau"] = per_titrant_tau

    # Write out theta fit
    calibration_dict["theta_param"] = theta_param
    calibration_dict["dilution"] = dilution

    write_calibration(calibration_dict=calibration_dict,
                    json_file=output_file)

    calibration_dict = read_calibration(output_file)


    return calibration_dict, pred_df, param_df

    
