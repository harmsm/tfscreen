
from tfscreen.calibration import (
    read_calibration,
    get_wt_theta
)

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error,
    FitManager,
)

from tfscreen.util.io import read_dataframe
from tfscreen.util.dataframe import (
    check_columns,
    get_scaled_cfu,
    chunk_by_group,
)

from tfscreen.genetics import (
    set_categorical_genotype
)

from tfscreen.analysis.independent.get_indiv_growth import get_indiv_growth
from tfscreen.util.design import build_param_df
from tfscreen.models.growth_linkage import get_model
from tfscreen.models.transition_linkage import get_transition_model
from tfscreen.models.occupancy_growth_model import OccupancyGrowthModel

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _prep_inference_df(df,
                       calibration_data,
                       max_batch_size,
                       model_name=None,
                       transition_model_name=None):
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
    
    # Check for model name, default to linear    # If not provided, get from calibration file
    if model_name is None:
        model_name = calibration_data.get("model_name", "linear")
    
    if transition_model_name is None:
        transition_model_name = calibration_data.get("transition_model_name", "constant")

    # Dilution from calibration
    dilution = calibration_data.get("dilution", 1.0)
    growth_model = get_model(model_name)
    growth_param_defs = growth_model.get_param_defs()
    suffixes = [p[0] for p in growth_param_defs]

    transition_model = get_transition_model(transition_model_name)
    transition_param_defs = transition_model.get_param_defs()
    suffixes.extend([p[0] for p in transition_param_defs])

    # Background slope and titrant vs titrant
    bg_df = calibration_data["k_bg_df"]
    
    # Background slope and titrant vs titrant
    bg_df = calibration_data["k_bg_df"]
    
    # Calculate k_bg for each row: k_bg = b + m * titrant_conc
    # bg_df is indexed by titrant_name
    k_bg_m_map = bg_df["m"]
    k_bg_b_map = bg_df["b"]
    
    k_bg_m = df["titrant_name"].map(k_bg_m_map).to_numpy()
    k_bg_b = df["titrant_name"].map(k_bg_b_map).to_numpy()
    
    # Compute total background growth rate
    df["k_bg"] = k_bg_b + k_bg_m * df["titrant_conc"]

    # Condition parameters
    # We need to map condition_pre/sel -> params

    # If "dk_cond" in dict is dict of dicts:
    dk_cond_data = calibration_data.get("dk_cond", {})
    
    # Helper to map params
    def map_params(condition_col, prefix):
        for suffix in suffixes:
            # Check if we have this suffix
            if suffix not in dk_cond_data:
                # Fallback for "linear" if using old calibration file that had "m" and "b" but no "model_name"
                # Old keys: "dk_cond_m", "dk_cond_b" in calibration_data?
                # or calibration_data["dk_cond"]["m"]?
                pass
            
            # Map values
            # dk_cond_data[suffix] is {condition: value}
            param_map = dk_cond_data[suffix]
            # Handle if map is Series (if read_calibration converted it)
            if hasattr(param_map, "get"):
                vals = df[condition_col].map(param_map)
            else:
                 # If it's a dataframe or something else
                 pass
            
            # If using old calibration data (linear), "dk_cond_df" might be present with "m" and "b".
            # The code I wrote in calibrate saved to "dk_cond" -> {suffix: {cond: val}}.
            
            df[f"{prefix}_{suffix}"] = vals

    # If new format exists
    if isinstance(dk_cond_data, dict) and len(dk_cond_data) > 0:
        per_titrant_tau = calibration_data.get("per_titrant_tau", False)
        for suffix in suffixes:
            param_map = dk_cond_data[suffix]
            
            if per_titrant_tau:

                pre_keys = df.apply(lambda r: f"{r['condition_pre']}:{r['titrant_name']}:0.0", axis=1)
                sel_keys = df.apply(lambda r: f"{r['condition_sel']}:{r['titrant_name']}:{r['titrant_conc']}", axis=1)
                
                # Optimized map with fallback
                def map_with_fallback(keys, cond_col):
                    # Try granular first
                    res = keys.map(param_map)
                    # For those that failed, try condition only
                    mask = res.isna()
                    if mask.any():
                        res.loc[mask] = df.loc[mask, cond_col].map(param_map)
                    return res

                df[f"dk_pre_{suffix}"] = map_with_fallback(pre_keys, "condition_pre")
                df[f"dk_sel_{suffix}"] = map_with_fallback(sel_keys, "condition_sel")
            else:
                df[f"dk_pre_{suffix}"] = df["condition_pre"].map(param_map)
                df[f"dk_sel_{suffix}"] = df["condition_sel"].map(param_map)
    else:
        # Fallback for old linear calibration files
        # They usually have dk_cond_df with m and b cols.
        dk_df = calibration_data["dk_cond_df"]
        df["dk_pre_m"] = dk_df.loc[df["condition_pre"],"m"].to_numpy()
        df["dk_pre_b"] = dk_df.loc[df["condition_pre"],"b"].to_numpy()
        df["dk_sel_m"] = dk_df.loc[df["condition_sel"],"m"].to_numpy()
        df["dk_sel_b"] = dk_df.loc[df["condition_sel"],"b"].to_numpy()
    return df, model_name, transition_model_name, dilution

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
    # have its own lnA0 that must be estimated. But for *growth rate* estimation,
    # we must fit each condition separately.
    lnA0_selector = ["genotype","library","replicate"]
    fit_series_selector = ["genotype","library","replicate", "titrant_name", "titrant_conc"]

    # Run individual fits on all genotype/library/replicate growth rates. 
    indiv_param_df, _ = get_indiv_growth(df,
                                         series_selector=fit_series_selector,
                                         calibration_data=calibration_data,
                                         dk_geno_selector=["genotype"],
                                         dk_geno_mask_col="_dk_geno_mask",
                                         lnA0_selector=lnA0_selector)

    # make sure the genotype column is preserved as a category (which preserves
    # it's pretty sorting). 
    indiv_param_df["genotype"] = indiv_param_df["genotype"].astype(df["genotype"].dtype)
    if "titrant_conc" in indiv_param_df.columns:
        indiv_param_df["titrant_conc"] = indiv_param_df["titrant_conc"].astype(float)
        
    # Merge estimates back into the main dataframe so we can use them to build
    # guesses for the final fit parameters.
    df = pd.merge(df, indiv_param_df,
                  on=fit_series_selector,
                  how="left")
    
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
    
    # Add fixed column, default to False
    final_df = final_df.copy()
    final_df["fixed"] = False

    return df_idx, final_df

    

class InferencePredictor:
    """Wrapper to handle non-linear inference predictions."""
    def __init__(self, growth_model, transition_model, num_growth_params, num_transition_params, dilution):
        self.growth_model = growth_model
        self.transition_model = transition_model
        self.num_growth_params = num_growth_params
        self.num_transition_params = num_transition_params
        self.dilution = dilution
        self._growth_model = OccupancyGrowthModel()

    def predict(self, params, X):
        """
        Predict ln_cfu (offset adjusted) based on parameters and design matrix X.
        
        X layout:
        0: lnA0_idx
        1: dk_geno_idx
        2: theta_idx
        3: t_total
        4: t_pre
        5: t_sel
        6: pre_start_idx
        7: sel_start_idx
        """
        # Unpack indices (cast to int for indexing)
        lnA0_idx = X[:, 0].astype(int)
        dk_geno_idx = X[:, 1].astype(int)
        theta_idx = X[:, 2].astype(int)
        dk_cond_pre_idx = X[:, 6].astype(int)
        dk_cond_sel_idx = X[:, 7].astype(int)
        
        # Unpack floats
        # t_total = X[:, 3]
        t_pre = X[:, 4]
        t_sel = X[:, 5]
        
        # 1. Base components
        # ln_cfu0
        ln_cfu0 = params[lnA0_idx]
        
        # Genotype effect: dk_geno
        dk_geno = params[dk_geno_idx]
        
        # Theta
        theta = params[theta_idx]
        
        # 2. Condition-specific growth and shift (tau, k_sharp)
        
        # Helper to extract (N, num_params) array of parameters
        def get_model_params(start_indices, num_params):
            # Create a 2D index array: row i contains [start[i], start[i]+1, ...]
            # specific_indices shape: (N, num_params)
            offsets = np.arange(num_params)
            specific_indices = start_indices[:, None] + offsets[None, :]
            return params[specific_indices]

        params_pre_all = get_model_params(dk_cond_pre_idx, self.num_growth_params + self.num_transition_params)
        params_sel_all = get_model_params(dk_cond_sel_idx, self.num_growth_params + self.num_transition_params)
        
        # Split into growth model params and transition model parameters
        params_pre_growth = params_pre_all[:, :self.num_growth_params]
        params_pre_trans = params_pre_all[:, self.num_growth_params:]
        
        params_sel_growth = params_sel_all[:, :self.num_growth_params]
        params_sel_trans = params_sel_all[:, self.num_growth_params:]
        
        # Predict growth rate perturbation from theta and model params
        # Note: Inference assumes theta_pre = 1.0 (maximal saturation/repression)
        # for non-selection conditions. 
        dk_pre = self.growth_model.predict(1.0, params_pre_growth.T) 
        dk_sel = self.growth_model.predict(theta, params_sel_growth.T) 
        
        # Transition parameters
        tau_sel = self.transition_model.predict_tau(theta, params_sel_trans.T)
        k_sel = self.transition_model.predict_k_sharp(theta, params_sel_trans.T)
        
        # Calculate full growth rates
        # Here we assume k_bg is 0 or rather already incorporated into b if needed?
        # Actually cfu_to_theta uses dk_b_sel and dk_m_sel from calibration. 
        # But here mu1 and mu2 are build from calibration background + model. 
        # Wait, how does cfu_to_theta handle background?
        # Looking at _prep_inference_df... it adds dk_b_pre, dk_m_pre etc. 
        
        # mu1 = k_bg_pre + dk_geno + dk_pre
        # but in cfu_to_theta, we don't have k_bg separately, it's usually handled by 
        # offset or by the caller. 
        
        # Re-reading calibrate.py CalibrationPredictor.predict:
        # mu1 = k_bg_pre + dk_geno + dk_pre
        # mu2 = k_bg_sel + dk_geno + dk_sel
        
        # For cfu_to_theta, the parameters in the X matrix (pre_start_idx, sel_start_idx)
        # point into the 'params' vector which contains BOTH the linkage model params
        # AND some background? No, usually they are just fixed constants from calibration.
        
        # Let's check _setup_inference in cfu_to_theta.py
        
        mu1 = dk_geno + dk_pre
        mu2 = dk_geno + dk_sel
        
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

def _setup_inference(df, logistic_theta=False, model_name="linear", transition_model_name="constant", dilution=1.0):
    """
    Assemble the final design matrix and response vector for regression.
    """
    
    # Get model definitions
    growth_model = get_model(model_name)
    growth_param_defs = growth_model.get_param_defs()
    num_growth_params = len(growth_param_defs)

    transition_model = get_transition_model(transition_model_name)
    transition_param_defs = transition_model.get_param_defs()
    num_transition_params = len(transition_param_defs)
    
    # Combine suffixes for parameter mapping
    suffixes = [p[0] for p in growth_param_defs]
    suffixes.extend([p[0] for p in transition_param_defs]) # Add transition model suffixes

    # -----------------------------------------------------------------------------
    # Infer column/value combination to parameter mapping. 

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

    # Fix dk_geno to 0.0 for wildtype
    wt_mask = dk_geno_df["name"] == "dk_geno_wt"
    if np.any(wt_mask):
        dk_geno_df.loc[wt_mask, "guess"] = 0.0
        dk_geno_df.loc[wt_mask, "fixed"] = True
    
    offset = dk_geno_df["idx"].max() + 1
    
    transform = "logistic" if logistic_theta else "none"

    theta_selector = ["genotype","titrant_name","titrant_conc"]
    theta_idx, theta_df = _build_param_df(
        df=df,
        base_name="theta",
        series_selector=theta_selector,
        guess_column="wt_theta",
        transform=transform,
        offset=offset
    )

    offset = theta_df["idx"].max() + 1

    # Parameters for condition-specific growth and transition models
    # These are fixed from calibration, but we need to create parameter entries
    # for them in the global parameter vector so they can be indexed by X.
    # We'll use a 'none' transform and their actual values as guesses.
    
    # Selector for condition parameters (e.g., m, b, tau, k_sharp for each condition)
    dk_cond_selector = ["condition_pre", "titrant_name", "titrant_conc"] # Use condition_pre for both pre/sel
    
    # Create a unique identifier for each set of condition parameters
    # We need to handle both pre and sel conditions, and their parameters might differ.
    # For simplicity, let's create a combined set of unique condition parameter vectors.
    
    # Collect all unique condition parameter vectors
    unique_cond_params = []
    for prefix in ["dk_pre", "dk_sel"]:
        cols_to_extract = [f"{prefix}_{s}" for s in suffixes]
        # Ensure all columns exist, fill with 0 if not (e.g., for old linear models)
        for col in cols_to_extract:
            if col not in df.columns:
                df[col] = 0.0 # Default to 0 if missing
        
        # Rename columns to just the suffix so they align when concatenating
        rename_dict = {f"{prefix}_{s}": s for s in suffixes}
        unique_cond_params.append(df[cols_to_extract].drop_duplicates().rename(columns=rename_dict))
    
    # Concatenate and find truly unique parameter vectors
    all_cond_params_df = pd.concat(unique_cond_params, ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # Assign a unique index to each unique parameter vector
    all_cond_params_df["idx"] = all_cond_params_df.index + offset
    all_cond_params_df["class"] = "dk_cond_params"
    all_cond_params_df["name"] = "dk_cond_params_" + all_cond_params_df.index.astype(str)
    all_cond_params_df["guess"] = 0.0 # Not directly used as they are fixed, but needed for structure
    all_cond_params_df["transform"] = "none"
    all_cond_params_df["scale_mu"] = 0
    all_cond_params_df["scale_sigma"] = 1
    
    # Now, map each row in df to the index of its corresponding condition parameters
    # This requires creating a temporary key for merging
    
    # For pre-selection conditions
    pre_cols = [f"dk_pre_{s}" for s in suffixes]
    df["_temp_key"] = df[pre_cols].astype(str).agg('_'.join, axis=1)
    
    all_cond_params_df_temp = all_cond_params_df[suffixes + ["idx"]].copy()
    all_cond_params_df_temp["_temp_key"] = all_cond_params_df_temp[suffixes].astype(str).agg('_'.join, axis=1)
    
    df = pd.merge(df, all_cond_params_df_temp[["_temp_key", "idx"]].rename(columns={"idx": "dk_cond_pre_idx"}),
                  on="_temp_key", how="left")
    df.drop(columns=["_temp_key"], inplace=True)

    # For selection conditions
    sel_cols = [f"dk_sel_{s}" for s in suffixes]
    df["_temp_key"] = df[sel_cols].astype(str).agg('_'.join, axis=1)
    
    all_cond_params_df_temp = all_cond_params_df[suffixes + ["idx"]].copy()
    all_cond_params_df_temp["_temp_key"] = all_cond_params_df_temp[suffixes].astype(str).agg('_'.join, axis=1)
    
    df = pd.merge(df, all_cond_params_df_temp[["_temp_key", "idx"]].rename(columns={"idx": "dk_cond_sel_idx"}),
                  on="_temp_key", how="left")
    df.drop(columns=["_temp_key"], inplace=True)

    # Now, we need to add the actual parameter values for each condition to the param_df.
    # Each row in all_cond_params_df represents a unique set of condition parameters.
    # We need to expand this into individual parameters in the param_df.
    
    dk_cond_param_dfs = []
    for _, row in all_cond_params_df.iterrows():
        base_idx = row["idx"]
        for i, suffix in enumerate(suffixes):
            param_name = f"dk_cond_params_{row.name}_{suffix}" # Unique name for each individual parameter
            param_value = row[suffix] 
            
            dk_cond_param_dfs.append(pd.DataFrame([{
                "class": "dk_cond_param_val",
                "name": param_name,
                "guess": param_value,
                "transform": "none",
                "scale_mu": 0,
                "scale_sigma": 1,
                "idx": base_idx + i # Each individual parameter gets an index relative to its base_idx
            }]))
    
    if dk_cond_param_dfs:
        dk_cond_params_final_df = pd.concat(dk_cond_param_dfs, ignore_index=True)
    else:
        dk_cond_params_final_df = pd.DataFrame(columns=["class","name","guess","transform","scale_mu","scale_sigma","idx"])

    # -----------------------------------------------------------------------------
    # build final parameter dataframe
    
    param_df = pd.concat([lnA0_df,dk_geno_df,theta_df, dk_cond_params_final_df],ignore_index=True)
    
    # -----------------------------------------------------------------------------
    # Construct design matrix
    
    # Cols: lnA0_idx(0), dk_geno_idx(1), theta_idx(2), t_total(3), t_pre(4), t_sel(5),
    #       dk_cond_pre_idx(6), dk_cond_sel_idx(7)
    num_cols = 8
    X = np.zeros((df.shape[0], num_cols), dtype=float)
    
    X[:, 0] = lnA0_idx
    X[:, 1] = dk_geno_idx
    X[:, 2] = theta_idx
    X[:, 3] = df["t_pre"] + df["t_sel"]
    X[:, 4] = df["t_pre"]
    X[:, 5] = df["t_sel"]
    X[:, 6] = df["dk_cond_pre_idx"]
    X[:, 7] = df["dk_cond_sel_idx"]

    # -----------------------------------------------------------------------------
    # build y_obs and y_std
    
    # Build final y_obs  (ln_cfu - constant terms)
    # k_bg is constant (parameters fixed).
    # t_total * k_bg
    y_offset = df["k_bg"] * (df["t_pre"] + df["t_sel"])
    y_obs = (df["ln_cfu"] - y_offset).to_numpy()

    # y_std is unchanged
    y_std = df["ln_cfu_std"].to_numpy()
        
    predictor = InferencePredictor(growth_model, transition_model, num_growth_params, num_transition_params, dilution)

    # -------------------------------------------------------------------------
    # 4. Final FitManager
    # -------------------------------------------------------------------------
    fm = FitManager(y_obs, y_std, X, param_df)
    fm.set_model_func(predictor.predict)
    
    return fm


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
    guesses = np.clip(fm.guesses_transformed, 
                      fm.lower_bounds_transformed,
                      fm.upper_bounds_transformed)
    

    params, std_errors, cov_matrix, fit = run_least_squares(
        fm.predict_from_transformed,
        obs=fm.y_obs,
        obs_std=fm.y_std,
        guesses=guesses,
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
    param_df = fm.param_df.copy()    # Extract parameters
    param_df["est"] = fm.back_transform(params)
    param_df["std"] = fm.back_transform_std_err(params,std_errors)

    
    # Add grouping columns to param_df. 
    # This assumes we want to parse the "name" column or rely on "class" + "idx"?
    # The original logic might have relied on "param_class".
    # Since we built this using build_param_df, it has "class" and "name".
    # We can perform any necessary clean up here. 
    
    return param_df, pred_df


def cfu_to_theta(df,
                 non_sel_conditions,
                 calibration_data,
                 max_batch_size=250,
                 logistic_theta=False,
                 model_name=None,
                 transition_model_name=None):
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
    logistic_theta : bool, optional
        Whether to use a "logistic" transform for the `theta` variable. If
        False (default), use "none".

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
    
    
    # Prepare calibration data
    calibration_data = read_calibration(calibration_data)
    
    # Prepare dataframe, creating all necessary columns etc. 
    # cfu_to_theta is the main entry point
    df = df.copy()
    
    # Enforce titrant_conc is float for proper grouping and math
    if "titrant_conc" in df.columns:
        try:
            df["titrant_conc"] = df["titrant_conc"].astype(float)
        except Exception:
            pass
        
    df, model_name, transition_model_name, dilution = _prep_inference_df(df,
                                        max_batch_size=max_batch_size,
                                        calibration_data=calibration_data,
                                        model_name=model_name,
                                        transition_model_name=transition_model_name)
    
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
        # Prepare fit for this batch
        fm = _setup_inference(
            batch_df, 
            logistic_theta=logistic_theta, 
            model_name=model_name,
            transition_model_name=transition_model_name,
            dilution=dilution
        )


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
