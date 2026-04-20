
import numpy as np
import pandas as pd

def build_param_df(df,
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
    if guess_column == "none":
        param_df["guess"] = 0.0
    else:
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
    cols_to_keep = ["class","name","guess",
                    "transform","scale_mu","scale_sigma","idx"]
    # Add grouping columns so downstream code can use them
    cols_to_keep.extend(series_selector)
    
    # Ensure we don't have duplicates if series_selector includes any of the above
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    
    final_df = param_df[cols_to_keep]

    return df_idx, final_df
