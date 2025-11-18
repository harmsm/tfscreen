import pandas as pd
from tfscreen.genetics import set_categorical_genotype
from typing import Optional

def _parse_parameter(param_df: pd.DataFrame,
                     ground_truth_df: Optional[pd.DataFrame],
                     param_class: str,
                     split_cols_to_drop: list[int],
                     rename_map: dict[int, str],
                     output_prefix: str,
                     sort_keys: list[str],
                     ground_truth_keys: list[str],
                     ground_truth_col_name: str,
                     type_conversions: Optional[dict[str, type]] = None):
    """
    A general utility to parse a class of parameters from a tidy dataframe.
    
    This function handles the common logic for parsing parameter estimates
    by splitting their names, renaming columns, and optionally mapping
    ground truth values.
    """
    # 1. Filter for the specific parameter class and parse the name
    params = param_df[param_df["class"] == param_class].copy()
    
    # Handle case where no parameters of this class exist
    if params.empty:
        return params

    expanded_names = (params["name"]
                      .str.split("_", expand=True)
                      .drop(columns=split_cols_to_drop)
                      .rename(columns=rename_map)
                     )
    df = pd.concat([expanded_names, params], axis=1)

    # 2. Apply any required type conversions
    if type_conversions:
        for col, new_type in type_conversions.items():
            if col in df.columns:
                df[col] = df[col].astype(new_type)

    # 3. Reorganize columns, renaming 'est' and 'std'
    to_get = df[["est", "std"]]
    df = df.drop(columns=["est", "std"])
    insert_idx = df.columns.get_loc("class")
    df.insert(loc=insert_idx, column=f"{output_prefix}_std", value=to_get["std"])
    df.insert(loc=insert_idx, column=f"{output_prefix}_est", value=to_get["est"])

    # 4. Sort and set genotype category
    df = (set_categorical_genotype(df, sort=True)
          .sort_values(sort_keys)
          .reset_index(drop=True))

    # 5. Map ground truth values, if provided
    if ground_truth_df is not None:
        gt_df = set_categorical_genotype(ground_truth_df, sort=True)
        
        # Get one ground truth value for each key combination
        unique_gt = gt_df.groupby(ground_truth_keys, observed=True).agg("first").reset_index()
        
        # Create a mapping from keys -> ground truth value
        mapper = unique_gt.set_index(ground_truth_keys)[ground_truth_col_name]
        
        # Use the map to look up the real value for each row in our parsed df
        real_values = df.set_index(ground_truth_keys).index.map(mapper)
        
        # Insert the new column
        insert_loc = df.columns.get_loc(f'{output_prefix}_std') + 1
        df.insert(loc=insert_loc, column=f"{output_prefix}_real", value=real_values)

    return df


def _parse_theta(param_df, ground_truth_df=None):
    """Parse theta parameters and optionally map to ground truth."""
    return _parse_parameter(
        param_df=param_df,
        ground_truth_df=ground_truth_df,
        param_class="theta",
        split_cols_to_drop=[0],
        rename_map={1: "genotype", 2: "titrant_name", 3: "titrant_conc"},
        output_prefix="theta",
        sort_keys=["genotype", "titrant_name", "titrant_conc"],
        ground_truth_keys=["genotype", "titrant_name", "titrant_conc"],
        ground_truth_col_name="theta",
        type_conversions={"titrant_conc": float}
    )

def _parse_ln_cfu0(param_df, ground_truth_df=None):
    """Parse ln_cfu0 parameters and optionally map to ground truth."""
    return _parse_parameter(
        param_df=param_df,
        ground_truth_df=ground_truth_df,
        param_class="lnA0",
        split_cols_to_drop=[0],
        rename_map={1: "genotype", 2: "library", 3: "replicate"},
        output_prefix="ln_cfu0",
        sort_keys=["genotype", "library", "replicate"],
        ground_truth_keys=["genotype", "library", "replicate"],
        ground_truth_col_name="ln_cfu_0",
        type_conversions={"replicate": str}
    )

def _parse_dk_geno(param_df, ground_truth_df=None):
    """Parse dk_geno parameters and optionally map to ground truth."""
    return _parse_parameter(
        param_df=param_df,
        ground_truth_df=ground_truth_df,
        param_class="dk_geno",
        split_cols_to_drop=[0, 1],
        rename_map={2: "genotype"},
        output_prefix="dk_geno",
        sort_keys=["genotype"],
        ground_truth_keys=["genotype"],
        ground_truth_col_name="dk_geno"
    )

def _build_ground_truth(counts_df,sample_df):
    """
    Build a ground_truth dataframe from counts_df and sample_df from a 
    tfscreen simulation. In most cases, the output dataframe will hold 
    ln_cfu_0, dk_geno, theta, k_pre, and k_sel. 
    """

    #  Drop cfu/mL part of sample_df
    sample_df = sample_df.copy()
    if "cfu_per_mL" in sample_df.columns:
        drop_at = sample_df.columns.get_loc("cfu_per_mL")
        sample_df = sample_df[sample_df.columns[:drop_at]]

    # Move sample from a column to the index
    sample_df = sample_df.set_index("sample")

    # Look up meta data using counts_df["sample"] column
    meta_data = sample_df.loc[counts_df["sample"],:]

    # concatenate meta data with real data
    ground_truth_df = pd.concat([meta_data.reset_index(drop=True),
                                 counts_df.reset_index(drop=True)],axis=1)

    # Clean up and return 
    ground_truth_df = (set_categorical_genotype(ground_truth_df)
                       .sort_values(["genotype","sample"])
                       .reset_index(drop=True))

    # Set column types and names (for look up later)
    ground_truth_df["replicate"] = ground_truth_df["replicate"].astype(str)
    ground_truth_df["titrant_conc"] = ground_truth_df["titrant_conc"].astype(float)
    
    return ground_truth_df


def _clean_pred_df(pred_df):
    """
    Clean up the columns in pred_df and put in a standard order.
    """

    pred_df = set_categorical_genotype(pred_df)
    to_keep = ["genotype","replicate","library",
               "titrant_name","titrant_conc",
               "condition_pre","t_pre","condition_sel","t_sel",
               "y_obs","y_std",
               "calc_est","calc_std"]
    columns = [k for k in to_keep if k in  pred_df.columns]
    pred_df = pred_df.loc[:,columns].sort_values(columns).reset_index(drop=True)

    return pred_df

def process_theta_fit(
    param_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    counts_df: Optional[pd.DataFrame] = None,
    sample_df: Optional[pd.DataFrame] = None
) -> dict[str, pd.DataFrame]:
    """Process parameter and prediction dataframes into human-readable formats.

    This function serves as the main entry point for post-processing the
    outputs of a model fit. It parses different parameter classes (`theta`,
    `lnA0`, `dk_geno`), cleans the prediction data, and optionally merges in
    "ground truth" values from simulation or experimental data if the
    relevant dataframes are provided.

    Parameters
    ----------
    param_df : pandas.DataFrame
        A tidy DataFrame of parameter estimates from a model fit. Must contain
        the columns "name", "class", "est", and "std".
    pred_df : pandas.DataFrame
        A tidy DataFrame of model predictions. Expected to have columns like
        "y_obs", "y_std", "calc_est", etc.
    counts_df : pandas.DataFrame, optional
        The raw counts data from a simulation or experiment. If provided along
        with `sample_df`, it is used to generate ground truth values for
        comparison with estimated parameters.
    sample_df : pandas.DataFrame, optional
        The sample metadata corresponding to `counts_df`. Required if
        `counts_df` is provided for ground truth generation.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the processed dataframes. The keys are:
        - "theta": DataFrame of parsed theta parameters.
        - "dk_geno": DataFrame of parsed dk_geno parameters.
        - "ln_cfu0": DataFrame of parsed ln_cfu0 (lnA0) parameters.
        - "pred": Cleaned and sorted DataFrame of model predictions.
    """

    ground_truth_dict = {"theta":None,
                         "dk_geno":None,
                         "ln_cfu0":None}

    # If enough information is provided, build a ground_truth_df
    if (counts_df is not None) and (sample_df is not None):
        
        ground_truth_df = _build_ground_truth(counts_df,
                                              sample_df)

        # Check for various ground_truths that could be in the data. If present,
        # record that they should be used
        
        if "theta" in ground_truth_df.columns:
            ground_truth_dict["theta"] = ground_truth_df
        
        if "dk_geno" in ground_truth_df.columns:
            ground_truth_dict["dk_geno"] = ground_truth_df

        if "ln_cfu_0" in ground_truth_df.columns:
            ground_truth_dict["ln_cfu0"] = ground_truth_df

    theta_df = _parse_theta(param_df,ground_truth_dict["theta"])
    dk_geno_df = _parse_dk_geno(param_df,ground_truth_dict["dk_geno"])
    ln_cfu0_df = _parse_ln_cfu0(param_df,ground_truth_dict["ln_cfu0"])

    pred_df = _clean_pred_df(pred_df)

    return {"theta":theta_df,
            "dk_geno":dk_geno_df,
            "ln_cfu0":ln_cfu0_df,
            "pred":pred_df}
