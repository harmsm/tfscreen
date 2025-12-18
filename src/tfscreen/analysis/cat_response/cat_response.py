
from .cat_fit import cat_fit
from tfscreen.models.generic import MODEL_LIBRARY

from tfscreen.util.numerical import xfill

import pandas as pd

def cat_response(df,
                 x_column="titrant_conc",
                 y_column="theta_est",
                 y_std_column="theta_std",
                 models_to_run=None,
                 verbose=False):
    """
    Processes a DataFrame of genotype data, running fits for each genotype
    and aggregating the results into summary and model-specific DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the experimental data. Must include columns:
        'genotype', which it uses to break up individual samples to fit. 
    x_column, y_column, y_std_columns : str, optional
        names of columns in dataframe corresponding to x (independent variable),
        y (measured value), and y_std (standard error on the measured value). 
    models_to_run : list of str, optional
        A list of model names to test. If None (default), all models in the
        global MODEL_LIBRARY will be tested.
    verbose : bool, optional
        If True, prints warnings to the console when a model fails to fit.
        Defaults to False.

    Returns
    -------
    tuple of (dict, pd.DataFrame)
        - model_dataframes: A dictionary where keys are model names and values
          are DataFrames. Each DataFrame contains all genotypes as rows, with
          columns for that model's parameter estimates, standard errors, and
          overall fit statistics (is_best_model, R2, AIC_weight).
        - summary_dataframe: A single DataFrame where each row is a genotype.
          Columns include the best model name, its R2 and AIC weight, the
          overall fit status, and the AIC weights for all tested models.
        - pred_df : a single DataFrame where each row is a genotype/model 
          combination holding the predicted values of each model given the 
          fit parameters. 
    """

    if models_to_run is None:
        models_to_run = list(MODEL_LIBRARY.keys())
    
    # Loop over all genotypes
    results_list = []
    pred_results_list = []
    get_columns = [x_column, y_column, y_std_column]

    x_pred = xfill(pd.unique(df[x_column]),num_points=100)

    for genotype in pd.unique(df["genotype"]):
        this_data = df.loc[
            df["genotype"] == genotype, get_columns
        ].values

        # Call the fitting function which returns a flat dictionary
        fit_results, fit_pred_df = cat_fit(
            this_data[:, 0],
            this_data[:, 1],
            this_data[:, 2],
            x_pred=x_pred,
            verbose=verbose,
            models_to_run=models_to_run
        )
        
        fit_results['genotype'] = genotype
        results_list.append(fit_results)

        fit_pred_df["genotype"] = genotype
        pred_results_list.append(fit_pred_df)
        
    # Create a single "main" DataFrame from the list of dictionaries
    main_df = pd.DataFrame(results_list).set_index("genotype")

    # Create the summary DataFrame
    summary_cols = [
        "best_model", "best_model_R2", "best_model_AIC_weight", "status"
    ]
    aic_weight_cols = [c for c in main_df.columns if c.startswith('AIC_weight|')]

    summary_dataframe = main_df[summary_cols + aic_weight_cols].copy()
    
    # clean up AIC weight column names
    summary_dataframe.rename(
        columns={c: c.replace('AIC_weight|', 'w_') for c in aic_weight_cols},
        inplace=True
    )

    # Create the model-specific DataFrames 
    model_dataframes = {}    
    for model_name in models_to_run:
                
        # Select all columns related to this model
        model_cols = [c for c in main_df.columns if c.startswith(f"{model_name}|")]
        
        # Create a new DataFrame for this model
        model_df = main_df[model_cols].copy()

        # Clean up the column names
        # from "model_name|param_name|est" -> "param_name_est"
        new_colnames = {}
        for col in model_df.columns:
            parts = col.split('|')
            new_name = f"{parts[1]}_{parts[2]}"
            new_colnames[col] = new_name
        model_df.rename(columns=new_colnames, inplace=True)
        
        # Add summary columns for this specific model
        model_df['is_best_model'] = (main_df['best_model'] == model_name)
        
        # The following lines assume that cat_fit returns R2 and AIC_weight for
        # each model, named like 'R2|model_name' and 'AIC_weight|model_name'.
        model_df['R2'] = main_df.get(f'R2|{model_name}', pd.Series(index=main_df.index, dtype=float))
        model_df['AIC_weight'] = main_df[f'AIC_weight|{model_name}']

        # Clean up the columns so they come out in est, std, stats order
        param_names = MODEL_LIBRARY[model_name]["param_names"]
        current_columns = list(model_df.columns)
        for p in param_names:
            current_columns.remove(f"{p}_est")
            current_columns.remove(f"{p}_std")
        
        new_columns = [f"{p}_est" for p in param_names]
        new_columns.extend([f"{p}_std" for p in param_names])
        new_columns.extend(current_columns)
        model_df = model_df.loc[:,new_columns]

        model_dataframes[model_name] = model_df

    pred_df = pd.concat(pred_results_list,ignore_index=True)
    pred_df.index = pred_df["genotype"]

    return model_dataframes, summary_dataframe, pred_df