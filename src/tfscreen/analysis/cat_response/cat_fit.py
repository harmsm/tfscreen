from .cat_library import MODEL_LIBRARY

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error
)

from tfscreen.util import xfill

import numpy as np
import pandas as pd

def cat_fit(x, y, y_std, x_pred=None, models_to_run=None, verbose=False):
    """
    Fits multiple models to a single dataset and returns a flat dictionary
    of all results, suitable for aggregation.

    Parameters
    ----------
    x, y, y_std : np.ndarray
        The independent variable, dependent variable, and standard error of the
        dependent variable.
    x_pred : np.ndarray, optional
        array at which to predict x after fitting each model. If not specified, 
        fill in values within x. 
    models_to_run : list of str, optional
        A list of model names to test. If None (default), all models in the
        global MODEL_LIBRARY will be tested.
    verbose : bool, optional
        If True, prints warnings to the console when a model fails to fit.
        Defaults to False.

    Returns
    -------
    dict
        A single, flat dictionary containing the best model summary, AIC
        weights for all tested models, and all parameter estimates and
        standard errors for all tested models.
    pd.DataFrame
        a pandas dataframe holding model predictions. dataframe has five columns:
        - 'model': name of model 
        - 'x': x-values used as the independent variable in the model
        - 'y': y-values predicted by the model
        - 'y_std': standard error on the y-values predicted by the model. (this
           will be nan if the covariance matrix is not finite)
        - 'is_best_model': whether or not this model was the best model tested
    """

    if models_to_run is None:
        models_to_run = list(MODEL_LIBRARY.keys()) 

    flat_output = {}

    # Sanitize inputs
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_std) & (y_std > 0)
    x = x[finite_mask]
    y = y[finite_mask]
    y_std = y_std[finite_mask]

    if x_pred is None:
        x_pred = xfill(x)

    model_pred_out = []
    x_pred_out = []
    y_pred_out = []
    y_pred_std_out = []

    # Handle insufficient data case by returning nan dict and pred_df
    if len(x) < 2:

        # Build nan flat_output
        flat_output['status'] = "missing"
        flat_output['best_model'] = "None"
        flat_output['best_model_R2'] = np.nan
        flat_output['best_model_AIC_weight'] = np.nan
        for name in models_to_run:

            param_names = MODEL_LIBRARY[name]["param_names"]
            flat_output[f"R2|{name}"] = np.nan
            flat_output[f"AIC_weight|{name}"] = np.nan
            for p_name in param_names:
                flat_output[f"{name}|{p_name}|est"] = np.nan
                flat_output[f"{name}|{p_name}|std"] = np.nan

        # Build nan dataframe for pred_df
        model_pred_out = np.repeat(models_to_run,len(x_pred))
        pred_df = pd.DataFrame({"model":model_pred_out,
                                "x":np.tile(x_pred,len(models_to_run)),
                                "y":np.full(np.nan,len(model_pred_out)),
                                "y_std":np.full(np.nan,len(model_pred_out)),
                                "is_best_model":np.zeros(len(model_pred_out),
                                                         dtype=bool)})
        
        return flat_output, pred_df

    # Iterate through models and fit
    n = len(y)
    summary_results = []
    param_results = {}
    for name in models_to_run:

        model_func = MODEL_LIBRARY[name]["model_func"]
        guess_func = MODEL_LIBRARY[name]["guess_func"]
        param_names = MODEL_LIBRARY[name]["param_names"]
        bounds = MODEL_LIBRARY[name]["bounds"]
        k = len(param_names)
        
        try:
            
            guesses = guess_func(x, y)
            
            params, std_err, cov_matrix, fit_obj = run_least_squares(
                model_func, y, y_std, guesses, bounds[0], bounds[1], args=(x,)
            )
            if not fit_obj.success:
                raise RuntimeError(f"Fit failed: {fit_obj.message}")

            y_fit = model_func(params, x)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            aic = 2 * k + n * np.log(ss_res / n) if ss_res > 0 else -np.inf

            summary_results.append({"model": name, "R2": r2, "AIC": aic, "success": True})
            param_results[name] = {"params": params, "std_err": std_err, "names": param_names}

        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"Warning: Model '{name}' failed to fit. Reason: {e}")
            summary_results.append({"model": name, "R2": np.nan, "AIC": np.nan, "success": False})
            param_results[name] = {
                "params": np.full(k, np.nan), "std_err": np.full(k, np.nan), "names": param_names
            }
            params = None
    
        if params is not None:
             
            y_pred, y_pred_std = predict_with_error(model_func,
                                                    params,
                                                    cov_matrix,
                                                    args=[x_pred])
            
            x_pred_out.extend(x_pred)
            y_pred_out.extend(y_pred)
            y_pred_std_out.extend(y_pred_std)
            model_pred_out.extend([name for _ in x_pred])

    # Post-process and flatten results
    summary_df = pd.DataFrame(summary_results)
    valid_aics = summary_df.loc[summary_df['success'], 'AIC']
    if not valid_aics.empty:
        min_aic = valid_aics.min()
        summary_df['delta_AIC'] = summary_df['AIC'] - min_aic
        relative_likelihood = np.exp(-0.5 * summary_df['delta_AIC'])
        sum_likelihoods = relative_likelihood.sum()
        summary_df['AIC_weight'] = relative_likelihood / sum_likelihoods
    else:
        summary_df['AIC_weight'] = np.nan
    
    summary_df = summary_df.sort_values(by="AIC").reset_index(drop=True)
    
    # Populate the flat output dictionary
    for _, row in summary_df.iterrows():
        model_name = row['model']
        flat_output[f"AIC_weight|{model_name}"] = row['AIC_weight']
        flat_output[f"R2|{model_name}"] = row['R2']
        
        # Unpack params for this model
        p_res = param_results[model_name]
        for i, p_name in enumerate(p_res['names']):
            flat_output[f"{model_name}|{p_name}|est"] = p_res['params'][i]
            flat_output[f"{model_name}|{p_name}|std"] = p_res['std_err'][i]

    # Add overall status and best model info
    success_states = summary_df['success'].unique()
    if len(success_states) == 1 and success_states[0] == True: flat_output['status'] = "success"
    elif len(success_states) == 1 and success_states[0] == False: flat_output['status'] = "failure"
    else: flat_output['status'] = "partial"
    
    if not valid_aics.empty:
        best_model_row = summary_df.iloc[0]
        flat_output['best_model'] = best_model_row['model']
        flat_output['best_model_R2'] = best_model_row['R2']
        flat_output['best_model_AIC_weight'] = best_model_row['AIC_weight']
    else:
        flat_output['best_model'] = "None"
        flat_output['best_model_R2'] = np.nan
        flat_output['best_model_AIC_weight'] = np.nan

    # Build final dataframe holding all predictions from the models
    pred_df = pd.DataFrame({"model":model_pred_out,
                            "x":x_pred_out,
                            "y":y_pred_out,
                            "y_std":y_pred_std_out})
    pred_df["is_best_model"] = pred_df["model"] == flat_output["best_model"]
    

    return flat_output, pred_df