
from tfscreen.analysis.independent.cfu_to_theta import cfu_to_theta
from tfscreen.analysis.independent.process_theta_fit import process_theta_fit
from tfscreen.util.cli import generalized_main

def analyze_theta(df,
                  calibration_data,
                  non_sel_conditions=None,
                  out_root="tfs",
                  max_batch_size=250,
                  logistic_theta=False,
                  model_name=None,
                  transition_model_name=None):
    """
    Perform a theta/growth rate fit on the provided experimental data.

    Parameters
    ----------
    df : pd.DataFrame or str
        Experimental data (ln_cfu, etc).
    calibration_data : dict or str
        Calibration constants for the models.
    non_sel_conditions : list, optional
        Conditions where selection is absent, used to guess dk_geno.
    out_root : str, optional
        Root name for output files.
    max_batch_size : int, optional
        Maximum number of genotypes per fitting batch.
    logistic_theta : bool, optional
        Whether to use a logistic transformation for theta.
    model_name : str, optional
        Growth model to use: 'linear', 'power_law', or 'saturation'. 
        If None, use value from calibration_data or 'linear'.
    transition_model_name : str, optional
        Transition model to use. If None, use value from calibration_data or 'constant'.
    """
    
    print("--> Performing fit.",flush=True)
    param_df, pred_df = cfu_to_theta(df=df,
                                     non_sel_conditions=non_sel_conditions,
                                     calibration_data=calibration_data,
                                     max_batch_size=max_batch_size,
                                     logistic_theta=logistic_theta,
                                     model_name=model_name,
                                     transition_model_name=transition_model_name)

    print("--> Aggregating results results.",flush=True)
    out = process_theta_fit(param_df,pred_df)

    param_df.to_csv(f"{out_root}_param.csv",index=False)
    pred_df.to_csv(f"{out_root}_pred.csv",index=False)
    out["theta"].to_csv(f"{out_root}_theta.csv",index=False)
    out["dk_geno"].to_csv(f"{out_root}_dk_geno.csv",index=False)
    out["ln_cfu0"].to_csv(f"{out_root}_ln_cfu0.csv",index=False)
    out["pred"].to_csv(f"{out_root}_pred_proc.csv",index=False)

    print(f"--> Wrote results to {out_root}_*.csv",flush=True)

def main():
    return generalized_main(analyze_theta,
                            manual_arg_nargs={"non_sel_conditions":"+"},
                            manual_arg_types={"non_sel_conditions":str})