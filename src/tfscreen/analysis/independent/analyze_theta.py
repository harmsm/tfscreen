
from tfscreen.analysis.independent.cfu_to_theta import cfu_to_theta
from tfscreen.analysis.independent.process_theta_fit import process_theta_fit
from tfscreen.util.cli import generalized_main

def analyze_theta(df,
                  calibration_data,
                  non_sel_conditions=None,
                  out_root="tfs",
                  max_batch_size=250,
                  logistic_theta=False,
                  model_name=None):
    
    print("--> Performing fit.",flush=True)
    param_df, pred_df = cfu_to_theta(df=df,
                                     non_sel_conditions=non_sel_conditions,
                                     calibration_data=calibration_data,
                                     max_batch_size=max_batch_size,
                                     logistic_theta=logistic_theta,
                                     model_name=model_name)

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