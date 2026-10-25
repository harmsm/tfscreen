"""
Compare the warp runs (README, run 4): m and the pheS/kanR ratio, the
fitted warp (a, x0), the final exact loss, and each deep single's
predicted within-tube swing against the observed (after
tfs-predict-growth at the MAP, run here if missing).

    python compare_warp.py runs warp_real_kanR warp_real_pheS warp_real_none warp_seed1_kanR
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTER = "/gpfs/projects/harmslab/harms/studies/two-stage-validation"
WITHIN_TUBE = {"kanR+kan": -0.0046, "pheS+4CP": 0.0125}   # without rep 2, t 82


def localize(stage1, run):
    cfg = os.path.join(stage1, "config_local.yaml")
    if not os.path.exists(cfg):
        text = open(os.path.join(stage1, "tfs_configure_config.yaml")).read()
        open(cfg, "w").write(text.replace(f"{CLUSTER}/{run}/stage1/", ""))
    return cfg


def predict(stage1, run):
    if os.path.exists(os.path.join(stage1, "pred_growth.csv")):
        return
    localize(stage1, run)
    code = ("from tfscreen.tfmodel.scripts.predict_growth_cli import predict_growth;"
            "predict_growth('config_local.yaml','tfs_fit_model_checkpoint.pkl',"
            "out_prefix='pred_growth')")
    subprocess.run([sys.executable, "-c", code], cwd=stage1, check=True,
                   stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        st = os.path.join(args.runs_dir, run, "stage1")
        if not os.path.exists(os.path.join(st, "km.csv")):
            print(f"{run}: not there yet")
            continue
        km = pd.read_csv(os.path.join(st, "km.csv"), index_col="condition_rep")
        z = np.load(os.path.join(st, "tfs_fit_model_params.npz"))
        conv = pd.read_csv(os.path.join(st, "tfs_fit_model_convergence.csv"))
        rec = dict(run=run, m_kanR=km.loc["kanR+kan", "m"], m_pheS=km.loc["pheS+4CP", "m"],
                   loss=conv["loss_exact"].iloc[-1], steps=int(conv.iloc[-1, 0]),
                   decision=conv["decision"].iloc[-1])
        rec["ratio"] = rec["m_pheS"] / rec["m_kanR"]
        if "condition_growth_warp_log_a_auto_loc" in z.files:
            rec["warp_a"] = float(np.exp(np.asarray(z["condition_growth_warp_log_a_auto_loc"]).ravel()[0]))
            rec["warp_x0"] = float(np.asarray(z["condition_growth_warp_x0_auto_loc"]).ravel()[0])
        predict(st, run)
        out = os.path.join(args.runs_dir, run, "swing")
        subprocess.run([sys.executable, os.path.join(HERE, "swing_ppc.py"), st, "--out", out],
                       check=True, stdout=subprocess.DEVNULL)
        s = pd.read_csv(os.path.join(out, "swing_ppc.csv"))
        sg = s[~s["genotype"].str.contains("/") & (s["genotype"] != "wt")]
        for cond, tag in (("kanR+kan", "kanR"), ("pheS+4CP", "pheS")):
            o, p = sg[f"obs_{cond}"], sg[f"pred_{cond}"]
            rec[f"ppc_slope_{tag}"] = np.polyfit(o, p, 1)[0]
            rec[f"ppc_med_ratio_{tag}"] = p.abs().median() / o.abs().median()
        w = s[s["genotype"] == "wt"].iloc[0]
        rec["wt_pred_pheS"], rec["wt_obs_pheS"] = w["pred_pheS+4CP"], w["obs_pheS+4CP"]
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("run")
    print(t.T.to_string(float_format=lambda v: f"{v:+.4g}"))
    print("within tube (no rep 2 t82): kanR -0.0046, pheS +0.0125, ratio -2.7")


if __name__ == "__main__":
    main()
