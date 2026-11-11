"""
Compare tau_test.sh's refits:  python tau_test_report.py

For each runs/<run>/tautest_<arm>_<mode>/ prints the final epoch, step
size, decision and exact loss (lower is better; the same objective for
cont and flip of one fit), the pair horseshoe's tau and c2, and the
original fit_<arm>'s final exact loss and tau. A flip that ends below its
cont means the regime the original fit was in is not the optimum.
"""

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _final(d):
    conv = os.path.join(d, "tfs_fit_model_convergence.csv")
    out = {"epoch": np.nan, "step_size": np.nan, "decision": "",
           "loss_exact": np.nan, "tau": np.nan, "c2": np.nan}
    if os.path.exists(conv):
        c = pd.read_csv(conv).iloc[-1]
        out.update(epoch=c["epoch"], step_size=c["step_size"],
                   decision=c["decision"], loss_exact=c["loss_exact"])
    npz = os.path.join(d, "tfs_fit_model_params.npz")
    if os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(conv):
        with np.load(npz) as p:
            out.update(tau=float(p["theta_epi_tau_auto_loc"]),
                       c2=float(p["theta_epi_c2_auto_loc"]))
    return out


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*", "tautest_*"))):
        run = os.path.basename(os.path.dirname(d))
        arm, mode = os.path.basename(d)[len("tautest_"):].rsplit("_", 1)
        orig = _final(os.path.join(os.path.dirname(d), f"fit_{arm}"))
        r = _final(d)
        rows.append({"run": run, "arm": arm, "mode": mode, **r,
                     "orig_loss_exact": orig["loss_exact"],
                     "orig_tau": orig["tau"]})
    if not rows:
        print("no runs/*/tautest_* directories yet: has tau_test_submit.sh "
              "run, and have its jobs started?")
        return
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    wide = df.pivot_table(index=["run", "arm"], columns="mode",
                          values="loss_exact")
    if {"cont", "flip"} <= set(wide.columns):
        wide["flip_minus_cont"] = wide["flip"] - wide["cont"]
        print()
        print(wide.to_string(float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
