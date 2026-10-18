"""
Report joint stage 1 fits on the real screen (README).

    python report.py runs/plain runs/full --binding BINDING_CSV --out out

For each fit: its final MAP loss; k and m per condition (with the
within-tube m of planning/studies/within-tube-m for reference); the warps,
log s and the offset scale tau when present; and per binding genotype its
offset and binding residuals (theta_obs minus the fitted curve at the
binding concentrations plus the offset), as RMS and RMS in units of
theta_std. Writes out/params.csv and out/binding_residuals.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

Q = ["q0.025", "q0.5", "q0.975"]
# kanR+kan and pheS+4CP, change of the wt/flat-reference slope from 0 to 1 mM
# (planning/studies/within-tube-m, run 3, swapped row left out).
WITHIN_TUBE_M = {"kanR+kan": -0.0046, "pheS+4CP": 0.0125}


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def params(run, name):
    rows = []
    for p in ("growth_k", "growth_m", "warp_log_a", "warp_x0",
              "growth_log_conc_shift", "offset_scale"):
        d = _read(os.path.join(run, f"tfs_params_{p}.csv"))
        if d is None:
            continue
        key = ("condition_rep" if "condition_rep" in d.columns
               else "parameter" if "parameter" in d.columns else d.columns[0])
        for _, r in d.iterrows():
            rows.append(dict(run=name, parameter=p, key=r[key],
                             **{q: r[q] for q in Q}))
    out = pd.DataFrame(rows)
    out["within_tube_m"] = np.where(out["parameter"] == "growth_m",
                                    out["key"].map(WITHIN_TUBE_M), np.nan)
    return out


def residuals(run, name, binding):
    pred = pd.read_csv(os.path.join(run, "tfs_pred_theta_binding.csv"))
    pred["genotype"] = pred["genotype"].astype(str)
    b = binding.merge(pred[["genotype", "titrant_conc", "q0.5"]],
                      on=["genotype", "titrant_conc"], how="inner")
    off = _read(os.path.join(run, "tfs_params_offset.csv"))
    offset = ({} if off is None
              else dict(zip(off["genotype"].astype(str), off["q0.5"])))
    b["offset"] = b["genotype"].map(offset).fillna(0.0)
    b["resid"] = b["theta_obs"] - (b["q0.5"] + b["offset"])
    b["z"] = b["resid"] / b["theta_std"]
    out = (b.groupby("genotype")
           .agg(n=("resid", "size"), offset=("offset", "first"),
                mean_resid=("resid", "mean"),
                rms_resid=("resid", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                rms_z=("z", lambda x: float(np.sqrt(np.mean(x ** 2)))))
           .reset_index())
    out.insert(0, "run", name)
    return out


def final_loss(run):
    c = _read(os.path.join(run, "tfs_fit_model_convergence.csv"))
    if c is None or c.empty:
        return np.nan, "no convergence windows"
    r = c.iloc[-1]
    return float(r["loss_exact"]), str(r["decision"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--binding", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    binding = pd.read_csv(args.binding)
    binding["genotype"] = binding["genotype"].astype(str)
    ps, rs = [], []
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        loss, decision = final_loss(run)
        print(f"{name}: final MAP loss {loss:.1f} ({decision})")
        ps.append(params(run, name))
        rs.append(residuals(run, name, binding))
    p = pd.concat(ps, ignore_index=True)
    r = pd.concat(rs, ignore_index=True)
    p.to_csv(os.path.join(args.out, "params.csv"), index=False)
    r.to_csv(os.path.join(args.out, "binding_residuals.csv"), index=False)
    pd.set_option("display.width", 200)
    print()
    print(p.to_string(index=False, float_format="%+.4f"))
    print()
    print(r.to_string(index=False, float_format="%+.3f"))


if __name__ == "__main__":
    main()
