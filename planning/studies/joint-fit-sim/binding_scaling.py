"""
Theta of growth-only singles against the number of binding singles
(README, run 10).

    python binding_scaling.py runs --out out

Scores every stage 1 fit in RUNS_DIR named fit_nocong_s1_plain_single[_libN[_bK]]
or fit_null_s1_plain[_libN[_bK]] (analyze.theta_rows; growth-only singles),
writes out/binding_scaling.csv (one row per fit: sim, n, pick, m_steep,
rmse, slope, cov95, width, sd_z) and draws RMSE, 95% coverage and SD of z
against n, each pick a point and the mean a line, with the fit at the true
k and m (run 5) as the floor and the curve floor + a / sqrt(n + 2).
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd

import analyze
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rc("pdf", fonttype=42)
plt.rc("font", size=14)
plt.rc("axes", titlesize=14, labelsize=16)
plt.rc("legend", fontsize=12)

FLOOR = {"nocong": dict(rmse=0.010, cov95=0.910, sd_z=0.909),
         "null": dict(rmse=0.020, cov95=0.699, sd_z=1.845)}
COLOR = {"nocong": "#0072B2", "null": "#D55E00"}
LABEL = {"nocong": "no congression", "null": "congression"}


BASES = {"nocong": "fit_nocong_s1_plain_single", "null": "fit_null_s1_plain"}


def parse(name, base):
    """(n, pick) for base, base_libN or base_libN_bK; None otherwise."""
    rest = name[len(base):]
    if rest == "":
        return 0, 1
    if not rest.startswith("_lib"):
        return None
    parts = rest[4:].split("_b")
    if not parts[0].isdigit() or (len(parts) > 1 and not parts[1].isdigit()):
        return None
    return int(parts[0]), (int(parts[1]) if len(parts) > 1 else 1)


def build_table(runs_dir):
    rows = []
    for sim, base in BASES.items():
        sim_dir = os.path.join(runs_dir, f"sim_{sim}_s1")
        truth = pd.read_csv(os.path.join(sim_dir, "tfs_sim_growth_parameters.csv"))
        truth = truth.set_index("condition_rep")["growth_m"]
        for d in sorted(os.listdir(runs_dir)):
            run = os.path.join(runs_dir, d)
            got = parse(d, base) if d.startswith(base) else None
            if got is None or not os.path.exists(os.path.join(run, "tfs_pred_theta.csv")):
                continue
            th = analyze.theta_rows(run, sim_dir, d)
            s = th[th["group"] == "single"].iloc[0]
            m = pd.read_csv(os.path.join(run, "tfs_params_growth_m.csv"))
            m = m.set_index("condition_rep")["q0.5"]
            steep = np.mean([abs(m[c] / truth[c] - 1) for c in ("kanR+kan", "pheS+4CP")])
            rows.append(dict(sim=sim, n=got[0], pick=got[1], m_steep=100 * steep,
                             rmse=s["rmse"], slope=s["slope"], cov95=s["coverage95"],
                             width=s["median_width95"], sd_z=s["sd_z"]))
    return pd.DataFrame(rows).sort_values(["sim", "n", "pick"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = build_table(args.runs_dir)
    r.to_csv(os.path.join(args.out, "binding_scaling.csv"), index=False)
    print(r.round(3).to_string(index=False))
    nocong = r[r["sim"] == "nocong"].groupby("n")["rmse"].mean()
    a = float(np.mean((nocong - FLOOR["nocong"]["rmse"]) * np.sqrt(nocong.index + 2)))
    print(f"no congression: excess RMSE = {a:.4f} / sqrt(n + 2)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, name in zip(axes, ["rmse", "cov95", "sd_z"],
                             ["theta RMSE", "95% coverage", "SD of z"]):
        for sim, d in r.groupby("sim"):
            x = d["n"] + 1
            ax.plot(x, d[col], "o", color=COLOR[sim], alpha=0.5, ms=5)
            mean = d.groupby("n")[col].mean()
            ax.plot(mean.index + 1, mean.values, "-", color=COLOR[sim], label=LABEL[sim])
            ax.axhline(FLOOR[sim][col], color=COLOR[sim], ls=":", lw=1)
        if col == "rmse":
            n = np.linspace(0, 60, 200)
            ax.plot(n + 1, FLOOR["nocong"]["rmse"] + a / np.sqrt(n + 2), "--",
                    color="0.4", lw=1, label=f"{FLOOR['nocong']['rmse']} + {a:.3f}/sqrt(n+2)")
        if col == "cov95":
            ax.axhline(0.95, color="0.6", lw=0.8)
        if col == "sd_z":
            ax.axhline(1.0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 6, 11, 21, 51])
        ax.set_xticklabels(["0", "2", "5", "10", "20", "50"])
        ax.set_xlabel("binding singles")
        ax.set_ylabel(name)
    axes[0].legend(fontsize=10)
    fig.suptitle("growth-only singles; dotted: fit at the true k and m", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out, f"{datetime.date.today().isoformat()}_binding-scaling.pdf")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
