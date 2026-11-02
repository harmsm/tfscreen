"""
Score the stage A fits against the simulation truth.

    ~/miniconda3/bin/python score.py            # from this directory

Writes results/stageA_constants.csv, stageA_theta.csv, stageA_runs.csv,
stageA_nesting.csv and stageA_sigma.csv, and prints them. All constants
are in kT.

- constants: each per-mutation d_ln_K_* the fit and the truth share, by
  posterior median and by MAP point: RMSE, Pearson r, 95% coverage and
  median 95% width of the Laplace interval.
- theta: every genotype x concentration of the growth data, the same
  scores.
- runs: MAP epochs, final exact loss and decision per stage, and the wall
  time and peak memory of the MAP and the Laplace from job.log.
- nesting (A1): the O2_C12 U1 PK fit against the dimer PK fit with the
  second engagement off, MAP point against MAP point.
- sigma (A2 PddG): learned sigma_s per state against the truth's.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")


def _scores(est, ref, lo=None, hi=None):
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    row = {"n": int(ok.sum()),
           "rmse": float(np.sqrt(np.mean((est - ref) ** 2))),
           "bias": float(np.mean(est - ref)),
           "r": float(np.corrcoef(est, ref)[0, 1]) if ok.sum() > 2 else np.nan}
    if lo is not None:
        lo, hi = lo[ok], hi[ok]
        row["cov95"] = float(np.mean((ref >= lo) & (ref <= hi)))
        row["width95"] = float(np.median(hi - lo))
    return row


def _fits():
    for d in sorted(glob.glob(os.path.join(RUNS, "*", "fit_*"))):
        if os.path.exists(os.path.join(d, ".done", "extract")):
            run = os.path.basename(os.path.dirname(d))
            yield run, os.path.basename(d)[4:], d


def constants():
    rows = []
    for run, arm, d in _fits():
        truth = pd.read_csv(os.path.join(RUNS, run, "sim",
                                         "tfs_sim_mutation_parameters.csv"))
        truth = truth.set_index("mutation")
        for f in sorted(glob.glob(os.path.join(d, "tfs_params_d_ln_K_*.csv"))):
            term = os.path.basename(f)[len("tfs_params_"):-4]
            if term not in truth.columns:
                continue
            post = pd.read_csv(f)
            if "titrant_name" in post.columns:
                post = post.groupby("mutation", as_index=False).first()
            post = post.set_index("mutation")
            ref = truth.loc[post.index, term].to_numpy()
            r = _scores(post["q0.5"].to_numpy(), ref,
                        post["q0.025"].to_numpy(), post["q0.975"].to_numpy())
            mp = os.path.join(d, f"tfs_map_params_{term}.csv")
            if os.path.exists(mp):
                m = pd.read_csv(mp)
                if "titrant_name" in m.columns:
                    m = m.groupby("mutation", as_index=False).first()
                m = m.set_index("mutation").loc[post.index]
                r["map_rmse"] = _scores(m["q0.5"].to_numpy(), ref)["rmse"]
            rows.append({"run": run, "arm": arm, "term": term, **r})
    return pd.DataFrame(rows)


def _theta_frame(path):
    df = pd.read_csv(path)
    return df.set_index(["genotype", "titrant_conc"])


def theta():
    rows = []
    for run, arm, d in _fits():
        truth = _theta_frame(os.path.join(RUNS, run, "sim",
                                          "tfs_sim_genotype_theta.csv"))
        post = _theta_frame(os.path.join(d, "tfs_pred_theta.csv"))
        post = post[post["in_training_data"] == 1]
        idx = post.index.intersection(truth.index)
        ref = truth.loc[idx, "theta"].to_numpy()
        p = post.loc[idx]
        r = _scores(p["q0.5"].to_numpy(), ref,
                    p["q0.025"].to_numpy(), p["q0.975"].to_numpy())
        m = _theta_frame(os.path.join(d, "tfs_map_theta.csv")).loc[idx]
        r["map_rmse"] = _scores(m["q0.5"].to_numpy(), ref)["rmse"]
        r["nonfinite"] = int((~np.isfinite(p["q0.5"])).sum())
        rows.append({"run": run, "arm": arm, **r})
    return pd.DataFrame(rows)


def _timings(log):
    """(step, wall s, peak GB) for each timed step in a job.log."""
    out, step = [], None
    with open(log) as f:
        for line in f:
            m = re.match(r">>> (\S+(?: \S+)?)", line)
            if m:
                step = m.group(1)
            m = re.match(r"\s+([\d.]+) real", line)
            if m:
                wall = float(m.group(1))
            m = re.match(r"\s+(\d+)\s+maximum resident set size", line)
            if m:
                out.append((step, wall, int(m.group(1)) / 1e9))
    return out


def runs():
    rows = []
    for run, arm, d in _fits():
        row = {"run": run, "arm": arm}
        for stage, tag in (("stage1", "_stage1"), ("stage2", "_stage2"),
                           ("stage3", "")):
            f = os.path.join(d, f"tfs_fit_model{tag}_convergence.csv")
            if not os.path.exists(f):
                continue
            c = pd.read_csv(f)
            last = c.iloc[-1]
            row[f"{stage}_step"] = int(last["step"])
            row[f"{stage}_decision"] = last["decision"]
            if "loss_exact" in c.columns:
                row[f"{stage}_loss"] = float(last["loss_exact"])
        for step, wall, gb in _timings(os.path.join(d, "job.log")):
            key = step.split()[0].lower()
            row[f"{key}_min"] = round(wall / 60, 1)
            row[f"{key}_GB"] = round(gb, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def nesting():
    a = os.path.join(RUNS, "A1", "fit_O2PK")
    b = os.path.join(RUNS, "A1", "fit_dimerPK_nested")
    if not all(os.path.exists(os.path.join(x, ".done", "extract")) for x in (a, b)):
        return pd.DataFrame()
    rows = []
    ta = _theta_frame(os.path.join(a, "tfs_map_theta.csv"))
    tb = _theta_frame(os.path.join(b, "tfs_map_theta.csv"))
    idx = ta.index.intersection(tb.index)
    rows.append({"quantity": "theta (MAP)",
                 **_scores(tb.loc[idx, "q0.5"].to_numpy(),
                           ta.loc[idx, "q0.5"].to_numpy())})
    for f in sorted(glob.glob(os.path.join(a, "tfs_map_params_d_ln_K_*.csv"))):
        term = os.path.basename(f)[len("tfs_map_params_"):-4]
        g = os.path.join(b, os.path.basename(f))
        if not os.path.exists(g):
            continue
        pa = pd.read_csv(f).groupby("mutation").first()
        pb = pd.read_csv(g).groupby("mutation").first()
        idx = pa.index.intersection(pb.index)
        rows.append({"quantity": f"{term} (MAP)",
                     **_scores(pb.loc[idx, "q0.5"].to_numpy(),
                               pa.loc[idx, "q0.5"].to_numpy())})
    return pd.DataFrame(rows)


def sigma():
    rows = []
    for run, arm, d in _fits():
        f = os.path.join(d, "tfs_params_ddG_sigma.csv")
        t = os.path.join(RUNS, run, "sim", "tfs_sim_ddG_sigma.csv")
        if not (os.path.exists(f) and os.path.exists(t)):
            continue
        est = pd.read_csv(f).set_index("structure")
        ref = pd.read_csv(t).set_index("structure")["ddG_sigma"]
        for s in est.index:
            rows.append({"run": run, "arm": arm, "structure": s,
                         "truth": float(ref.get(s, np.nan)),
                         "q0.5": float(est.loc[s, "q0.5"]),
                         "q0.025": float(est.loc[s, "q0.025"]),
                         "q0.975": float(est.loc[s, "q0.975"])})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for name, fn in (("runs", runs), ("constants", constants),
                     ("theta", theta), ("nesting", nesting),
                     ("sigma", sigma)):
        df = fn()
        df.to_csv(os.path.join(OUT, f"stageA_{name}.csv"), index=False)
        print(f"\n== {name}")
        print(df.round(3).to_string(index=False) if len(df) else "(none)")


if __name__ == "__main__":
    main()
