"""
Compare staged-MAP runs with the hand-chained real-data fit.

    python compare.py <reference_run> <run> [<run> ...]

Run from planning/dev-data/real_fit. The reference is the hand-chained
level-offset MAP and its arrowhead Laplace (rel_off_n05 with
rel_off_n05_alap for the intervals: pass rel_off_n05_alap, whose
directory holds the extracted parameters, and the script reads the MAP's
score from rel_off_n05). For each run it prints:

- the final loss (last row of tfs_fit_model_losses.txt) and the count
  log-likelihood (score_tubes.csv, from score_counts.py), and the
  difference from the reference;
- the tube offsets' SD and range (tfs_fit_model_params.npz);
- for every spiked genotype, whether its log K and n medians fall inside
  the reference's 95% intervals (the pipeline plan's done criterion).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

OFF = "sample_offset_offset_auto_loc"


def _loss(run):
    path = os.path.join(run, "tfs_fit_model_losses.txt")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    return float(df.iloc[-1]["loss"])


def _score(run):
    path = os.path.join(run, "score_tubes.csv")
    return float(pd.read_csv(path)["ll"].sum()) if os.path.exists(path) else np.nan


def _offsets(run):
    path = os.path.join(run, "tfs_fit_model_params.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if OFF not in z.files:
            return None
        off = np.asarray(z[OFF])
    return off[off != 0]


def _param(run, name):
    hits = [p for p in glob.glob(os.path.join(run, f"*_{name}.csv"))
            if "hyper" not in os.path.basename(p)]
    return pd.read_csv(hits[0]) if hits else None


def _spikes(run):
    path = os.path.join(run, "tfs_configure_library.csv")
    lib = pd.read_csv(path)
    return sorted(lib.loc[lib["in_spiked_origin"].astype(bool), "genotype"])


def main():
    ref, runs = sys.argv[1], sys.argv[2:]
    ref_map = ref[:-len("_alap")] if ref.endswith("_alap") else ref
    ref_loss, ref_score = _loss(ref_map), _score(ref_map)
    spikes = _spikes(ref_map)
    print(f"reference {ref_map}: loss {ref_loss:.6g}, count ll {ref_score:.6g}")

    for run in runs:
        print(f"\n{run}")
        loss, score = _loss(run), _score(run)
        print(f"  loss {loss:.6g} (ref {loss - ref_loss:+.4g}); "
              f"count ll {score:.6g} (ref {score - ref_score:+.4g})")
        off = _offsets(run)
        if off is not None and off.size:
            print(f"  tube offsets: {off.size} nonzero, SD {off.std():.3f}, "
                  f"range {off.min():+.2f} to {off.max():+.2f}")
        for name in ("log_hill_K", "hill_n"):
            mine, theirs = _param(run, name), _param(ref, name)
            if mine is None or theirs is None:
                print(f"  {name}: missing extracted parameters")
                continue
            m = mine.set_index("genotype")
            t = theirs.set_index("genotype")
            inside = []
            for g in spikes:
                if g not in m.index or g not in t.index:
                    continue
                v = float(m.loc[g, "q0.5"].mean())
                lo, hi = float(t.loc[g, "q0.025"].mean()), float(t.loc[g, "q0.975"].mean())
                inside.append(lo <= v <= hi)
                if not inside[-1]:
                    print(f"    {name} {g}: {v:.3g} outside [{lo:.3g}, {hi:.3g}]")
            print(f"  {name}: {sum(inside)} of {len(inside)} spikes inside the "
                  f"reference's 95% interval")


if __name__ == "__main__":
    main()
