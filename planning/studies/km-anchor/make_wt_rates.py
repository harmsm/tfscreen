"""
wt monoculture rate tables for the k/m anchor arms.

    python make_wt_rates.py <wt_control_rate_summary.csv> [--seed 1]

Writes, in the current directory:

- wt_rates_selection.csv: the selection conditions only (condition_sel
  containing '+'). The control conditions are left to the loose priors: in
  them wt's monoculture rate varies with IPTG for reasons the fit's m = 0
  cannot carry, and the full-size simulation sets their m to 0.
- wt_rates_selection_perturbed.csv: the same, with each rate_mean moved
  once by a Normal draw of the SD the prior assumes (the standard error of
  the replicate mean, floored at 0.002 per minute, as
  priors_edit.growth_priors_from_wt_rates does). The honest-anchor arm: a
  monoculture is a measurement, not the truth.
"""

import argparse

import numpy as np
import pandas as pd

SD_FLOOR = 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rates")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    r = pd.read_csv(args.rates)
    sel = r[r["condition_sel"].str.contains("+", regex=False)].copy()
    sel.to_csv("wt_rates_selection.csv", index=False)

    rng = np.random.default_rng(args.seed)
    se = sel["rate_sd"] / np.sqrt(sel["num_replicates"])
    sd = np.maximum(se.to_numpy(), SD_FLOOR)
    pert = sel.copy()
    pert["rate_mean"] = sel["rate_mean"].to_numpy() + rng.normal(0.0, sd)
    pert.to_csv("wt_rates_selection_perturbed.csv", index=False)

    show = sel[["condition_sel", "titrant_conc", "rate_mean"]].assign(
        perturbed=pert["rate_mean"].to_numpy(), sd=sd)
    print(show[show["titrant_conc"].isin([0.0, 1.0])].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
