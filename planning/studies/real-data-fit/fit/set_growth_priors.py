"""
Per-condition k and m priors for the growth-only relative fit (hill_relative).

    python ../set_growth_priors.py loose|monokan [priors_csv]

tfs-prefit-calibration refuses growth-only models, so these priors are
written by hand into tfs_configure_priors.csv as indexed rows (one per
condition_rep), which configuration_io joins to the model's conditions by
name.

In the X gauge (wt X = 1 at 0 IPTG, X = 0 at 1 mM; g = k + dk + m X, wt's
dk pinned at 0), wt grows at k + m at 0 IPTG and at k at 1 mM, so

    k = wt rate at 1 mM,   m = wt rate at 0 - wt rate at 1 mM.

loose:   every condition k ~ N(0.015, 0.01), m ~ N(0, 0.01).
monokan: kanR+kan and kanR-kan from the wt monoculture rates
         (inputs/wt_control_rate_summary.csv, a copy of
         ../processed/'s), SD the standard error
         of the replicate mean floored at 0.002 per min (the prefit's k
         scale; day-to-day and library-vs-monoculture differences are at
         least that). pheS conditions loose, as above. Mike, 2026-09-30:
         library wt grows on in pheS+4CP at high IPTG where the
         monoculture stops, so the monoculture cannot be its prior.
"""

import os
import sys

import numpy as np
import pandas as pd

CONDITIONS = ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]
LOOSE = dict(k=(0.015, 0.01), m=(0.0, 0.01))
SD_FLOOR = 0.002
MONO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "inputs", "wt_control_rate_summary.csv")


def monoculture(cond):
    r = pd.read_csv(MONO)
    r = r[r.condition_sel == cond].set_index("titrant_conc")
    lo, hi = r.loc[0.0], r.loc[1.0]
    se_lo = lo.rate_sd / np.sqrt(lo.num_replicates)
    se_hi = hi.rate_sd / np.sqrt(hi.num_replicates)
    k = (hi.rate_mean, max(se_hi, SD_FLOOR))
    m = (lo.rate_mean - hi.rate_mean, max(np.hypot(se_lo, se_hi), SD_FLOOR))
    return dict(k=k, m=m)


def main():
    which = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "tfs_configure_priors.csv"
    assert which in ("loose", "monokan"), which
    p = pd.read_csv(path)
    for c in ("flat_index", "condition_rep"):
        if c not in p.columns:
            p[c] = np.nan

    table = {}
    for cond in CONDITIONS:
        use_mono = which == "monokan" and cond.startswith("kanR")
        table[cond] = monoculture(cond) if use_mono else LOOSE

    pre = "growth.condition_growth."
    fields = {
        "k_loc": [table[c]["k"][0] for c in CONDITIONS],
        "k_scale": [table[c]["k"][1] for c in CONDITIONS],
        "m_loc": [table[c]["m"][0] for c in CONDITIONS],
        # each condition reads the one its +/- flag names; both carry the
        # same per-condition values so the flag cannot pick a wrong one
        "m_scale_plus": [table[c]["m"][1] for c in CONDITIONS],
        "m_scale_minus": [table[c]["m"][1] for c in CONDITIONS],
    }
    p = p[~p.parameter.isin([pre + f for f in fields])]
    rows = [dict(parameter=pre + f, value=v, flat_index=float(i), condition_rep=c)
            for f, vals in fields.items() for i, (c, v) in enumerate(zip(CONDITIONS, vals))]
    p = pd.concat([p, pd.DataFrame(rows)], ignore_index=True)
    p.to_csv(path, index=False)

    show = pd.DataFrame({f: vals for f, vals in fields.items()}, index=CONDITIONS)
    print(f"{which} priors written to {path}:")
    print(show.drop(columns="m_scale_minus").rename(
        columns={"m_scale_plus": "m_scale"}).round(5).to_string())


if __name__ == "__main__":
    main()
