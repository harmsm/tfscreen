"""Pool tails_s*.json into tails_results.csv and print median tables."""

import glob
import json

import pandas as pd

rows = []
for f in sorted(glob.glob("tails_s*.json")):
    rows += json.load(open(f))
df = pd.DataFrame(rows)
df.to_csv("tails_results.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

keys = ["close", "prior"]
head = ["n_dbl_rows", "frac_coupled_measured", "detect_z"]
for t in ("dbl", "quint"):
    cols = head + [f"{t}_all_{m}" for m in
                   ("err_q90", "cov95", "cov99", "frac_out99",
                    "width95_median", "width95_nopair_median",
                    "width95_curve_median")]
    cols += [f"{t}_uhit_cov95", f"{t}_mhit_cov95"]
    cols = [c for c in cols if c in df]
    print(f"\n{t}: medians over {df.seed.nunique()} seeds")
    print(df.groupby(keys, sort=False)[cols].median().round(3)
          .to_string())
print("\nnfev at cap:", int((df.nfev >= 400).sum()), "of", len(df))
