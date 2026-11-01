"""Pool remedies_s*.json into remedies_results.csv and print a compact table."""

import glob
import json

import pandas as pd

rows = []
for f in sorted(glob.glob("remedies_s*.json")):
    rows += json.load(open(f))
df = pd.DataFrame(rows)
df.to_csv("remedies_results.csv", index=False)

keys = ["design", "fit", "doubles", "pair_scale"]
cols = ["detect_z"]
for t in ("dbl", "quint"):
    for g in ("all", "uhit", "mhit"):
        cols += [f"{t}_{g}_err_q90", f"{t}_{g}_cov95", f"{t}_{g}_frac_z3"]
cols = [c for c in cols if c in df]
order = {"wt_only": 0, "rec6": 1, "star": 2}
summ = df.groupby(keys, sort=False)[cols].median().reset_index()
summ["o"] = summ.design.map(order)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(summ.sort_values("o", kind="stable").drop(columns="o")
      .round(3).to_string(index=False))
n = df.groupby("seed")[[c for c in df if c.endswith("_n")]].first()
print(n.to_string())
