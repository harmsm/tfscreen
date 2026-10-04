"""
Write synthetic inputs for tfs-calibrate-od600 (replicates.csv and
plate_counts.csv), to show the formats and exercise the tool.

The numbers are made up, not any instrument's: a culture with
cfu/mL = 8e8 OD + 2e8 OD^2, a reader whose readings sit on a 0.088 floor
with 1.5% noise, 1:3 dilution series read ten times, and 16 cultures
plated at 1e5 dilution (five 1:10 steps, 0.1 mL spread). Each lab
measures its own calibration; do not use these for real data.

    python make_example_data.py
    tfs-calibrate-od600 replicates.csv plate_counts.csv --out_prefix od600_calibration
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260927)


def cfu_per_mL(od):
    return 8e8 * od + 2e8 * od ** 2


# Ten readings of each of six 1:3 dilutions of an OD600 ~0.6 culture.
floor, rel_sd = 0.088, 0.015
dilution = 1.0 / 3.0 ** np.arange(6)
rows = []
for d in dilution:
    true = floor + 0.52 * d
    for _ in range(10):
        rows.append({"dilution": d,
                     "od600": round(true * (1 + rel_sd * rng.normal()), 4)})
pd.DataFrame(rows).to_csv("replicates.csv", index=False)

# Sixteen cultures across OD600 0.12-0.6, each read once and plated.
od_true = np.linspace(0.12, 0.6, 16)
od_read = od_true * (1 + rel_sd * rng.normal(size=od_true.size))
fold, volume = 1e5, 0.1
colonies = rng.poisson(cfu_per_mL(od_true) * volume / fold)
pd.DataFrame({"od600": np.round(od_read, 4),
              "colonies": colonies,
              "dilution": fold,
              "plated_volume_mL": volume,
              "num_dilutions": 5,
              "plating_steps": 1}).to_csv("plate_counts.csv", index=False)
