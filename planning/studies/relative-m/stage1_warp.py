"""
Relative stage 1 (stage1_relative.py) with one selection marker's growth
following a monotone warp of X (linear growth's m_warp; README, run 4),
optionally without some tubes.

    python stage1_warp.py ../two-stage-validation/two_stage_real.yaml \
        ../two-stage-validation/run_real/stage1/km.csv warp_real_kanR \
        --warp kanR+kan --drop_tubes "pheS+4CP:2:82"

--warp names the warped condition_rep (none: no warp). --drop_tubes takes
condition_sel:replicate:t_sel triples (repeatable; ":" because sbatch
--export splits on commas): every IPTG of that
row is left out (the pheS+4CP replicate 2, t_sel 82 row whose contents
disagree with its labels, run 3).
"""

import argparse
import os
import re

import pandas as pd

from tfscreen.genetics import library_composition_table
from tfscreen.tfmodel import two_stage as ts
from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli

from stage1_relative import SIGMA_FIXED, relative_priors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file")
    ap.add_argument("km_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--warp", default="none")
    ap.add_argument("--drop_tubes", action="append", default=[])
    args = ap.parse_args()
    cfg, settings = cli.read_two_stage_config(args.config_file)
    out = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out, "stage1"), exist_ok=True)

    pri = os.path.join(out, "growth_priors_relative.csv")
    if not os.path.exists(pri):
        t = relative_priors(cfg["growth_priors"], args.km_csv)
        if args.warp != "none":
            if args.warp not in set(t["condition_rep"]):
                raise ValueError(f"--warp {args.warp} is not a condition_rep")
        t["m_warp"] = (t["condition_rep"] == args.warp).astype(int)
        t.to_csv(pri, index=False)
    cfg["growth_priors"] = pri
    cfg["set_priors"] = [p for p in (cfg.get("set_priors") or [])
                         if not p.split("=")[0].strip().endswith("sigma_fixed")]
    cfg["set_priors"].append(f"sigma_fixed={SIGMA_FIXED}")

    growth = os.path.join(out, "stage1", "tfs_growth.csv")
    if not os.path.exists(growth):
        lib = library_composition_table(cfg["library_config"])
        keep = ts.stage1_genotypes(lib, ts.growth_genotypes(cfg["growth_df"]))
        tmp = growth + ".partial"
        ts.filter_growth_csv(cfg["growth_df"], tmp, genotypes=keep)
        if args.drop_tubes:
            g = pd.read_csv(tmp)
            drop = pd.Series(False, index=g.index)
            for spec in args.drop_tubes:
                parts = re.split(r"[,:]", spec)
                if len(parts) != 3:
                    raise ValueError(f"--drop_tubes {spec!r}: expected "
                                     "condition_sel:replicate:t_sel")
                cond, rep, t_sel = parts
                drop |= ((g["condition_sel"].astype(str) == cond)
                         & (pd.to_numeric(g["replicate"]) == float(rep))
                         & (g["t_sel"].astype(float) == float(t_sel)))
            print(f"dropping {int(drop.sum())} rows ({', '.join(args.drop_tubes)})",
                  flush=True)
            g[~drop].to_csv(tmp, index=False)
        # in place only when complete: run_stage1 reuses an existing table
        os.replace(tmp, growth)
    cli.run_stage1(out, cfg, settings)
    cli._say("done")


if __name__ == "__main__":
    main()
