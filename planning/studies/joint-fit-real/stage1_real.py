"""
Joint stage 1 on the real screen (README; planning/joint-fit-return.md
step 6): wt, spikes and singles, growth and binding together.

    python stage1_real.py GROWTH_CSV LIBRARY_CONFIG BINDING_CSV OUT_DIR --model plain|full
                          [--init_from OTHER/tfs_fit_model_params.npz]
                          [--drop_tubes COND:REP:T_SEL ...] [--congression_sets 50 20 5]
                          [--seed 1] [--hessian_chunk_size 8]

The recipe from planning/studies/joint-fit-sim: hill_geno (absolute theta,
shared by both markers), the congression mixture with the homodimer rule
(lambda prior 0.38 +- 0.15, the lab's measurement), binding at weight 1,
the count likelihood with level tube offsets (SD 0.17), log(n) population
SD 0.5, one k and m per condition with loose priors.

- plain: linear growth in theta, s = 1, binding as measured.
- full: plus a logistic warp of occupancy on both selection conditions,
  the growth titrant scale s (log s ~ Normal(0, 1)) and one binding offset
  per binding genotype (genotype_offset). Start it from the plain fit
  (--init_from): from scratch the full model can stop in a wrong mode
  (joint-fit-sim, runs 1-3).

Each step is skipped when its output exists, so a rerun resumes. theta is
predicted at the growth concentrations (tfs_pred_theta.csv) and at the
binding concentrations (tfs_pred_theta_binding.csv), both in binding units.
"""

import argparse
import os

import pandas as pd

from tfscreen.genetics.library_design import library_composition_table
from tfscreen.tfmodel import two_stage as ts
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.extract_params_cli import extract_params
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.scripts.predict_theta_cli import predict_theta
from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior

CONFIG = "tfs_configure_config.yaml"
CHECKPOINT = "tfs_fit_model_checkpoint.pkl"
LOOSE = dict(k_loc=0.015, k_scale=0.01, m_loc=0.0, m_scale=0.01)


def growth_priors(growth_csv, warp):
    """Loose k/m priors for every condition; m_warp on selection conditions."""
    conds = set()
    for chunk in pd.read_csv(growth_csv, usecols=["condition_pre", "condition_sel"],
                             chunksize=1_000_000):
        conds |= set(chunk["condition_pre"]) | set(chunk["condition_sel"])
    t = pd.DataFrame({"condition_rep": sorted(conds)}).assign(**LOOSE)
    if warp:
        t["m_warp"] = t["condition_rep"].str.contains(r"\+").astype(int)
    return t


def drop_rows(df, specs):
    """Drop the tubes named condition_sel:replicate:t_sel."""
    keep = pd.Series(True, index=df.index)
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"--drop_tubes {spec!r}: expected condition_sel:replicate:t_sel")
        cond, rep, t_sel = parts
        hit = ((df["condition_sel"] == cond)
               & (df["replicate"].astype(float) == float(rep))
               & (df["t_sel"].astype(float) == float(t_sel)))
        if not hit.any():
            raise ValueError(f"--drop_tubes {spec!r} matches no rows")
        keep &= ~hit
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("growth_csv")
    ap.add_argument("library_config")
    ap.add_argument("binding_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--model", choices=["full", "plain"], required=True)
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--drop_tubes", nargs="*", default=[])
    ap.add_argument("--congression_sets", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hessian_chunk_size", type=int, default=8)
    ap.add_argument("--max_num_epochs", type=int, default=None)
    args = ap.parse_args()
    growth_csv = os.path.abspath(args.growth_csv)
    library_config = os.path.abspath(args.library_config)
    binding_csv = os.path.abspath(args.binding_csv)
    init_from = None if args.init_from is None else os.path.abspath(args.init_from)
    full = args.model == "full"
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)

    if not os.path.exists("tfs_growth.csv"):
        lib = library_composition_table(library_config)
        keep = ts.stage1_genotypes(lib, ts.growth_genotypes(growth_csv))
        ts.filter_growth_csv(growth_csv, "tfs_growth.csv.partial", genotypes=keep)
        if args.drop_tubes:
            g = pd.read_csv("tfs_growth.csv.partial")
            n = len(g)
            g = drop_rows(g, args.drop_tubes)
            print(f"dropped {n - len(g)} rows ({', '.join(args.drop_tubes)})", flush=True)
            g.to_csv("tfs_growth.csv.partial", index=False)
        os.replace("tfs_growth.csv.partial", "tfs_growth.csv")
        print(f"stage 1: {len(keep)} genotypes (wt, spikes, singles)", flush=True)

    if not os.path.exists(CONFIG):
        growth_priors("tfs_growth.csv", warp=full).to_csv("growth_priors.csv", index=False)
        set_priors = ["sigma_fixed=0.17", "theta_log_hill_n_hyper_scale_fixed=0.5"]
        if full:
            set_priors.append("theta_growth_log_conc_shift_scale=1.0")
        extra = {} if args.congression_sets is None else {
            "congression_sets": list(args.congression_sets)}
        configure_model(
            binding_df=binding_csv,
            growth_df=os.path.abspath("tfs_growth.csv"),
            library_config=library_config,
            theta_model="hill_geno",
            transformation_model="mixture",
            transformation_lambda=[0.38, 0.15],
            congression_theta_rule="homodimer",
            theta_binding_noise_model="genotype_offset" if full else "zero",
            growth_shares_replicates=True,
            batch_size=4096,
            allow_missing_spikes=True,
            growth_priors=os.path.abspath("growth_priors.csv"),
            set_priors=set_priors,
            skip_model_stats=True,
            out_prefix="tfs_configure",
            **extra)

    fit_kw = {} if args.max_num_epochs is None else {"max_num_epochs": args.max_num_epochs}
    if init_from is not None:
        fit_kw["init_from"] = init_from
    if not os.path.exists("tfs_fit_model_params.npz"):
        fit_model(CONFIG, seed=args.seed, analysis_method="map", **fit_kw)
    if not os.path.exists("tfs_posterior.h5"):
        sample_posterior(CONFIG, CHECKPOINT, out_prefix="tfs_posterior",
                         num_posterior_samples=500, skip_growth_observations=True,
                         hessian_chunk_size=args.hessian_chunk_size)
    if not os.path.exists("tfs_params_provenance.json"):
        extract_params(CONFIG, "tfs_posterior.h5", out_prefix="tfs_params")
    if not os.path.exists("tfs_pred_theta.csv"):
        predict_theta(CONFIG, "tfs_posterior.h5", out_prefix="tfs_pred_theta")
    if not os.path.exists("tfs_pred_theta_binding.csv"):
        b = pd.read_csv(binding_csv)
        with open("binding_genotypes.txt", "w") as fh:
            fh.write("\n".join(sorted(b["genotype"].astype(str).unique())) + "\n")
        with open("binding_concs.txt", "w") as fh:
            fh.write("\n".join(str(c) for c in sorted(b["titrant_conc"].unique())) + "\n")
        with open("binding_titrants.txt", "w") as fh:
            fh.write("\n".join(sorted(b["titrant_name"].astype(str).unique())) + "\n")
        predict_theta(CONFIG, "tfs_posterior.h5", out_prefix="tfs_pred_theta_binding",
                      genotypes_file="binding_genotypes.txt",
                      titrant_names_file="binding_titrants.txt",
                      titrant_concs_file="binding_concs.txt", only_files=True)
    print("stage 1 done", flush=True)


if __name__ == "__main__":
    main()
