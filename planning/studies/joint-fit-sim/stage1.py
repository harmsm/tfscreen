"""
Joint stage 1 on a simulation (README): wt, spikes and singles, growth and
binding together.

    python stage1.py SIM_DIR OUT_DIR --model full|plain [--seed 1] [--max_num_epochs N]
                     [--init_from OTHER_FIT/tfs_fit_model_params.npz]
                     [--pin_km_truth]

SIM_DIR holds a tfs-simulate run of make_sim.py's config (simulate_config.yaml,
tfs_sim_growth.csv, tfs_sim_binding.csv). Each step is skipped when its
output exists, so a rerun resumes.

Both models: hill_geno (absolute theta, shared by both markers), the
congression mixture with the homodimer rule (lambda prior 0.357 +- 0.15),
binding at weight 1, the count likelihood with level tube offsets
(SD 0.17), log(n) population SD 0.5, one k and m per condition, loose
k/m priors (growth_priors_loose.csv's values).

- full: plus a logistic warp of occupancy on both selection conditions,
  the growth titrant scale s (log s ~ Normal(0, 1)) and one binding offset
  per binding genotype (genotype_offset).
- plain: none of those (linear growth in theta, s = 1, binding as
  measured).

--init_from starts the MAP at another fit's MAP (tfs-fit-model --init_from;
not staged): run 1's full fit on the null simulation stopped in a wrong
mode about 1,000 nats above the plain fit, so the full model is started
from the plain one, its extra sites (warps, s, offsets) at their guesses.

--pin_km_truth pins every condition's k and m at the simulation's truth
(tfs_sim_growth_parameters.csv; growth_priors k_pinned/m_pinned): run 3's
fits put m 15-25% steep even with the true model, and the loss at the
true k and m against the free fit's says whether stage 1 prefers that m
or only drifts along a flat direction.
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


def growth_priors(growth_csv, warp, truth=None):
    """
    Loose k/m priors for every condition; m_warp on selection conditions;
    with ``truth`` (a growth-parameters table), k and m pinned at it.
    """
    conds = set()
    for chunk in pd.read_csv(growth_csv, usecols=["condition_pre", "condition_sel"],
                             chunksize=1_000_000):
        conds |= set(chunk["condition_pre"]) | set(chunk["condition_sel"])
    t = pd.DataFrame({"condition_rep": sorted(conds)}).assign(**LOOSE)
    if warp:
        t["m_warp"] = t["condition_rep"].str.contains(r"\+").astype(int)
    if truth is not None:
        tr = truth.set_index("condition_rep")
        missing = sorted(set(t["condition_rep"]) - set(tr.index))
        if missing:
            raise ValueError(f"no growth truth for {missing}")
        t["k_loc"] = t["condition_rep"].map(tr["growth_k"])
        t["m_loc"] = t["condition_rep"].map(tr["growth_m"])
        t["k_pinned"] = 1
        t["m_pinned"] = 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", choices=["full", "plain"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_num_epochs", type=int, default=None)
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--pin_km_truth", action="store_true")
    args = ap.parse_args()
    sim = os.path.abspath(args.sim_dir)
    init_from = None if args.init_from is None else os.path.abspath(args.init_from)
    full = args.model == "full"
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)

    library_config = os.path.join(sim, "simulate_config.yaml")
    if not os.path.exists("tfs_growth.csv"):
        lib = library_composition_table(library_config)
        src = os.path.join(sim, "tfs_sim_growth.csv")
        keep = ts.stage1_genotypes(lib, ts.growth_genotypes(src))
        ts.filter_growth_csv(src, "tfs_growth.csv.partial", genotypes=keep)
        os.replace("tfs_growth.csv.partial", "tfs_growth.csv")
        print(f"stage 1: {len(keep)} genotypes (wt, spikes, singles)", flush=True)

    if not os.path.exists(CONFIG):
        truth = (pd.read_csv(os.path.join(sim, "tfs_sim_growth_parameters.csv"))
                 if args.pin_km_truth else None)
        growth_priors("tfs_growth.csv", warp=full, truth=truth).to_csv(
            "growth_priors.csv", index=False)
        set_priors = ["sigma_fixed=0.17", "theta_log_hill_n_hyper_scale_fixed=0.5"]
        if full:
            set_priors.append("theta_growth_log_conc_shift_scale=1.0")
        configure_model(
            binding_df=os.path.join(sim, "tfs_sim_binding.csv"),
            growth_df=os.path.abspath("tfs_growth.csv"),
            library_config=library_config,
            theta_model="hill_geno",
            transformation_model="mixture",
            transformation_lambda=[0.357, 0.15],
            congression_theta_rule="homodimer",
            theta_binding_noise_model="genotype_offset" if full else "zero",
            growth_shares_replicates=True,
            batch_size=4096,
            allow_missing_spikes=True,
            growth_priors=os.path.abspath("growth_priors.csv"),
            set_priors=set_priors,
            skip_model_stats=True,
            out_prefix="tfs_configure")

    fit_kw = {} if args.max_num_epochs is None else {"max_num_epochs": args.max_num_epochs}
    if init_from is not None:
        fit_kw["init_from"] = init_from
    if not os.path.exists("tfs_fit_model_params.npz"):
        fit_model(CONFIG, seed=args.seed, analysis_method="map", **fit_kw)
    if not os.path.exists("tfs_posterior.h5"):
        sample_posterior(CONFIG, CHECKPOINT, out_prefix="tfs_posterior",
                         num_posterior_samples=500, skip_growth_observations=True)
    if not os.path.exists("tfs_params_provenance.json"):
        extract_params(CONFIG, "tfs_posterior.h5", out_prefix="tfs_params")
    if not os.path.exists("tfs_pred_theta.csv"):
        predict_theta(CONFIG, "tfs_posterior.h5", out_prefix="tfs_pred_theta")
    print("stage 1 done", flush=True)


if __name__ == "__main__":
    main()
