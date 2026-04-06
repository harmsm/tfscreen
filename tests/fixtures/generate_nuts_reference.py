"""
Generate NumPyro NUTS reference posteriors from the gold-standard dataset.

Run ONCE from the repo root (with the NumPyro environment active):

    NUMBA_DISABLE_JIT=1 python tests/fixtures/generate_nuts_reference.py

Saves tests/fixtures/nuts_gold_standard_reference.json.  Commit that file so
the Pyro port's statistical-equivalence smoke test can load it without
needing NumPyro installed.

Gold-standard data
------------------
notebooks/development-tools/growth_gold-standared.csv
notebooks/development-tools/binding_gold-standared.csv

These contain 9 genotypes (wt + 8 single-point mutants) extracted from a
full KanR screen.

What is saved
-------------
For each parameter in [activity, dk_geno, theta_theta_low, theta_theta_high,
theta_log_hill_K, theta_hill_n]:
  - per-genotype (or per-genotype-titrant) median, lower_95, upper_95

Also saved:
  - num_warmup, num_samples, num_chains, seed (for reproducibility)
  - num_divergences
"""

import json
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(REPO_ROOT, "notebooks", "development-tools")
OUT_FILE  = os.path.join(REPO_ROOT, "tests", "fixtures", "nuts_gold_standard_reference.json")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

GROWTH_CSV  = os.path.join(DATA_DIR, "growth_gold-standared.csv")
BINDING_CSV = os.path.join(DATA_DIR, "binding_gold-standared.csv")

# NUTS settings — enough to give a stable posterior on 9 genotypes
SEED         = 1242
NUM_WARMUP   = 500
NUM_SAMPLES  = 500
NUM_CHAINS   = 1


def _df_to_dict(df):
    """Serialise a DataFrame with a 'genotype' column to a nested dict."""
    label_col = "genotype" if "genotype" in df.columns else df.columns[0]
    out = {}
    for _, row in df.iterrows():
        key = str(row[label_col])
        if "titrant_name" in row.index:
            key = f"{row['genotype']}|{row['titrant_name']}"
        out[key] = {
            "median":   float(row["median"]),
            "lower_95": float(row["lower_95"]),
            "upper_95": float(row["upper_95"]),
        }
    return out


def run():
    import jax
    import jax.numpy as jnp

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    from tfscreen.analysis.hierarchical.run_inference import RunInference
    from tfscreen.analysis.hierarchical.growth_model.extraction import extract_parameters

    print(f"Loading data from {DATA_DIR} …")
    import pandas as pd
    growth_df  = pd.read_csv(GROWTH_CSV)
    binding_df = pd.read_csv(BINDING_CSV)

    print(f"  {len(growth_df)} growth rows, "
          f"{growth_df['genotype'].nunique()} genotypes")

    # Use a config chosen for good MCMC mixing rather than production accuracy.
    # Horseshoe activity and empirical transformation create multimodal posteriors
    # that mix poorly in NUTS with small datasets — independent chains land in
    # different modes, making coverage comparisons unreliable.
    # hierarchical activity (Normal hyperprior) and single transformation mix well.
    print("Initialising model (NUTS-friendly settings) …")
    model = ModelClass(
        growth_df=growth_df,
        binding_df=binding_df,
        activity="hierarchical",
        transformation="single",
    )

    ri = RunInference(model=model, seed=SEED)

    print(f"Running NUTS: {NUM_WARMUP} warmup + {NUM_SAMPLES} samples × {NUM_CHAINS} chain(s) …")
    mcmc = ri.run_nuts(
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
    )

    # Count divergences
    extra = mcmc.get_extra_fields()
    divergences = extra.get("diverging", None)
    num_div = int(jnp.sum(divergences)) if divergences is not None else 0
    print(f"  divergences: {num_div}")

    posteriors = mcmc.get_samples()
    # Convert JAX arrays to numpy for extract_parameters
    posteriors_np = {k: np.asarray(v) for k, v in posteriors.items()}

    print("Extracting parameter summaries …")
    params = extract_parameters(
        model, posteriors_np,
        q_to_get={"median": 0.5, "lower_95": 0.025, "upper_95": 0.975}
    )

    reference = {
        "metadata": {
            "seed":        SEED,
            "num_warmup":  NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
            "num_chains":  NUM_CHAINS,
            "num_divergences": num_div,
            "model_config": {
                "condition_growth": "linear",
                "dk_geno":          "hierarchical",
                "activity":         "hierarchical",
                "theta":            "hill",
                "transformation":   "single",
                "theta_growth_noise": "zero",
                "theta_binding_noise": "zero",
            },
        },
        "parameters": {},
    }

    for param_name, df in params.items():
        reference["parameters"][param_name] = _df_to_dict(df)
        med = df["median"].values
        print(f"  {param_name}: median range [{med.min():.4f}, {med.max():.4f}]")

    with open(OUT_FILE, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nSaved reference → {OUT_FILE}")


if __name__ == "__main__":
    run()
