"""
Generate NumPyro reference posteriors for the PyTorch/Pyro port validation.

Run this script ONCE from the repo root before starting the port:

    NUMBA_DISABLE_JIT=1 python tests/fixtures/generate_reference_posteriors.py

Saves numpyro_reference_posteriors.npz to tests/fixtures/.
That file is checked in and used by Tier 3 statistical equivalence tests to
confirm the Pyro port produces compatible posterior distributions.

What is saved
-------------
- elbo_curve       : (n_steps,) float array of SVI losses
- {param}_mean     : posterior mean per key parameter
- {param}_std      : posterior std per key parameter
- {param}_lo95     : 2.5th percentile
- {param}_hi95     : 97.5th percentile
  where param in: activity, dk_geno, theta_low, theta_high, log_hill_K, hill_n
"""

import os
import sys
import numpy as np

# Make sure we're running from the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(REPO_ROOT, "tests", "smoke-tests", "test_data")
OUT_DIR   = os.path.join(REPO_ROOT, "tests", "fixtures")
OUT_FILE  = os.path.join(OUT_DIR, "numpyro_reference_posteriors.npz")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.run_inference import RunInference

GROWTH_CSV  = os.path.join(DATA_DIR, "growth-smoke.csv")
BINDING_CSV = os.path.join(DATA_DIR, "binding-smoke.csv")

# Inference settings — enough steps to give stable posterior statistics on
# the small smoke dataset without taking too long.
SEED        = 42
NUM_STEPS   = 2000
BATCH_SIZE  = None   # use all genotypes (smoke data is small)
NUM_SAMPLES = 500    # posterior samples to draw


def run():
    print("Initialising model …")
    model = ModelClass(
        growth_df=GROWTH_CSV,
        binding_df=BINDING_CSV,
        batch_size=BATCH_SIZE,
        condition_growth="linear",
        transformation="empirical",
        theta="hill",
        growth_transition="instant",
        dk_geno="hierarchical",
        activity="horseshoe",
        ln_cfu0="hierarchical",
        theta_growth_noise="zero",
        theta_binding_noise="zero",
    )

    print("Setting up SVI …")
    ri = RunInference(model=model, seed=SEED)
    svi = ri.setup_svi(adam_step_size=5e-3)

    out_root = os.path.join(OUT_DIR, "_tmp_ref")
    print(f"Running {NUM_STEPS} optimisation steps …")
    svi_state, params, converged = ri.run_optimization(
        svi=svi,
        max_num_epochs=NUM_STEPS,
        out_root=out_root,
        convergence_check_interval=100,
        checkpoint_interval=500,
    )
    print(f"  converged={converged}")

    # Read the saved loss curve
    loss_file = f"{out_root}_losses.bin"
    elbo_curve = np.fromfile(loss_file, dtype=np.float32) if os.path.exists(loss_file) else np.array([])

    print(f"Sampling {NUM_SAMPLES} posterior draws …")
    ri.get_posteriors(svi, svi_state, out_root,
                      num_posterior_samples=NUM_SAMPLES,
                      sampling_batch_size=NUM_SAMPLES)

    import h5py
    h5_path = f"{out_root}_posterior.h5"
    save_dict = {"elbo_curve": elbo_curve}

    with h5py.File(h5_path, "r") as f:
        for param in ["activity", "dk_geno",
                      "theta_theta_low", "theta_theta_high",
                      "theta_log_hill_K", "theta_hill_n"]:
            # Try exact key, then _auto_loc suffix used by MAP guides
            key = param if param in f else (f"{param}_auto_loc" if f"{param}_auto_loc" in f else None)
            if key is None:
                print(f"  WARNING: {param} not found in posteriors, skipping")
                continue
            arr = f[key][:]  # shape (num_samples, ...)
            save_dict[f"{param}_mean"]  = arr.mean(axis=0)
            save_dict[f"{param}_std"]   = arr.std(axis=0)
            save_dict[f"{param}_lo95"]  = np.percentile(arr, 2.5,  axis=0)
            save_dict[f"{param}_hi95"]  = np.percentile(arr, 97.5, axis=0)
            print(f"  {param}: mean={save_dict[param+'_mean'].mean():.4f}  "
                  f"std={save_dict[param+'_std'].mean():.4f}")

    np.savez(OUT_FILE, **save_dict)
    print(f"\nSaved reference posteriors → {OUT_FILE}")

    # Clean up temporary files
    for suffix in ["_losses.bin", "_losses.txt", "_posterior.h5", "_checkpoint.pkl"]:
        p = out_root + suffix
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    run()
