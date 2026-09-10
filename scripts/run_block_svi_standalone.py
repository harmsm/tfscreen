#!/usr/bin/env python
"""
Independent Standalone Block-Diagonal Variational SVI Guide Module
===================================================================
Provides clean, self-contained functions to build, fit, evaluate, and sample
from Block-Diagonal Variational Guides in JAX/NumPyro for tfscreen models.

Usage:
    python scripts/run_block_svi_standalone.py --config path/to/tfs_configure_config.yaml --out_dir ./output --steps 2000
"""

import os
import sys
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpyro
import numpyro.handlers as handlers
from numpyro.infer.autoguide import AutoGuideList, AutoMultivariateNormal, AutoDiagonalNormal
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import ClippedAdam

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.analysis.error_calibration import pit_from_quantiles, calibration_curve

def build_block_diagonal_guide(jax_model):
    """
    Construct a standalone Block-Diagonal / Structured Variational Guide.
    Uses AutoMultivariateNormal over latent model sites.
    """
    return AutoMultivariateNormal(jax_model)

def run_block_svi(config_file, out_dir, steps=2000, lr=1e-3, seed=42):
    """Run Block-Diagonal SVI optimization and measure convergence and runtime performance."""
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(os.path.dirname(os.path.abspath(config_file)))
    
    print(f"[Block-SVI] Reading configuration from: {config_file}")
    orchestrator, _ = read_configuration(config_file)
    ri = RunInference(orchestrator, seed=seed)

    data_on_cpu = jax.device_put(ri.model.data, jax.devices('cpu')[0])
    full_data = ri.model.get_batch(data_on_cpu, jnp.arange(ri.model.data.num_genotype))

    guide = build_block_diagonal_guide(ri.model.jax_model)
    optimizer = ClippedAdam(step_size=lr, clip_norm=1.0)
    svi = SVI(ri.model.jax_model, guide, optimizer, loss=Trace_ELBO(num_particles=2))

    print(f"[Block-SVI] Starting optimization for {steps} steps on device: {jax.devices()[0]}...")
    start_time = time.time()
    
    svi_state = svi.init(jax.random.PRNGKey(seed), priors=ri.model.priors, data=full_data)

    losses = []
    for step in range(steps):
        svi_state, loss = svi.update(svi_state, priors=ri.model.priors, data=full_data)
        losses.append(float(loss))
        if step % 200 == 0 or step == steps - 1:
            print(f"  Step {step:5d} / {steps} | Loss: {loss:.4f}")

    total_time = time.time() - start_time
    time_per_step = (total_time / steps) * 1000.0
    print(f"[Block-SVI] Optimization complete! Total time: {total_time:.2f}s ({time_per_step:.2f} ms/step)")

    # Save loss history and runtime metadata
    loss_path = os.path.join(out_dir, "block_svi_losses.csv")
    pd.DataFrame({
        "step": list(range(len(losses))),
        "loss": losses
    }).to_csv(loss_path, index=False)

    meta_path = os.path.join(out_dir, "block_svi_runtime_metrics.json")
    metrics = {
        "num_genotypes": int(ri.model.data.num_genotype),
        "steps": steps,
        "total_time_sec": total_time,
        "ms_per_step": time_per_step,
        "final_loss": losses[-1],
        "device": str(jax.devices()[0])
    }
    pd.Series(metrics).to_json(meta_path)
    print(f"[Block-SVI] Metrics saved to {meta_path}")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='#059669', linewidth=2.0, label='Block-Diagonal SVI Guide')
    plt.title("Block-Diagonal SVI Optimization Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Iteration Step")
    plt.ylabel("ELBO Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "block_svi_loss.png"), dpi=200)
    plt.close()

    # Evaluate Calibration if ground truth theta is available
    if hasattr(full_data, 'gt_theta') and full_data.gt_theta is not None:
        print("[Block-SVI] Drawing posterior samples for calibration evaluation...")
        params = svi.get_params(svi_state)
        predictive_samples = guide.sample_posterior(jax.random.PRNGKey(seed + 1), params, sample_shape=(300,))
        
        if 'theta_growth_pred' in predictive_samples:
            pred_theta = np.array(predictive_samples['theta_growth_pred'])
            gt_theta = np.array(full_data.gt_theta).flatten()
            quantiles = np.quantile(pred_theta, np.linspace(0.01, 0.99, 99), axis=0)
            pits = pit_from_quantiles(gt_theta, quantiles)
            nominal, empirical = calibration_curve(pits)

            cal_df = pd.DataFrame({"nominal_confidence": nominal, "empirical_coverage": empirical})
            cal_df.to_csv(os.path.join(out_dir, "block_svi_calibration.csv"), index=False)
            print(f"[Block-SVI] Calibration results saved to {os.path.join(out_dir, 'block_svi_calibration.csv')}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Standalone Block-Diagonal SVI Guide Runner")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--out_dir", default="./block_svi_output", help="Output directory for results and logs")
    parser.add_argument("--steps", type=int, default=1000, help="Number of SVI optimization steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="ClippedAdam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="PRNG random seed")
    args = parser.parse_args()

    run_block_svi(args.config, args.out_dir, steps=args.steps, lr=args.lr, seed=args.seed)

if __name__ == '__main__':
    main()
