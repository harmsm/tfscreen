"""
diagnose_nan.py — NaN debugging tool for SVI optimization.

This script helps identify which variational parameters are collapsing toward
zero and causing NaN explosions during SVI. It loads a checkpoint, disables
JIT compilation, enables JAX NaN trapping, and runs the optimizer step-by-step
so that the exact operation that first produces a NaN raises an informative
traceback rather than silently contaminating all parameters via lax.scan.

Typical usage
-------------
After observing a "model exploded" RuntimeError during a long SVI run, locate
the most recent checkpoint file (e.g. ``tfs_checkpoint.pkl``) and run::

    tfs-diagnose-nan tfs_config.yaml tfs_checkpoint.pkl --num_steps 200

If a NaN occurs within num_steps, JAX will raise a FloatingPointError with a
traceback pointing to the first offending operation. The per-step printout of
near-zero scale parameters helps identify which hyperparameters are collapsing
even before the NaN appears.

Background
----------
The root cause of late-stage NaN explosions in SVI is typically a variational
scale parameter (e.g. ``*_hyper_loc_scale`` or ``*_hyper_scale_scale``) drifting
toward zero when the corresponding level of the hierarchy is poorly constrained
by data (e.g., all genotypes are spiked and there is no free library to pool
over). When a scale crosses float32 subnormal (~1e-38), ``1/scale`` in the
gradient overflows to ``inf``, which propagates backward through the ELBO as
``inf - inf = NaN``. The NaN typically surfaces in a different parameter than
the one that caused it, because the backward pass has already accumulated NaN
cotangents by the time it reaches the crash site.

The fix is to add a floor to those constraints via
``dist.constraints.greater_than(1e-4)`` instead of ``dist.constraints.positive``.
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import dill

from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.run_inference import RunInference


def diagnose_nan(config_file,
                 checkpoint_file,
                 seed=0,
                 num_steps=200,
                 near_zero_threshold=0.01):
    """
    Run SVI step-by-step from a checkpoint with JAX NaN trapping enabled.

    Disables JIT compilation so that JAX raises a FloatingPointError with a
    full traceback at the exact operation that first produces a NaN, rather
    than silently propagating it through a compiled lax.scan block.

    Each step prints the current loss and any variational scale parameters
    whose constrained value is below ``near_zero_threshold``. These are the
    likely culprits: a scale approaching zero causes ``1/scale`` to overflow
    in the gradient computation.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file for the model.
    checkpoint_file : str
        Path to the ``.pkl`` checkpoint file to resume from.
    seed : int, optional
        Random seed passed to RunInference (default 0). The seed only affects
        parameter initialization; the checkpoint state overrides it.
    num_steps : int, optional
        Number of optimizer steps to run (default 200). Each step processes
        one random mini-batch.
    near_zero_threshold : float, optional
        Scalar variational parameters whose constrained value falls below this
        threshold are printed each step (default 0.01).
    """

    print("Enabling JAX NaN trapping and disabling JIT.", flush=True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_disable_jit", True)

    gm, init_params = read_configuration(config_file)
    ri = RunInference(gm, seed=seed)

    svi_obj = ri.setup_svi(adam_step_size=1e-6, guide_type="component")

    # SVI requires init() to be called before update() so that constrain_fn
    # is wired up. We call it here and then immediately replace the returned
    # state with the checkpoint state.
    data_on_gpu = jax.device_put(ri.model.data)
    idx = ri.model.get_random_idx(num_batches=1)
    batch_data = ri.model.get_batch(data_on_gpu, idx)
    init_key = ri.get_key()
    _ = svi_obj.init(init_key,
                     init_params=init_params,
                     priors=ri.model.priors,
                     data=batch_data)

    print(f"Loading checkpoint from: {checkpoint_file}", flush=True)
    with open(checkpoint_file, "rb") as f:
        chk = dill.load(f)
    svi_state = chk["svi_state"]

    update_fn = svi_obj.update  # not JIT-compiled (jax_disable_jit is on)

    print(f"Running {num_steps} steps. Near-zero threshold: {near_zero_threshold}\n",
          flush=True)

    for i in range(num_steps):
        idx = ri.model.get_random_idx(num_batches=1)
        batch = ri.model.get_batch(data_on_gpu, idx)
        svi_state, loss = update_fn(svi_state,
                                    priors=ri.model.priors,
                                    data=batch)

        params = svi_obj.get_params(svi_state)

        # Report scalar scale parameters that are near zero
        near_zero = {k: float(v)
                     for k, v in params.items()
                     if v.ndim == 0 and float(v) < near_zero_threshold}

        print(f"Step {i:4d}: loss={loss:.4e}  near-zero scales: {near_zero}",
              flush=True)

    print("\nCompleted all steps without a NaN.", flush=True)
    print("If no NaN appeared, the checkpoint may already be past the point of",
          flush=True)
    print("collapse, or num_steps was too small. Try increasing num_steps or",
          flush=True)
    print("starting from an earlier checkpoint.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SVI step-by-step from a checkpoint with JAX NaN trapping "
            "enabled. Prints near-zero scale parameters each step and raises "
            "a FloatingPointError with a full traceback at the first NaN."
        )
    )
    parser.add_argument("config_file",
                        help="Path to the YAML configuration file.")
    parser.add_argument("checkpoint_file",
                        help="Path to the .pkl SVI checkpoint file.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for RunInference (default: 0).")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Number of optimizer steps to run (default: 200).")
    parser.add_argument("--near_zero_threshold", type=float, default=0.01,
                        help="Print scalar scale params below this value "
                             "(default: 0.01).")
    args = parser.parse_args()

    diagnose_nan(config_file=args.config_file,
                 checkpoint_file=args.checkpoint_file,
                 seed=args.seed,
                 num_steps=args.num_steps,
                 near_zero_threshold=args.near_zero_threshold)


if __name__ == "__main__":
    main()
