import os
import dill
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import _run_svi
from tfscreen.util.cli.generalized_main import generalized_main


def sample_posterior(config_file,
                     checkpoint_file,
                     out_prefix="tfs_posterior",
                     seed=0,
                     num_posterior_samples=10000,
                     sampling_batch_size=100,
                     forward_batch_size=512):
    """
    Draw posterior samples from an existing MAP, SVI, or NUTS checkpoint.

    Three checkpoint types are handled automatically:

    1. NUTS checkpoint: runs the forward model over the saved MCMC samples and
       writes posterior predictives.
    2. MAP checkpoint (AutoDelta guide): forms a Laplace (Gaussian) approximation
       via the Hessian of the log-joint at the MAP point, then draws samples.
    3. SVI checkpoint (component guide): resumes the fitted guide with 0 additional
       epochs and draws posterior samples directly.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file used when fitting the model.
    checkpoint_file : str
        Path to the checkpoint .pkl file produced by tfs-growth-analysis or
        tfs-prefit-calibration.
    out_prefix : str, optional
        Prefix for the posterior output file (default 'tfs_posterior').
        Posterior samples are written to {out_prefix}.h5.
    seed : int, optional
        Random seed used when constructing the RunInference object (default 0).
        For SVI and MAP checkpoints the PRNG state is restored from the
        checkpoint, so this value has no effect on the posterior samples.
    num_posterior_samples : int, optional
        Number of posterior samples to draw (default 10000). Not used for NUTS
        checkpoints (all MCMC samples are used directly).
    sampling_batch_size : int, optional
        Parameter sampling batch size (default 100). Not used for NUTS.
    forward_batch_size : int, optional
        Forward-model batch size for posterior predictives (default 512).
    """
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{checkpoint_file}'. "
            "Run tfs-growth-analysis first to produce a checkpoint."
        )

    gm, init_params = read_configuration(config_file)
    ri = RunInference(gm, seed)

    with open(checkpoint_file, "rb") as f:
        chk_data = dill.load(f)

    # RunInference methods write {out_prefix}_posterior.h5; rename to {out_prefix}.h5
    # after each call so the output matches the documented convention.
    ri_prefix = f"{out_prefix}_tmp_posterior"

    if "mcmc_samples" in chk_data:
        # NUTS checkpoint: regenerate posteriors from saved samples.
        print("Detected NUTS checkpoint. Writing posterior predictives...", flush=True)
        ri.get_nuts_posteriors(chk_data["mcmc_samples"],
                               out_prefix=ri_prefix,
                               forward_batch_size=forward_batch_size)
    else:
        temp_svi = ri.setup_svi(guide_type="delta")
        chk_params = temp_svi.optim.get_params(chk_data["svi_state"].optim_state)

        if any("_auto_loc" in k for k in chk_params):
            # MAP checkpoint: Hessian-based Laplace approximation.
            print("Detected MAP checkpoint. Drawing Laplace posterior samples...", flush=True)
            ri.get_laplace_posteriors(
                map_params=chk_params,
                out_prefix=ri_prefix,
                num_posterior_samples=num_posterior_samples,
                sampling_batch_size=sampling_batch_size,
                forward_batch_size=forward_batch_size,
            )
        else:
            # SVI checkpoint: resume with 0 epochs, draw samples directly.
            print("Detected SVI checkpoint. Drawing variational posterior samples...", flush=True)
            _run_svi(ri,
                     init_params=init_params,
                     checkpoint_file=checkpoint_file,
                     out_prefix=ri_prefix,
                     max_num_epochs=0,
                     num_posterior_samples=num_posterior_samples,
                     sampling_batch_size=sampling_batch_size,
                     forward_batch_size=forward_batch_size,
                     always_get_posterior=True,
                     # Convergence / optimizer kwargs are unused at 0 epochs but
                     # required by _run_svi's signature; use neutral defaults.
                     adam_step_size=1e-3,
                     adam_final_step_size=1e-6,
                     adam_clip_norm=1.0,
                     elbo_num_particles=2,
                     convergence_tolerance=0.01,
                     convergence_window=10,
                     patience=10,
                     convergence_check_interval=2,
                     checkpoint_interval=10,
                     init_param_jitter=0.0,
                     epoch_checkpoint_interval=None)

    src = f"{ri_prefix}_posterior.h5"
    dst = f"{out_prefix}.h5"
    os.rename(src, dst)
    print(f"Posterior samples written to {dst}", flush=True)


def main():
    generalized_main(sample_posterior,
                     manual_arg_types={"seed": int,
                                       "num_posterior_samples": int,
                                       "sampling_batch_size": int,
                                       "forward_batch_size": int})


if __name__ == "__main__":
    main()
