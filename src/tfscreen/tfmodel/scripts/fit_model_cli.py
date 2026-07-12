import os
import dill

import optax

from tfscreen.tfmodel.inference.run_inference import RunInference

from tfscreen.util.cli.generalized_main import generalized_main

from tfscreen.tfmodel.configuration_io import read_configuration


def _run_map(ri,
             init_params,
             checkpoint_file=None,
             out_prefix="tfs",
             adam_step_size=1e-3,
             adam_final_step_size=1e-6,
             adam_clip_norm=1.0,
             elbo_num_particles=2,
             convergence_tolerance=1e-5,
             convergence_window=10,
             patience=10,
             convergence_check_interval=2,
             checkpoint_interval=10,
             max_num_epochs=100000,
             init_param_jitter=0.0,
             epoch_checkpoint_interval=1000):
    """
    Run maximum a posteriori (MAP) optimization for hierarchical model inference.

    This function sets up and runs MAP optimization using the provided RunInference
    object and initial parameters. It manages checkpointing, convergence checking,
    and writes parameter values to disk. Returns the final optimizer state, parameters,
    and convergence status.

    Parameters
    ----------
    ri : RunInference
        RunInference object that manages model setup and optimization routines.
    init_params : dict
        Initial parameter values for MAP optimization.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume optimization from, or None to start fresh.
    out_prefix : str, optional
        Output file root for checkpoints and results (default "tfs").
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Final step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during MAP (default 2).
    max_num_epochs : int, optional
        Maximum number of MAP optimization epochs (default 100000).
    convergence_tolerance : float, optional
        Relative change in smoothed loss to declare MAP convergence (default 1e-5).
    convergence_window : int, optional
        Number of epochs to average for convergence check (default 10).
    patience : int, optional
        Number of consecutive checks meeting tolerance to declare convergence
        (default 10).
    convergence_check_interval : int, optional
        Frequency (in epochs) to check for convergence (default 2).
    checkpoint_interval : int, optional
        Steps between checkpoints and convergence checks (default 10).
    init_param_jitter : float, optional
        Amount of jitter to add to init_params (default 0.0).
    epoch_checkpoint_interval : int or None, optional
        Frequency (in epochs) to write numbered epoch checkpoints to a
        ``checkpoints/`` subdirectory (default 1000). Set to 0 or None to
        disable.

    Notes
    -----
    Posterior sampling is not performed here. Call ``tfs-sample-posterior``
    after fitting to draw posterior samples.

    Returns
    -------
    svi_state : Any
        Final optimizer state object from MAP.
    params : dict
        Final optimized parameters from MAP.
    converged : bool
        True if MAP converged according to the specified tolerance.
    """

    # Create a learning rate schedule
    schedule = optax.exponential_decay(
        init_value=adam_step_size,
        transition_steps=float(max_num_epochs * ri._iterations_per_epoch),
        decay_rate=adam_final_step_size / adam_step_size
    )

     # Create a maximum a posteriori svi object
    map_obj = ri.setup_svi(adam_step_size=schedule,
                           adam_clip_norm=adam_clip_norm,
                           elbo_num_particles=elbo_num_particles,
                           guide_type="delta")

    # Run MAP
    svi_state, params, converged = ri.run_optimization(
        map_obj,
        init_params=init_params,
        out_prefix=out_prefix,
        svi_state=checkpoint_file,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        patience=patience,
        convergence_check_interval=convergence_check_interval,
        checkpoint_interval=checkpoint_interval,
        max_num_epochs=max_num_epochs,
        init_param_jitter=init_param_jitter,
        epoch_checkpoint_interval=epoch_checkpoint_interval
    )

    # Write the current parameter values
    ri.write_params(params,out_prefix=out_prefix)

    # Write convergence information to stdout
    if converged:
        print("MAP run converged.",flush=True)
    else:
        print("MAP run has not yet converged.",flush=True)

    return svi_state, params, converged

def _run_svi(ri,
             init_params,
             checkpoint_file=None,
             out_prefix="tfs",
             adam_step_size=1e-3,
             adam_final_step_size=1e-6,
             adam_clip_norm=1.0,
             elbo_num_particles=2,
             convergence_tolerance=1e-5,
             convergence_window=10,
             patience=10,
             convergence_check_interval=2,
             checkpoint_interval=10,
             max_num_epochs=100000,
             init_param_jitter=0.1,
             epoch_checkpoint_interval=1000):
    """
    Run stochastic variational inference (SVI) for hierarchical model inference.

    This function sets up and runs SVI optimization using the provided RunInference
    object and initial parameters. It manages checkpointing, convergence checking,
    and optionally draws posterior samples after convergence.

    Parameters
    ----------
    ri : RunInference
        RunInference object that manages model setup and optimization routines.
    init_params : dict or None
        Initial parameter values for SVI optimization. If None, uses parameters
        captured during SVI object initialization.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume optimization from, or None to start fresh.
    out_prefix : str, optional
        Output file root for checkpoints and results (default "tfs").
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Final step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during SVI (default 2).
    convergence_tolerance : float, optional
        Relative change in loss to declare SVI convergence (default 1e-5).
    convergence_window : int, optional
        Number of epochs to average for convergence check (default 10).
    patience : int, optional
        Number of consecutive checks meeting tolerance to declare convergence
        (default 10).
    convergence_check_interval : int, optional
        Frequency (in epochs) to check for convergence (default 2).
    checkpoint_interval : int, optional
        Frequency (in epochs) between checkpoints (default 10).
    max_num_epochs : int, optional
        Maximum number of SVI optimization epochs (default 100000).
    init_param_jitter : float, optional
        Amount of jitter to add to init_params (default 0.1).
    epoch_checkpoint_interval : int or None, optional
        Frequency (in epochs) to write numbered epoch checkpoints to a
        ``checkpoints/`` subdirectory (default 1000). Set to 0 or None to
        disable.

    Notes
    -----
    Posterior sampling is not performed here. Call ``tfs-sample-posterior``
    after fitting to draw posterior samples.

    Returns
    -------
    svi_obj : Any
        The SVI optimizer object created by ``ri.setup_svi``.  Returned so
        that callers (e.g. ``tfs-sample-posterior``) can pass it directly to
        ``ri.get_posteriors`` without needing to re-create the guide.
    svi_state : Any
        Final optimizer state object from SVI.
    svi_params : dict
        Final optimized parameters from SVI.
    converged : bool
        True if SVI converged according to the specified tolerance.
    """

    # Create a learning rate schedule
    schedule = optax.exponential_decay(
        init_value=adam_step_size,
        transition_steps=float(max_num_epochs * ri._iterations_per_epoch),
        decay_rate=adam_final_step_size / adam_step_size
    )

    # Create an svi object
    svi_obj = ri.setup_svi(adam_step_size=schedule,
                           adam_clip_norm=adam_clip_norm,
                           elbo_num_particles=elbo_num_particles,
                           guide_type="component")

    # Run svi. Note that `init_params` is not used here, but is required by the
    # run_optimization method.
    svi_state, params, converged = ri.run_optimization(
        svi_obj,
        init_params=init_params,
        out_prefix=f"{out_prefix}",
        svi_state=checkpoint_file,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        patience=patience,
        convergence_check_interval=convergence_check_interval,
        checkpoint_interval=checkpoint_interval,
        max_num_epochs=max_num_epochs,
        init_param_jitter=init_param_jitter,
        epoch_checkpoint_interval=epoch_checkpoint_interval
    )

    # Write convergence information to stdout (skip when restoring from
    # checkpoint with no additional epochs — convergence is not meaningful).
    if max_num_epochs > 0:
        if converged:
            print("SVI run converged.",flush=True)
        else:
            print("SVI run has not yet converged.",flush=True)

    return svi_obj, svi_state, params, converged

def _run_nuts(ri,
              out_prefix="tfs",
              nuts_num_warmup=500,
              nuts_num_samples=500,
              nuts_num_chains=1,
              nuts_target_accept_prob=0.9,
              forward_batch_size=512):
    """
    Run NUTS (No-U-Turn Sampler) MCMC inference.

    Parameters
    ----------
    ri : RunInference
        RunInference object that manages model setup and MCMC routines.
    out_prefix : str, optional
        Output file root for checkpoints and results (default "tfs").
    nuts_num_warmup : int, optional
        Number of NUTS warmup steps (default 500).
    nuts_num_samples : int, optional
        Number of NUTS posterior samples (default 500).
    nuts_num_chains : int, optional
        Number of MCMC chains (default 1).
    nuts_target_accept_prob : float, optional
        Target acceptance probability for NUTS step-size adaptation (default 0.9).
    forward_batch_size : int, optional
        Number of genotypes to process per forward-model batch when computing
        posteriors (default 512).

    Returns
    -------
    mcmc_samples : dict
        Posterior samples as returned by ``mcmc.get_samples()``.
    """

    mcmc = ri.run_nuts(num_warmup=nuts_num_warmup,
                       num_samples=nuts_num_samples,
                       num_chains=nuts_num_chains,
                       target_accept_prob=nuts_target_accept_prob)

    mcmc_samples = mcmc.get_samples()

    # Save checkpoint
    tmp_checkpoint_file = f"{out_prefix}_checkpoint.tmp.pkl"
    checkpoint_file = f"{out_prefix}_checkpoint.pkl"
    with open(tmp_checkpoint_file, "wb") as f:
        dill.dump({"mcmc_samples": mcmc_samples}, f)
    os.replace(tmp_checkpoint_file, checkpoint_file)

    ri.get_nuts_posteriors(mcmc_samples,
                           out_prefix=out_prefix,
                           forward_batch_size=forward_batch_size)

    print("NUTS run complete.", flush=True)

    return mcmc_samples


def fit_model(config_file,
              seed=None,
              checkpoint_file=None,
              analysis_method="svi",
              out_prefix="tfs_fit_model",
              adam_step_size=1e-3,
              adam_final_step_size=1e-6,
              adam_clip_norm=1.0,
              elbo_num_particles=2,
              convergence_tolerance=1e-5,
              convergence_window=10,
              patience=10,
              convergence_check_interval=2,
              checkpoint_interval=10,
              max_num_epochs=100000,
              forward_batch_size=512,
              pre_map_num_epoch=1000,
              init_param_jitter=0.1,
              nuts_num_warmup=500,
              nuts_num_samples=500,
              nuts_num_chains=1,
              nuts_target_accept_prob=0.9,
              epoch_checkpoint_interval=1000):
    """
    Fit the joint hierarchical model using a previously generated configuration file.

    This function extracts estimates of transcription factor fractional occupancy (theta)
    and other latent parameters using Stochastic Variational Inference (SVI) or maximum a
    posteriori (MAP) approaches based on the config.

    Parameters
    ----------
    config_file : str
        Path to a YAML configuration file to load settings from.
    seed : int, optional
        Random seed for reproducibility. Must be provided if not loading from a checkpoint.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume SVI from, or None to start fresh.
    analysis_method : str, optional
        Method for inference. Allowed values are 'svi' (default), 'map', or 'nuts'.
        Case-insensitive. Posterior sampling is not performed; call
        ``tfs-sample-posterior`` after fitting.
    out_prefix : str, optional
        Prefix for all output files: checkpoints, parameter files, and the
        posterior HDF5 (default 'tfs_fit_model'). Files are named
        {out_prefix}_checkpoint.pkl, {out_prefix}_params.npz, etc.
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Final step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation (default 2).
    convergence_tolerance : float, optional
        Relative change in loss to declare SVI convergence (default 1e-5).
    convergence_window : int, optional
        Number of epochs to average for convergence check (default 10).
    patience : int, optional
        Number of consecutive checks meeting tolerance to declare convergence
        (default 10).
    convergence_check_interval : int, optional
        Frequency (in epochs) to check for convergence (default 2).
    checkpoint_interval : int, optional
        Frequency (in epochs) between checkpoints (default 10).
    max_num_epochs : int, optional
        Maximum number of SVI optimization epochs (default 100000).
    forward_batch_size : int, optional
        When getting NUTS posteriors, calculate forward predictions in batches
        of this size (default 512).
    pre_map_num_epoch : int, optional
        Number of MAP iterations to run prior to SVI (default 1000). Only used
        if analysis_method is 'svi'.
    init_param_jitter : float, optional
        Amount of jitter to add after the (optional) MAP run to break symmetry
        (default 0.1).
    nuts_num_warmup : int, optional
        Number of NUTS warmup steps (default 500). Only used if
        analysis_method is 'nuts'.
    nuts_num_samples : int, optional
        Number of NUTS posterior samples to draw (default 500). Only used if
        analysis_method is 'nuts'.
    nuts_num_chains : int, optional
        Number of MCMC chains (default 1). Only used if analysis_method is
        'nuts'.
    nuts_target_accept_prob : float, optional
        Target acceptance probability for NUTS step-size adaptation
        (default 0.9). Only used if analysis_method is 'nuts'.
    epoch_checkpoint_interval : int or None, optional
        Frequency (in epochs) to write numbered epoch checkpoints to a
        ``checkpoints/`` subdirectory alongside ``out_prefix`` (default 1000).
        Files are named ``{epoch:07d}_checkpoint.pkl``. Set to 0 or None to
        disable. Raises ``FileExistsError`` if a target file already exists.

    Returns
    -------
    state : Any
        Final state object (SVI state, MAP params, or NUTS samples dict).
    params : dict
        Final optimized or sampled parameters.
    converged : bool
        True if the run converged (always True for NUTS).
    """

    if seed is None and checkpoint_file is None:
        raise ValueError("seed must be provided unless loading from a checkpoint.")

    analysis_method = analysis_method.lower()

    # Check for existing results to avoid overwriting unless resuming
    if checkpoint_file is None:
        checkpoint_path = f"{out_prefix}_checkpoint.pkl"
        if os.path.exists(checkpoint_path):
            raise FileExistsError(
                f"Checkpoint file '{checkpoint_path}' already exists. To resume, "
                "provide this file as checkpoint_file. To overwrite, delete "
                "the file or change out_prefix."
            )

        if analysis_method == "svi" and pre_map_num_epoch > 0:
            premap_path = f"{out_prefix}_premap_checkpoint.pkl"
            if os.path.exists(premap_path):
                raise FileExistsError(
                    f"Premap checkpoint file '{premap_path}' already exists. To "
                    "overwrite, delete the file or change out_prefix."
                )

    orchestrator, init_params = read_configuration(config_file)

    # For posterior mode the seed is optional: the checkpoint restores the PRNG
    # key for SVI checkpoints, and any valid key works for MAP/Laplace sampling.
    effective_seed = seed if seed is not None else 0

    # Run SVI / MAP
    ri = RunInference(orchestrator, effective_seed)

    if analysis_method == "svi":
        if pre_map_num_epoch > 0 and checkpoint_file is None:
            _, init_params, _ = _run_map(ri,
                                         init_params=init_params,
                                         checkpoint_file=None,
                                         out_prefix=f"{out_prefix}_premap",
                                         adam_step_size=adam_step_size,
                                         adam_final_step_size=adam_final_step_size,
                                         adam_clip_norm=adam_clip_norm,
                                         elbo_num_particles=elbo_num_particles,
                                         convergence_tolerance=convergence_tolerance,
                                         convergence_window=convergence_window,
                                         patience=patience,
                                         checkpoint_interval=pre_map_num_epoch,
                                         max_num_epochs=pre_map_num_epoch,
                                         init_param_jitter=0.0,
                                         epoch_checkpoint_interval=None)

        return _run_svi(ri,
                        init_params=init_params,
                        checkpoint_file=checkpoint_file,
                        out_prefix=out_prefix,
                        adam_step_size=adam_step_size,
                        adam_final_step_size=adam_final_step_size,
                        adam_clip_norm=adam_clip_norm,
                        elbo_num_particles=elbo_num_particles,
                        convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        patience=patience,
                        convergence_check_interval=convergence_check_interval,
                        checkpoint_interval=checkpoint_interval,
                        max_num_epochs=max_num_epochs,
                        init_param_jitter=init_param_jitter,
                        epoch_checkpoint_interval=epoch_checkpoint_interval)

    elif analysis_method == "map":
        return _run_map(ri,
                        init_params=init_params,
                        checkpoint_file=checkpoint_file,
                        out_prefix=out_prefix,
                        adam_step_size=adam_step_size,
                        adam_final_step_size=adam_final_step_size,
                        adam_clip_norm=adam_clip_norm,
                        elbo_num_particles=elbo_num_particles,
                        convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        patience=patience,
                        convergence_check_interval=convergence_check_interval,
                        checkpoint_interval=checkpoint_interval,
                        max_num_epochs=max_num_epochs,
                        init_param_jitter=init_param_jitter,
                        epoch_checkpoint_interval=epoch_checkpoint_interval)

    elif analysis_method == "nuts":
        mcmc_samples = _run_nuts(ri,
                                 out_prefix=out_prefix,
                                 nuts_num_warmup=nuts_num_warmup,
                                 nuts_num_samples=nuts_num_samples,
                                 nuts_num_chains=nuts_num_chains,
                                 nuts_target_accept_prob=nuts_target_accept_prob,
                                 forward_batch_size=forward_batch_size)
        return None, mcmc_samples, True

    else:
        raise ValueError(
            f"analysis method '{analysis_method}' not recognized. This should "
            "be 'svi', 'map', or 'nuts'. To draw posteriors from an existing "
            "checkpoint, use tfs-sample-posterior."
        )

def main():
    return generalized_main(fit_model,
                            manual_arg_types={"config_file":str,
                                              "seed":int,
                                              "checkpoint_file":str,
                                              "pre_map_num_epoch":int,
                                              "init_param_jitter":float,
                                              "nuts_num_warmup":int,
                                              "nuts_num_samples":int,
                                              "nuts_num_chains":int,
                                              "nuts_target_accept_prob":float,
                                              "epoch_checkpoint_interval":int})

if __name__ == "__main__":
    main()
