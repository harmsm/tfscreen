import os
import dill

from tfscreen.tfmodel.inference.run_inference import (
    RunInference,
    check_guide_kwargs,
    resolve_guide_type,
)

from tfscreen.util.cli.generalized_main import generalized_main

from tfscreen.tfmodel.configuration_io import read_configuration

# Upper bound on the component guide's starting scales (the numpyro autoguides'
# own default init_scale).  Starting at prior width makes early SVI escape the
# guide variance by inflating the noise terms.
DEFAULT_COMPONENT_INIT_SCALE = 0.1

def _optimization_kwargs(convergence_window_steps=2000,
                         patience=3,
                         convergence_z=3.0,
                         loss_rtol=1e-6,
                         param_tolerance=0.05,
                         adam_final_step_size=1e-6,
                         adam_step_size_cut=0.1,
                         checkpoint_interval=10,
                         max_num_epochs=100000,
                         init_param_jitter=0.0,
                         epoch_checkpoint_interval=1000):
    """``RunInference.run_optimization`` keyword arguments from CLI names."""
    return dict(convergence_window_steps=convergence_window_steps,
                patience=patience,
                convergence_z=convergence_z,
                loss_rtol=loss_rtol,
                param_tolerance=param_tolerance,
                final_step_size=adam_final_step_size,
                step_size_cut=adam_step_size_cut,
                checkpoint_interval=checkpoint_interval,
                max_num_epochs=max_num_epochs,
                init_param_jitter=init_param_jitter,
                epoch_checkpoint_interval=epoch_checkpoint_interval)


def _run_map(ri,
             init_values=None,
             checkpoint_file=None,
             out_prefix="tfs",
             adam_step_size=1e-3,
             adam_clip_norm=None,
             elbo_num_particles=2,
             label="MAP",
             **optimization_kwargs):
    """
    Run maximum a posteriori (MAP) optimization (an AutoDelta guide).

    Parameters
    ----------
    ri : RunInference
        RunInference object that manages model setup and optimization routines.
    init_values : dict or None, optional
        Constrained site values the MAP starts from (``ri.site_values``).
        Sites without a value start at the prior median.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume optimization from, or None to start fresh.
    out_prefix : str, optional
        Output file root for checkpoints and results (default "tfs").
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_clip_norm : float or None, optional
        Clip each gradient element to +/- this value (numpyro ClippedAdam).
        None (default) disables clipping: when gradients are much larger
        than the clip, an elementwise clip makes Adam follow the gradient's
        sign, and its fixed point is biased by rare large ELBO penalties.
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during MAP (default 2).
    label : str, optional
        Name of the run in the closing message (default "MAP").
    **optimization_kwargs
        Passed to ``ri.run_optimization`` (see ``_optimization_kwargs``).

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
        True if the run converged (see ``RunInference.run_optimization``).
    """

    # Create a maximum a posteriori svi object
    map_obj = ri.setup_svi(adam_step_size=adam_step_size,
                           adam_clip_norm=adam_clip_norm,
                           elbo_num_particles=elbo_num_particles,
                           guide_type="delta",
                           init_values=init_values)

    # Run MAP
    svi_state, params, converged = ri.run_optimization(
        map_obj,
        out_prefix=out_prefix,
        svi_state=checkpoint_file,
        **optimization_kwargs
    )

    # Write the current parameter values
    ri.write_params(params,out_prefix=out_prefix)

    # Write convergence information to stdout
    if converged:
        print(f"{label} run converged.",flush=True)
    else:
        print(f"{label} run has not yet converged.",flush=True)

    return svi_state, params, converged

def _run_svi(ri,
             init_params,
             checkpoint_file=None,
             out_prefix="tfs",
             adam_step_size=1e-3,
             adam_clip_norm=None,
             elbo_num_particles=2,
             guide_type="component",
             guide_kwargs=None,
             init_values=None,
             **optimization_kwargs):
    """
    Run stochastic variational inference (SVI) for hierarchical model inference.

    Parameters
    ----------
    ri : RunInference
        RunInference object that manages model setup and optimization routines.
    init_params : dict or None
        Initial component-guide parameters (``ri.component_guide_start``).
        Not used by autoguides, whose parameter names differ.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume optimization from, or None to start fresh.
    out_prefix : str, optional
        Output file root for checkpoints and results (default "tfs").
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_clip_norm : float or None, optional
        Clip each gradient element to +/- this value (numpyro ClippedAdam).
        None (default) disables clipping: when gradients are much larger
        than the clip, an elementwise clip makes Adam follow the gradient's
        sign, and its fixed point is biased by rare large ELBO penalties.
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during SVI (default 2).
    guide_type : str, optional
        Variational family passed to ``ri.setup_svi`` (default 'component').
    guide_kwargs : dict or None, optional
        Autoguide options (``rank``, ``init_scale``) passed to ``ri.setup_svi``.
    init_values : dict or None, optional
        Constrained site values an autoguide's location starts from.
    **optimization_kwargs
        Passed to ``ri.run_optimization`` (see ``_optimization_kwargs``).

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
        True if the run converged (see ``RunInference.run_optimization``).
    """

    # Create an svi object
    svi_obj = ri.setup_svi(adam_step_size=adam_step_size,
                           adam_clip_norm=adam_clip_norm,
                           elbo_num_particles=elbo_num_particles,
                           guide_type=guide_type,
                           guide_kwargs=guide_kwargs,
                           init_values=init_values)

    # init_params are substituted by parameter name, which only the component
    # guide's names can match; an autoguide starts from init_values instead.
    if resolve_guide_type(guide_type) != "component":
        init_params = None

    svi_state, params, converged = ri.run_optimization(
        svi_obj,
        init_params=init_params,
        out_prefix=f"{out_prefix}",
        svi_state=checkpoint_file,
        **optimization_kwargs
    )

    # Write convergence information to stdout (skip when restoring from
    # checkpoint with no additional epochs — convergence is not meaningful).
    if optimization_kwargs.get("max_num_epochs", 1) > 0:
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


def _check_checkpoint_guide(checkpoint_file, guide_type):
    """Refuse to resume an SVI checkpoint with a different guide."""

    # A missing file is reported by run_optimization when it tries to restore.
    if not os.path.isfile(checkpoint_file):
        return

    with open(checkpoint_file, "rb") as f:
        saved = dill.load(f).get("guide_type") or "component"
    if saved != guide_type:
        raise ValueError(
            f"Checkpoint '{checkpoint_file}' was written with guide_type "
            f"'{saved}', but guide_type '{guide_type}' was requested. Resume "
            f"with guide_type='{saved}'."
        )


def fit_model(config_file,
              seed=None,
              checkpoint_file=None,
              analysis_method="svi",
              guide_type="component",
              guide_rank=None,
              guide_init_scale=None,
              out_prefix="tfs_fit_model",
              adam_step_size=1e-3,
              adam_final_step_size=1e-6,
              adam_step_size_cut=0.1,
              adam_clip_norm=None,
              elbo_num_particles=2,
              convergence_window_steps=2000,
              patience=3,
              convergence_z=3.0,
              loss_rtol=1e-6,
              param_tolerance=0.05,
              checkpoint_interval=10,
              max_num_epochs=100000,
              forward_batch_size=512,
              pre_map_num_epoch=10000,
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

    Optimization (SVI, MAP and the pre-MAP warm-up) runs in windows of
    ``convergence_window_steps`` optimizer steps.  After ``patience``
    consecutive windows without a significant improvement in the loss
    (against its own noise), the step size is cut by ``adam_step_size_cut``.
    Once it has reached ``adam_final_step_size``, the run stops after
    ``patience`` consecutive windows in which the loss has not improved and no
    parameter has moved (against its posterior or prior width).  Each window
    is recorded in ``{out_prefix}_convergence.csv``.

    Parameters
    ----------
    config_file : str
        Path to a YAML configuration file to load settings from.
    seed : int, optional
        Random seed for reproducibility. Must be provided if not loading from a checkpoint.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume SVI from, or None to start fresh.
        A resumed run continues at the checkpoint's step size and convergence
        stage.
    analysis_method : str, optional
        Method for inference. Allowed values are 'svi' (default), 'map', or 'nuts'.
        Case-insensitive. Posterior sampling is not performed; call
        ``tfs-sample-posterior`` after fitting.
    guide_type : str, optional
        Variational family for analysis_method 'svi' (default 'component',
        the guide assembled from the model components).  Also accepts the
        numpyro autoguides 'auto_normal', 'auto_diagonal_normal',
        'auto_multivariate_normal', 'auto_low_rank_multivariate_normal' and
        'delta'; numpyro class names such as 'AutoNormal' work too.  Every
        guide's location starts at the pre-MAP solution when
        ``pre_map_num_epoch > 0``, otherwise at the configured guesses.
        A resumed checkpoint must use the guide it was written with.
    guide_rank : int, optional
        Covariance rank for 'auto_low_rank_multivariate_normal' (numpyro's
        default when omitted).
    guide_init_scale : float, optional
        Initial scale of the variational distribution.  For 'component' it
        caps every guide scale at the start (default 0.1); for an autoguide
        it is numpyro's ``init_scale`` (numpyro's default when omitted).  Not
        accepted by 'delta'.
    out_prefix : str, optional
        Prefix for all output files: checkpoints, parameter files, and the
        posterior HDF5 (default 'tfs_fit_model'). Files are named
        {out_prefix}_checkpoint.pkl, {out_prefix}_params.npz, etc.
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Smallest step size (default 1e-6).  Set equal to ``adam_step_size``
        for a constant step size.
    adam_step_size_cut : float, optional
        Factor applied to the step size at each cut (default 0.1).
    adam_clip_norm : float or None, optional
        Clip each gradient element to +/- this value (numpyro ClippedAdam).
        None (default) disables clipping: when gradients are much larger
        than the clip, an elementwise clip makes Adam follow the gradient's
        sign, and its fixed point is biased by rare large ELBO penalties.
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation (default 2).
    convergence_window_steps : int, optional
        Optimizer steps per convergence window (default 2000; at least 10
        epochs, so with mini-batching every genotype is visited several times
        per window).
    patience : int, optional
        Consecutive windows without loss improvement required for a step-size
        cut, and (at the final step size) without loss improvement or
        parameter movement required for stopping (default 3).
    convergence_z : float, optional
        Standard errors of change attributed to noise (default 3).
    loss_rtol : float, optional
        Loss changes per window below this fraction of the loss count as no
        change (default 1e-6); matters only for (near-)deterministic losses.
    param_tolerance : float, optional
        Parameter movement per window, beyond noise, still counted as no
        change (default 0.05), in posterior SDs (SVI), prior SDs (MAP), or
        log/logit units (positive or bounded parameters).
    checkpoint_interval : int, optional
        Frequency (in epochs) between checkpoints (default 10).
    max_num_epochs : int, optional
        Maximum number of SVI or MAP epochs (default 100000); a cap only.
    forward_batch_size : int, optional
        When getting NUTS posteriors, calculate forward predictions in batches
        of this size (default 512).
    pre_map_num_epoch : int, optional
        Maximum number of epochs of the MAP warm-up run before SVI (default
        10000; 0 skips it).  The warm-up stops earlier when it converges.
        Only used if analysis_method is 'svi'.
    init_param_jitter : float, optional
        Multiplicative jitter on the component guide's starting parameters,
        to break symmetry (default 0.1).  Not used by autoguides or MAP.
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

    # Validate guide options before any (slow) fitting starts.
    guide_type = resolve_guide_type(guide_type)
    guide_kwargs = {}
    if guide_rank is not None:
        guide_kwargs["rank"] = guide_rank
    if guide_init_scale is not None and guide_type != "component":
        guide_kwargs["init_scale"] = guide_init_scale
    if analysis_method != "svi" and (guide_type != "component" or guide_kwargs
                                     or guide_init_scale is not None):
        raise ValueError(
            "guide_type, guide_rank and guide_init_scale apply only to "
            "analysis_method='svi' ('map' always uses 'delta'; 'nuts' uses no "
            "guide)."
        )
    check_guide_kwargs(guide_type, guide_kwargs)
    if guide_init_scale is not None and not guide_init_scale > 0:
        raise ValueError(
            f"guide_init_scale must be positive (got {guide_init_scale})."
        )
    if analysis_method == "svi" and checkpoint_file is not None:
        _check_checkpoint_guide(checkpoint_file, guide_type)

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

    orchestrator, guesses = read_configuration(config_file)

    # For posterior mode the seed is optional: the checkpoint restores the PRNG
    # key for SVI checkpoints, and any valid key works for MAP/Laplace sampling.
    effective_seed = seed if seed is not None else 0

    # Run SVI / MAP
    ri = RunInference(orchestrator, effective_seed)

    optimizer_kwargs = dict(adam_step_size=adam_step_size,
                            adam_clip_norm=adam_clip_norm,
                            elbo_num_particles=elbo_num_particles)
    convergence_kwargs = dict(convergence_window_steps=convergence_window_steps,
                              patience=patience,
                              convergence_z=convergence_z,
                              loss_rtol=loss_rtol,
                              param_tolerance=param_tolerance,
                              adam_final_step_size=adam_final_step_size,
                              adam_step_size_cut=adam_step_size_cut)

    if analysis_method == "svi":

        # Resuming: the checkpoint overwrites the starting point.
        if checkpoint_file is not None:
            init_params = guesses
            init_values = None

        else:
            # Configured guesses, as site values (they are keyed by site or by
            # component-guide parameter name).
            start_values = ri.site_values(guesses)

            if pre_map_num_epoch > 0:
                _, map_params, _ = _run_map(
                    ri,
                    init_values=start_values,
                    out_prefix=f"{out_prefix}_premap",
                    label="Pre-MAP",
                    **optimizer_kwargs,
                    **_optimization_kwargs(
                        **convergence_kwargs,
                        checkpoint_interval=pre_map_num_epoch,
                        max_num_epochs=pre_map_num_epoch,
                        epoch_checkpoint_interval=None))
                # The MAP point replaces the guesses where it has a value.
                start_values = {**start_values, **ri.site_values(map_params)}

            if guide_type == "component":
                init_scale = (DEFAULT_COMPONENT_INIT_SCALE
                              if guide_init_scale is None
                              else guide_init_scale)
                init_params = ri.component_guide_start(start_values,
                                                       guesses=guesses,
                                                       init_scale=init_scale)
                init_values = None
            else:
                init_params = None
                init_values = start_values

        return _run_svi(ri,
                        init_params=init_params,
                        checkpoint_file=checkpoint_file,
                        guide_type=guide_type,
                        guide_kwargs=guide_kwargs,
                        init_values=init_values,
                        out_prefix=out_prefix,
                        **optimizer_kwargs,
                        **_optimization_kwargs(
                            **convergence_kwargs,
                            checkpoint_interval=checkpoint_interval,
                            max_num_epochs=max_num_epochs,
                            init_param_jitter=init_param_jitter,
                            epoch_checkpoint_interval=epoch_checkpoint_interval))

    elif analysis_method == "map":
        init_values = None
        if checkpoint_file is None:
            init_values = ri.site_values(guesses)
        return _run_map(ri,
                        init_values=init_values,
                        checkpoint_file=checkpoint_file,
                        out_prefix=out_prefix,
                        **optimizer_kwargs,
                        **_optimization_kwargs(
                            **convergence_kwargs,
                            checkpoint_interval=checkpoint_interval,
                            max_num_epochs=max_num_epochs,
                            epoch_checkpoint_interval=epoch_checkpoint_interval))

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
                                              "adam_clip_norm":float,
                                              "seed":int,
                                              "checkpoint_file":str,
                                              "guide_rank":int,
                                              "guide_init_scale":float,
                                              "pre_map_num_epoch":int,
                                              "init_param_jitter":float,
                                              "nuts_num_warmup":int,
                                              "nuts_num_samples":int,
                                              "nuts_num_chains":int,
                                              "nuts_target_accept_prob":float,
                                              "epoch_checkpoint_interval":int})

if __name__ == "__main__":
    main()
