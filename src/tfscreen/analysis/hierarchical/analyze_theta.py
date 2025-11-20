from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.util.generalized_main import generalized_main

import os

def _run_map(ri,
             init_params,
             checkpoint_file=None,
             out_root="tfs",
             adam_step_size=1e-6,
             adam_clip_norm=1.0,
             map_elbo_num_particles=10,
             map_convergence_tolerance=1e-5,
             convergence_window=1000,
             checkpoint_interval=1000,
             map_num_steps=100000,
             batch_size=1024):
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
    out_root : str, optional
        Output file root for checkpoints and results (default "tfs").
    adam_step_size : float, optional
        Step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    map_elbo_num_particles : int, optional
        Number of particles for ELBO estimation during MAP (default 10).
    map_convergence_tolerance : float, optional
        Relative change in loss to declare MAP convergence (default 1e-5).
    convergence_window : int, optional
        Number of steps to average for convergence check (default 1000).
    checkpoint_interval : int, optional
        Steps between checkpoints and convergence checks (default 1000).
    map_num_steps : int, optional
        Number of MAP optimization steps (default 100000).
    batch_size : int, optional
        Mini-batch size for optimization (default 1024).

    Returns
    -------
    svi_state : Any
        Final optimizer state object from MAP.
    params : dict
        Final optimized parameters from MAP.
    converged : bool
        True if MAP converged according to the specified tolerance.
    """

     # Create a maximum a posteriori svi object
    map_obj = ri.setup_map(adam_step_size=adam_step_size,
                           adam_clip_norm=adam_clip_norm,
                           elbo_num_particles=map_elbo_num_particles)
    
    if os.path.isfile(f"{out_root}_losses.csv"):
        os.remove(f"{out_root}_losses.csv")

    # Run MAP
    svi_state, params, converged = ri.run_optimization(
        map_obj,
        init_params=init_params, 
        out_root=out_root,
        svi_state=checkpoint_file,
        convergence_tolerance=map_convergence_tolerance,
        convergence_window=convergence_window,
        checkpoint_interval=checkpoint_interval,
        num_steps=map_num_steps,
        batch_size=batch_size
    )

    # Write the current parameter values
    ri.write_params(params,out_root=out_root)

    # Write convergence information to stdout
    if converged:
        print("MAP run converged.",flush=True)
    else:
        print("MAP run has not yet converged.",flush=True)

    return svi_state, params, converged

def _run_svi(ri,
             init_params,
             checkpoint_file=None,
             out_root="tfs",
             adam_step_size=1e-6,
             adam_clip_norm=1.0,
             guide_rank=10,
             elbo_num_particles=10,
             convergence_tolerance=1e-5,
             convergence_window=1000,
             checkpoint_interval=1000,
             num_steps=100000,
             batch_size=1024,
             num_posterior_samples=10000,
             posterior_batch_size=500,
             always_get_posterior=False):
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
    out_root : str, optional
        Output file root for checkpoints and results (default "tfs").
    adam_step_size : float, optional
        Step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    guide_rank : int, optional
        Rank for the variational guide (default 10).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during SVI (default 10).
    convergence_tolerance : float, optional
        Relative change in loss to declare SVI convergence (default 1e-5).
    convergence_window : int, optional
        Number of steps to average for convergence check (default 1000).
    checkpoint_interval : int, optional
        Steps between checkpoints and convergence checks (default 1000).
    num_steps : int, optional
        Number of SVI optimization steps (default 100000).
    batch_size : int, optional
        Mini-batch size for optimization (default 1024).
    num_posterior_samples : int, optional
        Number of posterior samples to draw after convergence (default 10000).
    posterior_batch_size : int, optional
        Sample parameter posteriors in batches of this size (default 500).
    always_get_posterior : bool, optional
        If True, always sample posteriors even if not converged (default False).

    Returns
    -------
    svi_state : Any
        Final optimizer state object from SVI.
    svi_params : dict
        Final optimized parameters from SVI.
    converged : bool
        True if SVI converged according to the specified tolerance.
    """
    
    # Create an svi object
    svi_obj = ri.setup_svi(adam_step_size=adam_step_size,
                           adam_clip_norm=adam_clip_norm,
                           guide_rank=guide_rank,
                           elbo_num_particles=elbo_num_particles,
                           init_params=init_params)
    
    # Run svi
    svi_state, svi_params, converged = ri.run_optimization(
        svi_obj,
        init_params=None, # init_params captured during svi_obj initialization
        out_root=f"{out_root}",
        svi_state=checkpoint_file,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        checkpoint_interval=checkpoint_interval,
        num_steps=num_steps,
        batch_size=batch_size
    )

    if converged or always_get_posterior:
        ri.get_posteriors(svi=svi_obj,
                          svi_state=svi_state,
                          out_root=out_root,
                          num_posterior_samples=num_posterior_samples,
                          batch_size=posterior_batch_size)
        
    # Write convergence information to stdout
    if converged:
        print("SVI run converged.",flush=True)
    else:
        print("SVI run has not yet converged.",flush=True)

    
def analyze_theta(growth_df,
                  binding_df,
                  seed,
                  condition_growth_model="hierarchical",
                  ln_cfu0_model="hierarchical",
                  dk_geno_model="hierarchical",
                  activity_model="fixed",
                  theta_model="hill",
                  theta_growth_noise_model="beta",
                  theta_binding_noise_model="beta",
                  checkpoint_file=None,
                  analysis_method="svi",
                  out_root="tfs",
                  adam_step_size=1e-6,
                  adam_clip_norm=1.0,
                  guide_rank=10,
                  elbo_num_particles=10,
                  convergence_tolerance=1e-5,
                  convergence_window=1000,
                  checkpoint_interval=1000,
                  num_steps=100000,
                  batch_size=1024,
                  num_posterior_samples=10000,
                  posterior_batch_size=500,
                  always_get_posterior=False):
    """
    Run the joint hierarchical growth model to extract estimates of
    transcription factor fractional occupancy (theta) and other latent
    parameters using Stochastic Variational Inference (SVI) or maximum a
    posteriori (MAP) approaches. 

    The default runs MAP for the specified number of iterations to get
    reasonable starting guesses, then SVI until convergence. This writes
    checkpoints, checks for convergence, and writes posterior samples, and
    finally returns the last SVI state, parameters, and convergence status. It
    can also restart from a checkpoint file. 

    Parameters
    ----------
    growth_df : pandas.DataFrame
        Input data for the growth model (e.g., genotype/cfu measurements).
    binding_df : pandas.DataFrame
        Input data for the binding model (e.g., theta vs. titrant measurements)
    seed : int
        Random seed for reproducibility.
    condition_growth_model: str, optional
        model to use to describe growth under different conditions (e.g., 
        pheS+4CP). Allowed values are 'hierarchical' (default) or 'independent'.
    ln_cfu0_model : str, optional
        model to use to describe ln_cfu0, the initial populations of genotypes
        in each replicate. Only 'hierarchical' is allowed at this point. 
    dk_geno_model : str, optional
        model to use to describe dk_geno, the pleiotropic effect of a genotype
        on growth, independent of occupancy. Allowed values are 'hierarchical' 
        (default) or 'fixed'.
    activity_model : str, optional
        model to use to describe activity, a scalar multiplied against 
        occupancy that defines how strongly a genotype alters transcription 
        given its occupancy. Allowed values are 'fixed' (default), 'hierarchical',
        and 'horseshoe'.
    theta_model : str, optional
        model to use to describe theta, the fractional occupancy of a genotype
        on the transcription factor binding site. Allowed values are 'hill' 
        (default) or 'categorical'. 
    theta_growth_noise_model : str, optional
        model to use for stochastic experimental noise in theta measured by 
        bacterial growth. Allowed values are 'beta' (default) or 'none' (written
        as a string)
    theta_binding_noise_model : str, optional
        model to use for stochastic experimental noise in theta measured by 
        binding. Allowed values are 'beta' (default) or 'none' (written as a
        string)
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume SVI from, or None to start fresh. If
        `map_only` is False, the checkpoint_file is assumed to be from the SVI
        run started after a MAP run. 
    out_root : str, optional
        Output file root for checkpoints and results (default 'tfs').
    adam_step_size : float, optional
        Step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    guide_rank : int, optional
        Rank for the variational guide (default 10).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation (default 10).
    convergence_tolerance : float, optional
        Relative change in loss to declare SVI convergence (default 1e-5).
    convergence_window : int, optional
        Number of steps to average for convergence check (default 1000).
    checkpoint_interval : int, optional
        Steps between checkpoints and convergence checks (default 1000).
    num_steps : int, optional
        Number of SVI optimization steps (default 100000).
    batch_size : int, optional
        Mini-batch size for optimization (default 1024).
    num_posterior_samples : int, optional
        Number of posterior samples to draw after convergence (default 10000).
    posterior_batch_size : int, optional
        Sample parameter posteriors in batches of this size (default 500)
    map_elbo_num_particles : int, optional
        Number of particles for MAP ELBO estimation (default 1).
    map_convergence_tolerance : float or None, optional
        Convergence tolerance for MAP optimization (default None).
    map_num_steps : int, optional
        Number of MAP optimization steps (default 100000).
    always_get_posterior : bool, optional
        If True, always sample posteriors even if not converged (default False).

    Returns
    -------
    svi_state : Any
        Final SVI state object from the optimizer.
    svi_params : dict
        Final optimized parameters from SVI.
    converged : bool
        True if SVI converged according to the specified tolerance.

    Notes
    -----
    This function writes checkpoints and posterior samples to disk using the provided
    output root. 
    """

    # Construct growth model, which defines the jax model and all necessary 
    # parameters to describe the experiments
    gm = GrowthModel(growth_df,
                     binding_df,
                     condition_growth=condition_growth_model,
                     ln_cfu0=ln_cfu0_model,
                     dk_geno=dk_geno_model,
                     activity=activity_model,
                     theta=theta_model,
                     theta_growth_noise=theta_growth_noise_model,
                     theta_binding_noise=theta_binding_noise_model)
    
    # Create a run inference object, which manages things like checking for 
    # ELBO convergence.
    ri = RunInference(gm,seed)

    # Run SVI
    if analysis_method == "svi":

        return _run_svi(ri,
                        init_params=None,
                        checkpoint_file=checkpoint_file,
                        out_root=out_root,
                        adam_step_size=adam_step_size,
                        adam_clip_norm=adam_clip_norm,
                        guide_rank=guide_rank,
                        elbo_num_particles=elbo_num_particles,
                        convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        checkpoint_interval=checkpoint_interval,
                        num_steps=num_steps,
                        batch_size=batch_size,
                        num_posterior_samples=num_posterior_samples,
                        posterior_batch_size=posterior_batch_size,
                        always_get_posterior=always_get_posterior)
    
    # Run MAP
    elif analysis_method == "map":

        return _run_map(ri,
                        gm.init_params,
                        checkpoint_file=checkpoint_file,
                        out_root=out_root,
                        adam_step_size=adam_step_size,
                        adam_clip_norm=adam_clip_norm,
                        map_elbo_num_particles=elbo_num_particles,
                        map_convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        checkpoint_interval=checkpoint_interval,
                        map_num_steps=num_steps,
                        batch_size=batch_size)
        
    # Not recognized
    else:
        raise ValueError(
            f"analysis method '{analysis_method}' not recognized. This should "
            "be 'SVI' or 'MAP'."
        )


def main():

    return generalized_main(analyze_theta,
                            manual_arg_types={"seed":int,
                                              "checkpoint_file":str,
                                              "map_convergence_tolerance":float})
