from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.analysis.hierarchical.summarize_posteriors import summarize_posteriors
from tfscreen.util.cli.generalized_main import generalized_main

import os
import optax

def _run_map(ri,
             init_params,
             checkpoint_file=None,
             out_root="tfs",
             adam_step_size=1e-3,
             adam_final_step_size=1e-6,
             adam_clip_norm=1.0,
             elbo_num_particles=2,
             convergence_tolerance=0.01,
             convergence_window=10,
             patience=10,
             convergence_check_interval=2,
             checkpoint_interval=10,
             map_num_steps=100000,
             num_posterior_samples=10000,
             sampling_batch_size=100,
             forward_batch_size=512,
             always_get_posterior=False):
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
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Final step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during MAP (default 2).
    map_num_steps : int, optional
        Number of MAP optimization steps (default 100000).
    convergence_tolerance : float, optional
        Relative change in smoothed loss to declare MAP convergence (default 0.01).
    convergence_window : int, optional
        Number of epochs to average for convergence check (default 10).
    patience : int, optional
        Number of consecutive checks meeting tolerance to declare convergence
        (default 10).
    convergence_check_interval : int, optional
        Frequency (in epochs) to check for convergence (default 2).
    checkpoint_interval : int, optional
        Steps between checkpoints and convergence checks (default 10).
    map_guide_type : str, optional
        Type of guide to use for MAP. Allowed values are 'laplace',
        'diagonal_laplace' (default), 'normal', or 'delta'.

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
        transition_steps=int(map_num_steps * 0.25),
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
        out_root=out_root,
        svi_state=checkpoint_file,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        patience=patience,
        convergence_check_interval=convergence_check_interval,
        checkpoint_interval=checkpoint_interval,
        num_steps=map_num_steps,
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
             adam_step_size=1e-3,
             adam_final_step_size=1e-6,
             adam_clip_norm=1.0,
             elbo_num_particles=2,
             convergence_tolerance=0.01,
             convergence_window=10,
             patience=10,
             convergence_check_interval=2,
             checkpoint_interval=10,
             num_steps=100000,
             num_posterior_samples=10000,
             sampling_batch_size=100,
             forward_batch_size=512,
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
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Final step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation during SVI (default 2).
    convergence_tolerance : float, optional
        Relative change in loss to declare SVI convergence (default 0.01).
    convergence_window : int, optional
        Number of epochs to average for convergence check (default 10).
    patience : int, optional
        Number of consecutive checks meeting tolerance to declare convergence
        (default 10).
    convergence_check_interval : int, optional
        Frequency (in epochs) to check for convergence (default 2).
    checkpoint_interval : int, optional
        Frequency (in epochs) between checkpoints (default 10).
    num_steps : int, optional
        Number of SVI optimization steps (default 100000).
    num_posterior_samples : int, optional
        Number of posterior samples to draw after convergence (default 10000).
    sampling_batch_size : int, optional
        When getting posteriors, sample parameter posteriors in batches of this
        size (default 100).
    forward_batch_size : int, optional
        when getting posteriors, calculate forward predictions in batches of
        this size (default 512)
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
    
    # Create a learning rate schedule
    schedule = optax.exponential_decay(
        init_value=adam_step_size,
        transition_steps=int(num_steps * 0.25),
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
        out_root=f"{out_root}",
        svi_state=checkpoint_file,
        convergence_tolerance=convergence_tolerance,
        convergence_window=convergence_window,
        patience=patience,
        convergence_check_interval=convergence_check_interval,
        checkpoint_interval=checkpoint_interval,
        num_steps=num_steps,
    )

    if converged or always_get_posterior:

        ri.get_posteriors(svi=svi_obj,
                          svi_state=svi_state,
                          out_root=out_root,
                          num_posterior_samples=num_posterior_samples,
                          sampling_batch_size=sampling_batch_size,
                          forward_batch_size=forward_batch_size)
        
        # Write summary files
        summarize_posteriors(posterior_file=f"{out_root}_posterior.npz",
                             config_file=f"{out_root}_config.yaml",
                             out_root=out_root)
        
    # Write convergence information to stdout
    if converged:
        print("SVI run converged.",flush=True)
    else:
        print("SVI run has not yet converged.",flush=True)
    
    return svi_state, params, converged

    
def analyze_theta(growth_df=None,
                  binding_df=None,
                  seed=None,
                  config_file=None,
                  condition_growth_model="hierarchical",
                  ln_cfu0_model="hierarchical",
                  dk_geno_model="hierarchical",
                  activity_model="horseshoe",
                  theta_model="hill",
                  transformation_model="congression",
                  theta_growth_noise_model="none",
                  theta_binding_noise_model="none",
                  checkpoint_file=None,
                  analysis_method="svi",
                  out_root="tfs",
                  adam_step_size=1e-3,
                  adam_final_step_size=1e-6,
                  adam_clip_norm=1.0,
                  elbo_num_particles=2,
                  convergence_tolerance=0.01,
                  convergence_window=10,
                  patience=10,
                  convergence_check_interval=2,
                  checkpoint_interval=10,
                  num_steps=100000,
                  batch_size=1024,
                  num_posterior_samples=10000,
                  sampling_batch_size=100,
                  forward_batch_size=512,
                  always_get_posterior=False,
                  spiked=None):
    """
    Run the joint hierarchical growth model to extract estimates of
    transcription factor fractional occupancy (theta) and other latent
    parameters using Stochastic Variational Inference (SVI) or maximum a
    posteriori (MAP) approaches. 

    The default is SVI. This writes checkpoints, checks for convergence, and
    writes posterior samples, and finally returns the last SVI state,
    parameters, and convergence status. The function can also restart from a
    checkpoint file. 

    Parameters
    ----------
    growth_df : pandas.DataFrame or str, optional
        Input data for the growth model (e.g., genotype/cfu measurements).
        If config_file is provided, this overrides the config path.
    binding_df : pandas.DataFrame or str, optional
        Input data for the binding model (e.g., theta vs. titrant measurements)
        If config_file is provided, this overrides the config path.
    seed : int, optional
        Random seed for reproducibility. Must be provided if not loading from SVI.
    config_file : str, optional
        Path to a YAML configuration file to load settings from.
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
    transformation_model : str, optional
        model for transformation correction. Allowed values are 'congression'
        (default) or 'single'.
    theta_growth_noise_model : str, optional
        model to use for stochastic experimental noise in theta measured by 
        bacterial growth. Allowed values are 'beta' (default) or 'none' (written
        as a string)
    theta_binding_noise_model : str, optional
        model to use for stochastic experimental noise in theta measured by 
        binding. Allowed values are 'beta' (default) or 'none' (written as a
        string)
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume SVI from, or None to start fresh.
    analysis_method : str, optional
        Method for inference. Allowed values are 'svi' (default), 'map', or 
        'posterior'.
    out_root : str, optional
        Output file root for checkpoints and results (default 'tfs').
    adam_step_size : float, optional
        Starting step size for the Adam optimizer (default 1e-3).
    adam_final_step_size : float, optional
        Final step size for the Adam optimizer (default 1e-6).
    adam_clip_norm : float, optional
        Gradient clipping norm for Adam optimizer (default 1.0).
    elbo_num_particles : int, optional
        Number of particles for ELBO estimation (default 2).
    convergence_tolerance : float, optional
        Relative change in loss to declare SVI convergence (default 0.01).
    convergence_window : int, optional
        Number of epochs to average for convergence check (default 10).
    patience : int, optional
        Number of consecutive checks meeting tolerance to declare convergence
        (default 10).
    convergence_check_interval : int, optional
        Frequency (in epochs) to check for convergence (default 2).
    checkpoint_interval : int, optional
        Frequency (in epochs) between checkpoints (default 10).
    num_steps : int, optional
        Number of SVI optimization steps (default 100000).
    batch_size : int, optional
        Mini-batch size for optimization (default 1024).
    num_posterior_samples : int, optional
        Number of posterior samples to draw after convergence (default 10000).
    sampling_batch_size : int, optional
        When getting posteriors, sample parameter posteriors in batches of this
        size (default 100).
    forward_batch_size : int, optional
        when getting posteriors, calculate forward predictions in batches of
        this size (default 512)
    always_get_posterior : bool, optional
        If True, always sample posteriors even if not converged (default False).
    spiked : list, optional
        List of genotypes to mask from theta correction (e.g. spiked-in variants).

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

    # If config_file is provided, load settings from it. Overwrite any
    # settings provided as arguments.

    if config_file is not None:

        c_growth_df, c_binding_df, c_settings = GrowthModel.load_config(config_file)

        # Overwrite all other settings from the config
        growth_df = c_growth_df
        binding_df = c_binding_df
        condition_growth_model = c_settings["condition_growth"]
        ln_cfu0_model = c_settings["ln_cfu0"]
        dk_geno_model = c_settings["dk_geno"]
        activity_model = c_settings["activity"]
        theta_model = c_settings["theta"]
        transformation_model = c_settings["transformation"]
        theta_growth_noise_model = c_settings["theta_growth_noise"]
        theta_binding_noise_model = c_settings["theta_binding_noise"]
        spiked = c_settings["spiked_genotypes"]

    # validation
    if growth_df is None or binding_df is None:
        raise ValueError("growth_df and binding_df must be provided if config_file is not provided.")

    if seed is None and checkpoint_file is None:
        raise ValueError("seed must be provided unless loading from a checkpoint.")

    # Kind of a hack, but this forces no batching for the posterior calc
    if analysis_method == "posterior":
        batch_size = None

    # Construct growth model, which defines the jax model and all necessary 
    # parameters to describe the experiments
    gm = GrowthModel(growth_df,
                     binding_df,
                     batch_size=batch_size,
                     condition_growth=condition_growth_model,
                     ln_cfu0=ln_cfu0_model,
                     dk_geno=dk_geno_model,
                     activity=activity_model,
                     theta=theta_model,
                     transformation=transformation_model,
                     theta_growth_noise=theta_growth_noise_model,
                     theta_binding_noise=theta_binding_noise_model,
                     spiked_genotypes=spiked)
    
    # Save the model configuration
    gm.write_config(growth_df, binding_df, out_root)
    
    # Create a run inference object, which manages things like checking for 
    # ELBO convergence.
    ri = RunInference(gm,seed)

    # Run SVI
    if analysis_method == "svi":

        return _run_svi(ri,
                        init_params=gm.init_params,
                        checkpoint_file=checkpoint_file,
                        out_root=out_root,
                        adam_step_size=adam_step_size,
                        adam_final_step_size=adam_final_step_size,
                        adam_clip_norm=adam_clip_norm,
                        elbo_num_particles=elbo_num_particles,
                        convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        patience=patience,
                        convergence_check_interval=convergence_check_interval,
                        checkpoint_interval=checkpoint_interval,
                        num_steps=num_steps,
                        num_posterior_samples=num_posterior_samples,
                        sampling_batch_size=sampling_batch_size,
                        forward_batch_size=forward_batch_size,
                        always_get_posterior=always_get_posterior)
    
    # Run MAP
    elif analysis_method == "map":

        return _run_map(ri,
                        gm.init_params,
                        checkpoint_file=checkpoint_file,
                        out_root=out_root,
                        adam_step_size=adam_step_size,
                        adam_final_step_size=adam_final_step_size,
                        adam_clip_norm=adam_clip_norm,
                        elbo_num_particles=elbo_num_particles,
                        convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        patience=patience,
                        convergence_check_interval=convergence_check_interval,
                        checkpoint_interval=checkpoint_interval,
                        map_num_steps=num_steps,
                        num_posterior_samples=num_posterior_samples,
                        sampling_batch_size=sampling_batch_size,
                        forward_batch_size=forward_batch_size,
                        always_get_posterior=always_get_posterior)
                          
    elif analysis_method == "posterior":

        return _run_svi(ri,
                        init_params=None,
                        checkpoint_file=checkpoint_file,
                        out_root=out_root,
                        adam_step_size=adam_step_size,
                        adam_final_step_size=adam_final_step_size,
                        adam_clip_norm=adam_clip_norm,
                        elbo_num_particles=elbo_num_particles,
                        convergence_tolerance=convergence_tolerance,
                        convergence_window=convergence_window,
                        patience=patience,
                        convergence_check_interval=convergence_check_interval,
                        checkpoint_interval=checkpoint_interval,
                        num_steps=0, # Don't do any optimization
                        num_posterior_samples=num_posterior_samples,
                        sampling_batch_size=sampling_batch_size,
                        forward_batch_size=forward_batch_size,
                        always_get_posterior=True) # Force grabbing the posterior

    # Not recognized
    else:
        raise ValueError(
            f"analysis method '{analysis_method}' not recognized. This should "
            "be 'svi', 'map', or 'posterior'"
        )


def main():
    """
    CLI entry point for running the hierarchical analysis.

    This function wraps `analyze_theta` using `generalized_main`, allowing
    execution from the command line with argument parsing.
    """

    return generalized_main(analyze_theta,
                            manual_arg_types={"growth_df":str,
                                              "binding_df":str,
                                              "seed":int,
                                              "checkpoint_file":str,
                                              "config_file":str,
                                              "spiked":list},
                            manual_arg_nargs={"spiked":"+"})

if __name__ == "__main__":
    main()
