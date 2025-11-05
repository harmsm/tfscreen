from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.util.generalized_main import generalized_main

import os

def run_svi_analysis(df,
                     measured_hill,
                     seed,
                     checkpoint_file=None,
                     out_root="tfs",
                     adam_step_size=1e-6,
                     adam_clip_norm=1.0,
                     guide_rank=10,
                     elbo_num_particles=10,
                     init_params=None,
                     convergence_tolerance=1e-5,
                     convergence_window=1000,
                     checkpoint_interval=1000,
                     num_steps=100000,
                     batch_size=1024,
                     num_posterior_samples=10000,
                     map_elbo_num_particles=1,
                     map_convergence_tolerance=None,
                     map_num_steps=100000,
                     always_get_posterior=False):
    """
    Run stochastic variational inference (SVI) analysis for a hierarchical
    growth model.

    Set up and run MAP (maximum a posteriori) and/or SVI optimization for the
    provided data and model configuration, optionally resuming from a checkpoint.
    It writes checkpoints and posterior samples to disk, and returns the final
    SVI state, parameters, and convergence status.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data for the growth model (e.g., genotype/phenotype measurements).
    measured_hill : float or array-like
        Hill coefficient(s) or related measurement(s) for the model.
    seed : int
        Random seed for reproducibility.
    checkpoint_file : str or None, optional
        Path to a checkpoint file to resume SVI from, or None to start fresh.
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
    init_params : dict or None, optional
        Initial parameters for MAP optimization. If None, uses model defaults.
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
    output root. It can resume from a checkpoint or start from MAP optimization.
    """
    
    gm = GrowthModel(df,measured_hill)
    ri = RunInference(gm,seed)

    # Run from a checkpoint file
    if checkpoint_file is not None:

        # Create an svi object
        svi_obj = ri.setup_svi(adam_step_size=adam_step_size,
                               adam_clip_norm=adam_clip_norm,
                               guide_rank=guide_rank,
                               elbo_num_particles=elbo_num_particles,
                               init_params=None)
        
        # Run svi
        svi_state, svi_params, converged = ri.run_optimization(
            svi_obj,
            init_params=gm.init_params,
            svi_state=checkpoint_file,
            out_root=f"{out_root}",
            convergence_tolerance=convergence_tolerance,
            convergence_window=convergence_window,
            checkpoint_interval=checkpoint_interval,
            num_steps=num_steps,
            batch_size=batch_size
        )


    else:

        # Create a maximum a posteriori svi object
        map_obj = ri.setup_map(adam_step_size=adam_step_size,
                               adam_clip_norm=adam_clip_norm,
                               elbo_num_particles=map_elbo_num_particles)
        
        if os.path.isfile(f"{out_root}-map_losses.csv"):
            os.remove(f"{out_root}-map_losses.csv")
    
        # Grab model initial params if none specified
        if init_params is None:
            init_params = gm.init_params

        # Run MAP
        _, map_params, _ = ri.run_optimization(
            map_obj,
            init_params=init_params,
            out_root=f"{out_root}-map",
            convergence_tolerance=map_convergence_tolerance,
            convergence_window=convergence_window,
            checkpoint_interval=checkpoint_interval,
            num_steps=map_num_steps,
            batch_size=batch_size
        )
                                
        # Create an svi object
        svi_obj = ri.setup_svi(adam_step_size=adam_step_size,
                               adam_clip_norm=adam_clip_norm,
                               guide_rank=guide_rank,
                               elbo_num_particles=elbo_num_particles,
                               init_params=map_params)
        
        # Run svi
        svi_state, svi_params, converged = ri.run_optimization(
            svi_obj,
            init_params=map_params,
            out_root=f"{out_root}",
            convergence_tolerance=convergence_tolerance,
            convergence_window=convergence_window,
            checkpoint_interval=checkpoint_interval,
            num_steps=num_steps,
            batch_size=batch_size
        )

    if converged or always_get_posterior:
        ri.get_posteriors(guide=svi_obj.guide,
                            params=svi_params,
                            out_root=f"{out_root}-posterior",
                            num_posterior_samples=num_posterior_samples,
                            batch_size=batch_size)
        
    return svi_state, svi_params, converged

def main():
    return generalized_main(run_svi_analysis,
                            manual_arg_types={"seed":int,
                                              "checkpoint_file":str})