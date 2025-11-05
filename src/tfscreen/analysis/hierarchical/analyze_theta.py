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
    """
    
    gm = GrowthModel(df,measured_hill)
    ri = RunInference(gm,seed)

    # Run from a checkpoint file
    if checkpoint_file is not None:
        svi_obj, svi_state, params, converged = ri.run_from_checkpoint(
            checkpoint_file=checkpoint_file,
            adam_step_size=adam_step_size,
            adam_clip_norm=adam_clip_norm,
            guide_rank=guide_rank,
            elbo_num_particles=elbo_num_particles,
            convergence_tolerance=convergence_tolerance,
            convergence_window=convergence_window,
            checkpoint_interval=checkpoint_interval,
            num_steps=num_steps,
            batch_size=batch_size,
            num_posterior_samples=num_posterior_samples
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
    return generalized_main(run_svi_analysis)