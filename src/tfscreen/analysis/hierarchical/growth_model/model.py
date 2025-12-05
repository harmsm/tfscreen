

from .batch import generate_batch

# Import for typing
from .data_class import (
    DataClass,
    PriorsClass,
)

import jax.numpy as jnp
import numpyro as pyro
from typing import Dict, Any

def jax_model(data: DataClass,
              priors: PriorsClass,
              **control):
    """
    Defines the joint hierarchical model for bacterial growth and binding.

    This model can be called 

    Parameters
    ----------
    data : DataClass
        A DataClass pytree containing all observed data and experimental
        metadata for both growth and binding assays.
    priors : PriorsClass
        A PriorsClass pytree containing the prior distributions
        (as Numpyro distribution objects) for all latent parameters.
    control : dict
        dictionary of keyword arguments necessary to specify the model. Expects:
        - theta
        - condition_growth
        - ln_cfu0
        - activity
        - dk_geno
        - theta_binding_noise
        - theta_growth_noise
        - binding_observer
        - growth_observer
        - is_guide
        The dictionary should also optionally have `batch_idx` (which overrides
        whatever is in `batch_size`) or `batch_size`. 
    """
    
    # -------------------------------------------------------------------------
    # Parse control inputs 

    theta_model, calc_theta = control["theta"]    
    
    condition_growth_model = control["condition_growth"]
    ln_cfu0_model = control["ln_cfu0"]
    activity_model = control["activity"]
    dk_geno_model = control["dk_geno"]
    
    theta_binding_noise_model = control["theta_binding_noise"]
    theta_growth_noise_model = control["theta_growth_noise"]

    binding_observer = control["observe_binding"]
    growth_observer = control["observe_growth"]

    is_guide = control["is_guide"]

    # -------------------------------------------------------------------------
    # Deal with batching/indexing

    # `shared_genotype_plate` is used throughout the model.

    # Forward sampling -- explicit batch_idx passed to the model when it was 
    # initialized. 
    if "batch_idx" in control:

        # Create a plate that is the right size but is not sampled
        idx = control["batch_idx"]
        with pyro.plate("shared_genotype_plate",size=len(idx),dim=-1):
             batched_data = generate_batch(data,idx)

    # Training -- generate a randomly sampled batch
    else:

        # Create a plate that is sub-sampled
        batch_size = control["batch_size"]
        with pyro.plate("shared_genotype_plate", 
                        size=data.num_genotype, 
                        subsample_size=batch_size, 
                        dim=-1) as idx:
            
            batched_data = generate_batch(data,idx)


    # -------------------------------------------------------------------------
    # Calculate theta

    # Calculate shared theta
    theta = theta_model("theta",
                        batched_data.growth,
                        priors.theta)
    
    # -------------------------------------------------------------------------
    # Make prediction for the binding experiment

    theta_binding = calc_theta(theta,batched_data.binding)
    pyro.deterministic(f"theta_binding_pred",theta_binding)
    binding_pred = theta_binding_noise_model("theta_binding_noise",
                                             theta_binding,
                                             priors.binding.theta_binding_noise)
    
    # -------------------------------------------------------------------------
    # Make prediction for the growth experiment

    # theta
    theta_growth = calc_theta(theta,batched_data.growth)
    pyro.deterministic(f"theta_growth_pred",theta_growth)
    noisy_theta_growth = theta_growth_noise_model("theta_growth_noise",
                                                  theta_growth,
                                                  priors.growth.theta_growth_noise)
    
    # Get growth parameters
    k_pre, m_pre, k_sel, m_sel = condition_growth_model("condition_growth",
                                                        batched_data.growth,
                                                        priors.growth.condition_growth)

    # initial populations
    ln_cfu0 = ln_cfu0_model("ln_cfu0",
                            batched_data.growth,
                            priors.growth.ln_cfu0)
    
    # pleiotropic effect of mutation
    dk_geno = dk_geno_model("dk_geno",
                            batched_data.growth,
                            priors.growth.dk_geno)   

    # activity
    activity = activity_model("activity",
                              batched_data.growth,
                              priors.growth.activity)

    # -------------------------------------------------------------------------
    # finalize

    # If this is a guide, just make the final observations but do not calculate
    # final tensors
    if is_guide:

        growth_observer("final_binding_obs",batched_data.growth,None)
        binding_observer("final_growth_obs",batched_data.binding,None)

    # real calculation
    else:

        # calculate observable (all tensors have correct dimensions)
        g_pre = k_pre + dk_geno + activity*m_pre*noisy_theta_growth
        g_sel = k_sel + dk_geno + activity*m_sel*noisy_theta_growth
        ln_cfu_pred = ln_cfu0 + g_pre*data.growth.t_pre + g_sel*data.growth.t_sel

        # Register results
        pyro.deterministic(f"binding_pred",binding_pred)
        pyro.deterministic(f"growth_pred",ln_cfu_pred)
    
        # Calculate likelihood
        growth_observer("final_binding_obs",batched_data.growth,ln_cfu_pred)
        binding_observer("final_growth_obs",batched_data.binding,binding_pred)


