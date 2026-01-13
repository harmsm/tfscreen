import jax.numpy as jnp
import jax
import numpyro as pyro
import numpyro.distributions as dist
from typing import Callable, Any, Optional

def sample_genotype_parameter(name: str, 
                              data: Any, 
                              sample_fn: Callable[[str, int], jnp.ndarray],
                              is_log: bool = False) -> jnp.ndarray:
    """
    Sample a parameter at either the genotype or mutation level.

    Parameters
    ----------
    name : str
        The base name for the sample sites.
    data : GrowthData or BindingData
        The data object containing genotype/mutation metadata and batching info.
    sample_fn : callable
        A function with signature (site_name, size) -> samples.
        Used to sample the core effects (either per-genotype or per-mutation).
    is_log : bool, optional
        If True, the parameter is treated as being in log-space for additivity.
        Default is False.

    Returns
    -------
    jnp.ndarray
        The sampled parameter values for the current batch, shape (..., batch_size).
    """
    
    if data.epistasis_mode == "genotype":
        # Standard per-genotype sampling
        with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
            with pyro.handlers.scale(scale=data.scale_vector):
                return sample_fn(name, data.batch_size)

    elif data.epistasis_mode in ["horseshoe", "spikeslab", "none"]:
        # Mutation-level additive model + epistasis
        
        # 1. Sample mutation-level effects
        # These are sampled globally (shared across all genotypes)
        with pyro.plate("shared_mutation_plate", size=data.num_mutation, dim=-1):
             raw_mutation_effects = sample_fn(f"{name}_mut", data.num_mutation)

        # Add a zero for the "wt" mutation at index 0
        mutation_effects = jnp.concatenate([jnp.zeros(raw_mutation_effects.shape[:-1] + (1,)), 
                                            raw_mutation_effects], axis=-1)

        # 2. Sum mutation effects per genotype in the batch
        batch_map = data.map_genotype_to_mutations[data.batch_idx]
        additive_effects = mutation_effects[..., batch_map].sum(axis=-1)
        
        # 3. Handle epistasis
        if data.epistasis_mode == "none":
            return additive_effects

        # Global scales (shared across genotypes in the batch)
        if data.epistasis_mode == "horseshoe":
            tau_epi = pyro.sample(f"{name}_epi_tau", dist.HalfNormal(0.1))
        elif data.epistasis_mode == "spikeslab":
            p_loc = pyro.param(f"{name}_epi_prob_loc", jnp.array(-2.2)) # ~0.1 prob
            p_scale = pyro.param(f"{name}_epi_prob_scale", jnp.array(0.1), constraint=dist.constraints.positive)
            inclusion_prob = pyro.sample(f"{name}_epi_prob", dist.LogNormal(p_loc, p_scale))

        # Local variations (sampled per genotype in the batch)
        with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
            with pyro.handlers.scale(scale=data.scale_vector):
                
                if data.epistasis_mode == "horseshoe":
                    # Local scales
                    lambda_epi = pyro.sample(f"{name}_epi_lambda", dist.HalfNormal(1.0))
                    # Offset
                    z_epi = pyro.sample(f"{name}_epi_z", dist.Normal(0.0, 1.0))
                    epistasis_effects = z_epi * lambda_epi * tau_epi
                
                elif data.epistasis_mode == "spikeslab":
                    w_epi = pyro.sample(f"{name}_epi_w", 
                                        dist.RelaxedBernoulli(temperature=0.1, 
                                                               probs=inclusion_prob))
                    # Base effect (e.g. Normal)
                    z_epi = pyro.sample(f"{name}_epi_z", dist.Normal(0.0, 1.0))
                    epistasis_effects = w_epi * z_epi
        
        # Mask epistasis: only apply if num_mutations > 1
        batch_num_muts = data.genotype_num_mutations[data.batch_idx]
        epistasis_effects = jnp.where(batch_num_muts > 1, epistasis_effects, 0.0)
        
        # 4. Combine
        return additive_effects + epistasis_effects

    else:
        raise ValueError(f"Unknown epistasis_mode: {data.epistasis_mode}")

def sample_genotype_parameter_guide(name: str, 
                                    data: Any, 
                                    guide_fn: Callable[[str, int], jnp.ndarray]) -> jnp.ndarray:
    """
    Guide counterpart for sample_genotype_parameter.
    """
    if data.epistasis_mode == "genotype":
        with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
            with pyro.handlers.scale(scale=data.scale_vector):
                return guide_fn(name, data.batch_size)

    elif data.epistasis_mode in ["horseshoe", "spikeslab", "none"]:
        # 1. Mutation-level guide
        with pyro.plate("shared_mutation_plate", size=data.num_mutation, dim=-1):
            raw_mutation_effects = guide_fn(f"{name}_mut", data.num_mutation)

        mutation_effects = jnp.concatenate([jnp.zeros(raw_mutation_effects.shape[:-1] + (1,)), 
                                            raw_mutation_effects], axis=-1)

        # 2. Sum effects
        batch_map = data.map_genotype_to_mutations[data.batch_idx]
        additive_effects = mutation_effects[..., batch_map].sum(axis=-1)
        
        if data.epistasis_mode == "none":
            return additive_effects

        # 3. Epistasis Guide
        if data.epistasis_mode == "horseshoe":
            # tau (global)
            t_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-5.0))
            t_scale = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.1), constraint=dist.constraints.positive)
            tau_epi = pyro.sample(f"{name}_epi_tau", dist.LogNormal(t_loc, t_scale))
            
            # Local parameters
            l_loc = pyro.param(f"{name}_epi_lambda_loc", jnp.zeros(data.num_genotype))
            l_scale = pyro.param(f"{name}_epi_lambda_scale", jnp.ones(data.num_genotype), constraint=dist.constraints.positive)
            z_loc = pyro.param(f"{name}_epi_z_loc", jnp.zeros(data.num_genotype))
            z_scale = pyro.param(f"{name}_epi_z_scale", jnp.ones(data.num_genotype), constraint=dist.constraints.positive)
            
            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    lambda_epi = pyro.sample(f"{name}_epi_lambda", dist.LogNormal(l_loc[data.batch_idx], l_scale[data.batch_idx]))
                    z_epi = pyro.sample(f"{name}_epi_z", dist.Normal(z_loc[data.batch_idx], z_scale[data.batch_idx]))
                    
            epistasis_effects = z_epi * lambda_epi * tau_epi
        
        elif data.epistasis_mode == "spikeslab":
            # Global prob
            p_loc = pyro.param(f"{name}_epi_prob_loc", jnp.array(-2.2)) # ~0.1 prob
            p_scale = pyro.param(f"{name}_epi_prob_scale", jnp.array(0.1), constraint=dist.constraints.positive)
            inclusion_prob = pyro.sample(f"{name}_epi_prob", dist.LogNormal(p_loc, p_scale))

            # w and z guides (per genotype)
            w_logit = pyro.param(f"{name}_epi_w_logit", jnp.full(data.num_genotype, -5.0))
            z_loc = pyro.param(f"{name}_epi_z_loc", jnp.zeros(data.num_genotype))
            z_scale = pyro.param(f"{name}_epi_z_scale", jnp.ones(data.num_genotype), constraint=dist.constraints.positive)

            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    w_epi = pyro.sample(f"{name}_epi_w", 
                                        dist.RelaxedBernoulli(temperature=0.1, 
                                                               logits=w_logit[data.batch_idx]))
                    z_epi = pyro.sample(f"{name}_epi_z", dist.Normal(z_loc[data.batch_idx], z_scale[data.batch_idx]))
            
            epistasis_effects = w_epi * z_epi

        batch_num_muts = data.genotype_num_mutations[data.batch_idx]
        epistasis_effects = jnp.where(batch_num_muts > 1, epistasis_effects, 0.0)
        
        return additive_effects + epistasis_effects
    
    else:
        raise ValueError(f"Unknown epistasis_mode: {data.epistasis_mode}")

def get_genotype_parameter_guesses(name: str, data: Any, guess_fn: Callable[[str, int], dict]) -> dict:
    """
    Guess counterpart for sample_genotype_parameter.
    """
    if data.epistasis_mode == "genotype":
        return guess_fn(name, data.num_genotype)
    
    elif data.epistasis_mode in ["horseshoe", "spikeslab", "none"]:
        guesses = guess_fn(f"{name}_mut", data.num_mutation)
        
        if data.epistasis_mode == "none":
            return guesses

        if data.epistasis_mode == "horseshoe":
            guesses[f"{name}_epi_tau"] = 0.1
            guesses[f"{name}_epi_lambda"] = jnp.ones(data.num_genotype) * 0.1
            guesses[f"{name}_epi_z"] = jnp.zeros(data.num_genotype)
        
        elif data.epistasis_mode == "spikeslab":
            guesses[f"{name}_epi_prob"] = 0.1
            guesses[f"{name}_epi_w"] = jnp.zeros(data.num_genotype)
            guesses[f"{name}_epi_z"] = jnp.zeros(data.num_genotype)
            
        return guesses
    
    return {}
