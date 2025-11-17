import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=("batch_size",))
def sample_batch(rng_key,sharded_data,batch_size):
    """
    Generate a random sample of the data. 
    """

    # Generate a collection of random sample indexes
    total_size = sharded_data.growth.num_genotype
    idx = jax.random.choice(rng_key, total_size, shape=(batch_size,), replace=False)

    # ----- This existed to do a batched sample of binding data ----
    # growth_to_binding_idx is the map between the growth tensor genotype
    # indexes and their indexes in the binding dataset. These will be -1 if
    # there is no measured value. Sample from this with idx and create a 
    # mask for whether the value is True or not. 
    #binding_idx_padded = sharded_data.growth_to_binding_idx[idx]
    #obs_mask = binding_idx_padded >= 0

    # Replace any -1 with 0 to prevent out-of-bounds errors. This is fine, as
    # these entries will be masked out.
    #binding_idx = jnp.where(obs_mask,binding_idx_padded,0)

    batch_data = sharded_data.replace(
        growth=sharded_data.growth.replace(
            
            # Full data tensors
            ln_cfu=sharded_data.growth.ln_cfu[..., idx],
            ln_cfu_std=sharded_data.growth.ln_cfu_std[..., idx],
            t_pre=sharded_data.growth.t_pre[..., idx],
            t_sel=sharded_data.growth.t_sel[..., idx],

            # Full map tensors
            map_ln_cfu0=sharded_data.growth.map_ln_cfu0[..., idx],
            map_condition_pre=sharded_data.growth.map_condition_pre[..., idx],
            map_condition_sel=sharded_data.growth.map_condition_sel[..., idx],
            map_genotype=sharded_data.growth.map_genotype[..., idx],
            map_theta=sharded_data.growth.map_theta[..., idx],

            # small theta tensors
            titrant_conc=sharded_data.growth.titrant_conc[..., idx],
            map_theta_group=sharded_data.growth.map_theta_group[..., idx],
            
            good_mask=sharded_data.growth.good_mask[..., idx],
        ),

        # ----- This existed to do a batched sample of binding data ----
        # binding=sharded_data.binding.replace(
        #     theta_obs=sharded_data.binding.theta_obs[..., binding_idx],
        #     theta_std=sharded_data.binding.theta_std[..., binding_idx],
        #     titrant_conc=sharded_data.binding.titrant_conc[..., binding_idx],
        #     map_theta_group=sharded_data.binding.map_theta_group[..., binding_idx],
        #     good_mask=sharded_data.binding.good_mask[..., binding_idx],
        #     obs_mask=obs_mask,
        # )

    )


    return batch_data
    
def deterministic_batch(full_data, idx):
    """
    Slice the full data to create a deterministic batch. 
    """
    
    # growth_to_binding_idx is the map between the growth tensor genotype
    # indexes and their indexes in the binding dataset. These will be -1 if
    # there is no measured value. Sample from this with idx and create a 
    # mask for whether the value is True or not. 
    binding_idx_padded = full_data.growth_to_binding_idx[idx]
    obs_mask = binding_idx_padded >= 0

    # ----- This existed to do a batched sample of binding data ----
    # Replace any -1 with 0 to prevent out-of-bounds errors. This is fine, as
    # these entries will be masked out.
    #binding_idx = jnp.where(obs_mask,binding_idx_padded,0)

    batch_data = full_data.replace(
        growth=full_data.growth.replace(

            # Full data tensors
            ln_cfu=full_data.growth.ln_cfu[..., idx],
            ln_cfu_std=full_data.growth.ln_cfu_std[..., idx],
            t_pre=full_data.growth.t_pre[..., idx],
            t_sel=full_data.growth.t_sel[..., idx],

            # Full map tensors
            map_ln_cfu0=full_data.growth.map_ln_cfu0[..., idx],
            map_condition_pre=full_data.growth.map_condition_pre[..., idx],
            map_condition_sel=full_data.growth.map_condition_sel[..., idx],
            map_genotype=full_data.growth.map_genotype[..., idx],
            map_theta=full_data.growth.map_theta[..., idx],

            # small theta tensors
            titrant_conc=full_data.growth.titrant_conc[..., idx],
            map_theta_group=full_data.growth.map_theta_group[..., idx],
            
            good_mask=full_data.growth.good_mask[..., idx],
        ),
        # ----- This existed to do a batched sample of binding data ----
        # binding=full_data.binding.replace(
        #     theta_obs=full_data.binding.theta_obs[..., binding_idx],
        #     theta_std=full_data.binding.theta_std[..., binding_idx],
        #     titrant_conc=full_data.binding.titrant_conc[..., binding_idx],
        #     map_theta_group=full_data.binding.map_theta_group[..., binding_idx],
        #     good_mask=full_data.binding.good_mask[..., binding_idx],
        #     obs_mask=obs_mask,
        # )
    )

    return batch_data