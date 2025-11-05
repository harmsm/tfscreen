import jax
from functools import partial

@partial(jax.jit, static_argnames=("batch_size",))
def sample_batch(rng_key,sharded_data,batch_size):
    """
    Generate a random sample of the data 
    """

    total_size = sharded_data.num_genotype
    idx = jax.random.choice(rng_key, total_size, shape=(batch_size,), replace=False)

    batch_data = sharded_data.replace(
        ln_cfu=sharded_data.ln_cfu[..., idx],
        ln_cfu_std=sharded_data.ln_cfu_std[..., idx],
        t_pre=sharded_data.t_pre[..., idx],
        t_sel=sharded_data.t_sel[..., idx],
        titrant_conc=sharded_data.titrant_conc[..., idx],
        map_ln_cfu0=sharded_data.map_ln_cfu0[..., idx],
        map_cond_pre=sharded_data.map_cond_pre[..., idx],
        map_cond_sel=sharded_data.map_cond_sel[..., idx],
        map_genotype=sharded_data.map_genotype[..., idx],
        map_theta=sharded_data.map_theta[..., idx],
        map_theta_group=sharded_data.map_theta_group[..., idx],
        good_mask=sharded_data.good_mask[..., idx]
    )

    return batch_data
    
def deterministic_batch(full_data, idx):
    """
    Slice the full data to create a deterministic batch. 
    """
    
    batch_data = full_data.replace(
        ln_cfu=full_data.ln_cfu[..., idx],
        ln_cfu_std=full_data.ln_cfu_std[..., idx],
        t_pre=full_data.t_pre[..., idx],
        t_sel=full_data.t_sel[..., idx],
        titrant_conc=full_data.titrant_conc[..., idx],
        map_ln_cfu0=full_data.map_ln_cfu0[..., idx],
        map_cond_pre=full_data.map_cond_pre[..., idx],
        map_cond_sel=full_data.map_cond_sel[..., idx],
        map_genotype=full_data.map_genotype[..., idx],
        map_theta=full_data.map_theta[..., idx],
        map_theta_group=full_data.map_theta_group[..., idx],
        good_mask=full_data.good_mask[..., idx]
    )

    return batch_data