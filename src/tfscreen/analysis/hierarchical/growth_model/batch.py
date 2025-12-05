from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

import jax.numpy as jnp

def generate_batch(full_data: DataClass, idx: jnp.ndarray) -> DataClass:
    """
    Extracts a deterministic batch of data given specific indices.
    """
    
    batch_size = len(idx)
    batch_data = full_data.replace(
        growth=full_data.growth.replace(
            batch_size=batch_size,
            batch_idx=idx,
            ln_cfu=full_data.growth.ln_cfu[...,idx],
            ln_cfu_std=full_data.growth.ln_cfu_std[...,idx],
            t_pre=full_data.growth.t_pre[...,idx],
            t_sel=full_data.growth.t_sel[...,idx],
            map_condition_pre=full_data.growth.map_condition_pre[...,idx],
            map_condition_sel=full_data.growth.map_condition_sel[...,idx],
            good_mask=full_data.growth.good_mask[...,idx],
        ),
        binding=full_data.binding.replace(
            batch_size=batch_size,
            batch_idx=idx,
            theta_obs=full_data.binding.theta_obs[...,idx],
            theta_std=full_data.binding.theta_std[...,idx],
            good_mask=full_data.binding.good_mask[...,idx],
        )
    )

    return batch_data
