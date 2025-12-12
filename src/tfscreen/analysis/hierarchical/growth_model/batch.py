from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

import jax
import jax.numpy as jnp

def random_batch(sample_key, full_data: DataClass):

    choose_from = full_data.not_binding_idx
    size = (full_data.not_binding_batch_size,)
    batch_idx = jax.random.choice(sample_key,
                                  choose_from,
                                  size,
                                  replace=False)

    idx = full_data.batch_idx.at[full_data.num_binding:].set(batch_idx)

    return get_batch(full_data,idx)


def get_batch(full_data: DataClass, idx: jnp.ndarray) -> DataClass:
    """
    Extracts a deterministic batch of data given specific indices.
    """
    
    batch_size = len(idx)
    batch_data = full_data.replace(
        growth=full_data.growth.replace(
            batch_size=batch_size,
            batch_idx=idx,
            scale_vector=full_data.growth.scale_vector[...,idx],
            ln_cfu=full_data.growth.ln_cfu[...,idx],
            ln_cfu_std=full_data.growth.ln_cfu_std[...,idx],
            t_pre=full_data.growth.t_pre[...,idx],
            t_sel=full_data.growth.t_sel[...,idx],
            map_condition_pre=full_data.growth.map_condition_pre[...,idx],
            map_condition_sel=full_data.growth.map_condition_sel[...,idx],
            good_mask=full_data.growth.good_mask[...,idx],
        )
    )

    return batch_data
