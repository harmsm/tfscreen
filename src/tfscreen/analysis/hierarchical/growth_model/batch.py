from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

import jax
import jax.numpy as jnp

def get_batch(full_data: DataClass, idx: jnp.ndarray) -> DataClass:
    """
    Extracts a deterministic batch of data given specific indices.

    This function creates a new `DataClass` structure containing subsets of
    the growth and binding data corresponding to the indices in `idx`.
    It handles slicing of all relevant tensors (ln_cfu, std, time points,
    masks) and updates the batch size information.

    Parameters
    ----------
    full_data : DataClass
        The complete dataset containing all observations.
    idx : jnp.ndarray
        An array of indices (integers) specifying which elements (genotypes/
        conditions) to include in the batch.

    Returns
    -------
    DataClass
        A new DataClass instance containing only the batched data.
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
