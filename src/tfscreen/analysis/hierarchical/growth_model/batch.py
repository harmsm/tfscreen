from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

import jax.numpy as jnp

def store_batch_indices(full_data: DataClass, idx: jnp.ndarray) -> DataClass:
    """
    Extracts a deterministic batch of data given specific indices.
    """
    

    batch_size = len(idx)
    batch_data = full_data.replace(
        growth=full_data.growth.replace(
            batch_size=batch_size,
            batch_idx=idx,
        ),
        binding=full_data.binding.replace(
            batch_size=batch_size,
            batch_idx=idx,
        )
    )

    return batch_data