from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

import jax
import jax.numpy as jnp

def get_batch(full_data: DataClass, idx: jnp.ndarray) -> DataClass:
    """
    Extracts a deterministic batch of data given specific indices.

    This function creates a new `DataClass` structure containing subsets of
    the growth and/or binding data corresponding to the indices in `idx`.
    It handles slicing of all relevant tensors (ln_cfu, std, time points,
    masks, theta observations) and updates the batch size information.

    In the joint model (growth + binding), only growth tensors are sliced;
    the binding tensors are always included in full because binding genotypes
    are always part of every growth mini-batch.

    In binding-only mode (growth is None), the binding tensors ARE sliced
    along the genotype axis so that true mini-batching over binding genotypes
    is supported.

    Parameters
    ----------
    full_data : DataClass
        The complete dataset containing all observations.
    idx : jnp.ndarray
        An array of indices (integers) specifying which genotypes to include
        in the batch.

    Returns
    -------
    DataClass
        A new DataClass instance containing only the batched data.
    """

    batch_size = len(idx)

    if full_data.growth is not None:
        # Joint model: slice growth tensors; binding tensors are unchanged
        # because binding genotypes are always included in every mini-batch.
        new_growth = full_data.growth.replace(
            batch_size=batch_size,
            batch_idx=idx,
            scale_vector=full_data.growth.scale_vector[...,idx],
            geno_theta_idx=jnp.arange(batch_size, dtype=jnp.int32),
            ln_cfu=full_data.growth.ln_cfu[...,idx],
            ln_cfu_std=full_data.growth.ln_cfu_std[...,idx],
            t_pre=full_data.growth.t_pre[...,idx],
            t_sel=full_data.growth.t_sel[...,idx],
            map_condition_pre=full_data.growth.map_condition_pre[...,idx],
            map_condition_sel=full_data.growth.map_condition_sel[...,idx],
            good_mask=full_data.growth.good_mask[...,idx],
            congression_mask=full_data.growth.congression_mask[...,idx],
        )
        return full_data.replace(growth=new_growth)

    # Binding-only mode: idx is a set of binding-genotype indices.
    # Slice the binding tensors so the model sees only the current mini-batch.
    if full_data.binding is not None:
        new_binding = full_data.binding.replace(
            batch_size=batch_size,
            batch_idx=idx,
            scale_vector=full_data.binding.scale_vector[idx],
            geno_theta_idx=jnp.arange(batch_size, dtype=jnp.int32),
            theta_obs=full_data.binding.theta_obs[..., idx],
            theta_std=full_data.binding.theta_std[..., idx],
            good_mask=full_data.binding.good_mask[..., idx],
        )
        return full_data.replace(binding=new_binding)

    # No data at all (shouldn't happen in practice, but handle gracefully)
    return full_data
