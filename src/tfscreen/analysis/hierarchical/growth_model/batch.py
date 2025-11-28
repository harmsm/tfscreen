from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

import jax
import jax.numpy as jnp
from functools import partial

from typing import Any

# Mock type hints for the dataclasses
PRNGKey = Any

@partial(jax.jit, static_argnames=("batch_size",))
def sample_batch(rng_key: PRNGKey, 
                 sharded_data: DataClass, 
                 batch_size: int) -> DataClass:
    """
    Generates a random batch of data by subsampling genotypes.

    This function performs subsampling along the genotype dimension, which
    is assumed to be the last axis of all relevant tensors within
    ``sharded_data.growth``. It slices all tensors in ``sharded_data.growth``
    by the randomly selected genotype indices.

    Per the model design, the ``sharded_data.binding`` attribute is
    passed through unmodified, as the full binding dataset is used in
    each model step. This function is JIT-compiled, with ``batch_size``
    treated as a static argument.

    Parameters
    ----------
    rng_key : PRNGKey
        A JAX random key used for sampling genotype indices.
    sharded_data : DataClass
        The *full* ``DataClass`` object containing all growth and binding
        data.
    batch_size : int
        The number of genotypes to sample. This must be static for
        JIT compilation.

    Returns
    -------
    DataClass
        A new ``DataClass`` pytree where ``data.growth`` tensors have been
        sliced to ``batch_size`` along the last dimension. ``data.binding``
        remains unmodified.

    Raises
    ------
    ValueError
        If ``batch_size > total_size`` (from ``jax.random.choice`` with
        ``replace=False``).
    """

    # Generate a collection of random sample indexes
    total_size = sharded_data.growth.num_genotype
    idx = jax.random.choice(rng_key, total_size, shape=(batch_size,), replace=False)

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
            log_titrant_conc=sharded_data.growth.log_titrant_conc[..., idx],
            map_theta_group=sharded_data.growth.map_theta_group[..., idx],
            
            not_wt_mask=sharded_data.growth.not_wt_mask[..., idx],
            good_mask=sharded_data.growth.good_mask[..., idx],
        )
    )

    return batch_data
    
def deterministic_batch(full_data: DataClass, idx: jnp.ndarray) -> DataClass:
    """
    Extracts a deterministic batch of data given specific indices.

    This function performs subsampling along the genotype dimension (assumed
    to be the last axis) using a provided array of indices. It slices all
    tensors within the ``full_data.growth`` attribute.

    Per the model design, the ``full_data.binding`` attribute is passed
    through unmodified, as the full binding dataset is used in each
    model step.

    Parameters
    ----------
    full_data : DataClass
        The *full* ``DataClass`` object containing all growth and binding
        data.
    idx : jnp.ndarray
        An array of integer indices to use for slicing the genotype dimension.

    Returns
    -------
    DataClass
        A new ``DataClass`` pytree where ``data.growth`` tensors have been
        sliced according to ``idx`` along the last dimension.
        ``data.binding`` remains unmodified.
    """
    
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
            log_titrant_conc=full_data.growth.log_titrant_conc[..., idx],
            map_theta_group=full_data.growth.map_theta_group[..., idx],
            
            not_wt_mask=full_data.growth.not_wt_mask[..., idx],
            good_mask=full_data.growth.good_mask[..., idx],
        )
    )

    return batch_data