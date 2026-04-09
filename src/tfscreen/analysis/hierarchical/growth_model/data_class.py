
import jax.numpy as jnp
import numpy as np
from flax.struct import (
    dataclass,
    field
)
from typing import Any


@dataclass(frozen=True)
class GrowthData:
    
    # Batch information
    batch_size: int = field(pytree_node=False)
    batch_idx: jnp.ndarray
    scale_vector: jnp.ndarray
    geno_theta_idx: jnp.ndarray

    # Data tensors
    ln_cfu: jnp.ndarray
    ln_cfu_std: jnp.ndarray
    t_pre: jnp.ndarray
    t_sel: jnp.ndarray
    good_mask: jnp.ndarray
    congression_mask: jnp.ndarray

        
    # Tensor shape
    num_replicate: int = field(pytree_node=False)
    num_time: int = field(pytree_node=False)
    num_condition_pre: int = field(pytree_node=False)
    num_condition_sel: int = field(pytree_node=False)
    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False)

    # mappers
    num_condition_rep: int = field(pytree_node=False)
    map_condition_pre: jnp.ndarray
    map_condition_sel: jnp.ndarray

    # 1D arrays of titrant concentration (corresponds to the second-to-last 
    # tensor dimension)
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray

    # meta data
    wt_indexes: jnp.ndarray
    scatter_theta: int = field(pytree_node=False)

    # Boolean mask, shape (num_genotype,), True = spiked genotype.
    # Used by ln_cfu0 component to apply a separate prior location.
    ln_cfu0_spiked_mask: jnp.ndarray

    growth_shares_replicates: bool = field(pytree_node=False, default=False)

    # Optional mutation-decomposition matrices (set when using *_mut_decomp components).
    # Stored as pytree_node=False so they are treated as static by JAX tracing.
    # Shape: mut_geno_matrix (num_mutation, num_genotype),
    #        pair_geno_matrix (num_pair, num_genotype).
    num_mutation: int = field(pytree_node=False, default=0)
    num_pair: int = field(pytree_node=False, default=0)
    mut_geno_matrix: Any = field(pytree_node=False, default=None)
    pair_geno_matrix: Any = field(pytree_node=False, default=None)

@dataclass(frozen=True)
class BindingData:

    # Batch information
    batch_size: int = field(pytree_node=False)
    batch_idx: jnp.ndarray
    scale_vector: jnp.ndarray
    geno_theta_idx: jnp.ndarray

    # Main data tensors
    theta_obs: jnp.ndarray
    theta_std: jnp.ndarray
    good_mask: jnp.ndarray

    # Tensor dimensions
    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False) 

    # 1D arrays of titrant concentration (corresponds to the second-to-last 
    # tensor dimension)
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray

    scatter_theta: int = field(pytree_node=False) 


@dataclass(frozen=True)
class DataClass:
    """
    A container holding data needed to specify growth_model, treated as a JAX
    Pytree.
    """

    num_genotype: int = field(pytree_node=False)
    
    batch_idx: jnp.ndarray
    batch_size: int = field(pytree_node=False)

    not_binding_idx: jnp.ndarray
    not_binding_batch_size: int = field(pytree_node=False)
    num_binding: int = field(pytree_node=False) 

    # This will be a GrowthData and BindingData
    growth: GrowthData
    binding: BindingData


@dataclass(frozen=True)
class GrowthPriors:
    condition_growth: Any
    growth_transition: Any
    ln_cfu0: Any
    dk_geno: Any
    activity: Any
    transformation: Any
    theta_growth_noise: Any

@dataclass(frozen=True)
class BindingPriors:
    theta_binding_noise: Any


@dataclass(frozen=True)
class PriorsClass:
    
    ## GrowthPriors and BindingPriors
    theta: BindingPriors
    growth: GrowthPriors
    binding: BindingPriors
