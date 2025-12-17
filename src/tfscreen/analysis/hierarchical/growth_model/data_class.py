
import jax.numpy as jnp
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
        
    # Tensor shape
    num_replicate: int = field(pytree_node=False)
    num_time: int = field(pytree_node=False)
    num_condition_pre: int = field(pytree_node=False)
    num_condition_sel: int = field(pytree_node=False)
    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False)

    # mappers
    num_condition: int = field(pytree_node=False)
    map_condition_pre: jnp.ndarray
    map_condition_sel: jnp.ndarray

    # 1D arrays of titrant concentration (corresponds to the second-to-last 
    # tensor dimension)
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray

    # meta data
    wt_indexes: jnp.ndarray    
    scatter_theta: int = field(pytree_node=False) 

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
    
    not_binding_idx: jnp.ndarray
    not_binding_batch_size: int = field(pytree_node=False)
    num_binding: int = field(pytree_node=False) 

    # This will be a GrowthData and BindingData
    growth: GrowthData
    binding: BindingData


@dataclass(frozen=True)
class GrowthPriors:
    condition_growth: Any
    ln_cfu0: Any
    dk_geno: Any
    activity: Any
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
