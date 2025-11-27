
import jax.numpy as jnp
from flax.struct import (
    dataclass,
    field
)
from typing import Any


@dataclass(frozen=True)
class GrowthData:

    # Main data tensors
    ln_cfu: jnp.ndarray
    ln_cfu_std: jnp.ndarray
    
    # Fixed experimental parameters
    t_pre: jnp.ndarray
    t_sel: jnp.ndarray
    
    # mappers
    map_ln_cfu0: jnp.ndarray
    map_condition_pre: jnp.ndarray
    map_condition_sel: jnp.ndarray
    map_genotype: jnp.ndarray
    map_theta: jnp.ndarray 
    
    # Tensor shape (static)
    num_replicate: int = field(pytree_node=False)
    num_time: int = field(pytree_node=False)
    num_treatment: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False)

    # other lengths for plates (various total sizes)
    num_ln_cfu0: int = field(pytree_node=False)
    num_condition: int = field(pytree_node=False)
    num_theta: int = field(pytree_node=False)
    
    # small tensor used for theta calculations
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray
    map_theta_group: jnp.ndarray
    num_titrant_name: int  = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)

    # meta data
    wt_index: int = field(pytree_node=False)
    num_not_wt: int = field(pytree_node=False)
    not_wt_mask: jnp.ndarray
    good_mask: jnp.ndarray
    scatter_theta: int = field(pytree_node=False) 

@dataclass(frozen=True)
class BindingData:

    theta_obs: jnp.ndarray
    theta_std: jnp.ndarray
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray
    map_theta_group: jnp.ndarray

    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False) 
    
    good_mask: jnp.ndarray
    scatter_theta: int = field(pytree_node=False) 


@dataclass(frozen=True)
class DataClass:
    """
    A container holding data needed to specify growth_model, treated as a JAX
    Pytree.
    """
    
    num_genotype: int = field(pytree_node=False)

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
class ControlClass:
     condition_growth: int = field(pytree_node=False)
     ln_cfu0: int = field(pytree_node=False)
     dk_geno: int = field(pytree_node=False)
     activity: int = field(pytree_node=False)
     theta: int = field(pytree_node=False)
     theta_growth_noise: int = field(pytree_node=False)
     theta_binding_noise: int = field(pytree_node=False)

@dataclass(frozen=True)
class PriorsClass:
    
    ## GrowthPriors and BindingPriors
    theta: BindingPriors
    growth: GrowthPriors
    binding: BindingPriors
