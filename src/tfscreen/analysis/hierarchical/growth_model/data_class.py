
import jax.numpy as jnp
from flax.struct import (
    dataclass,
    field
)


@dataclass(frozen=True)
class DataClass:
    """
    A container holding data needed to specify growth_model, treated as a JAX
    Pytree.
    """
    
    # Main data tensors
    ln_cfu: jnp.ndarray
    ln_cfu_std: jnp.ndarray
    
    # Fixed experimental parameters
    t_pre: jnp.ndarray
    t_sel: jnp.ndarray
    titrant_conc: jnp.ndarray

    # mappers
    map_ln_cfu0: jnp.ndarray
    map_cond_pre: jnp.ndarray
    map_cond_sel: jnp.ndarray
    map_genotype: jnp.ndarray
    map_theta: jnp.ndarray
    map_theta_group: jnp.ndarray

    # Tensor shape (static)
    tensor_shape_i: int = field(pytree_node=False)
    tensor_shape_j: int = field(pytree_node=False)
    tensor_shape_k: int = field(pytree_node=False)
    tensor_shape_l: int = field(pytree_node=False)

    # lengths for plates (various total sizes)
    num_ln_cfu0: int = field(pytree_node=False)
    num_condition: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False)
    num_theta: int = field(pytree_node=False)
    num_replicate: int = field(pytree_node=False)
    num_titrant: int = field(pytree_node=False)
    num_theta_group: int = field(pytree_node=False)

    # meta data
    wt_index: int = field(pytree_node=False)
    num_not_wt: int = field(pytree_node=False)
    not_wt_mask: jnp.ndarray
    good_mask: jnp.ndarray

