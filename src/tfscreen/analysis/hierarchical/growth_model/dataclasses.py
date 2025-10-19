import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class GrowthModelData:
    """
    A container holding data needed to specify growth_model, treated as a JAX
    Pytree.
    """
    
    # Main data tensors
    ln_cfu_obs: jnp.ndarray
    ln_cfu_std: jnp.ndarray
    
    # Fixed experimental parameters
    t_pre: jnp.ndarray
    t_sel: jnp.ndarray

    # Indexes
    wt_index: int
    
    # Mapping arrays
    cond_pre_map: jnp.ndarray
    cond_sel_map: jnp.ndarray
    theta_map: jnp.ndarray
    ln_cfu0_block_map: jnp.ndarray

    # masks
    good_mask: jnp.ndarray
    not_wt_mask: jnp.ndarray

    # counts
    num_geno: int
    num_rep: int
    num_cond: int
    num_time: int
    num_raw_cond: int
    num_theta: int
    num_ln_cfu0_block: int

@dataclass 
class GrowthModelPriors:
    """
    A container holding data needed to specify growth_model priors, treated
    as a JAX Pytree.
    """

    ln_cfu0_hyper_loc_loc: float
    ln_cfu0_hyper_loc_scale: float
    ln_cfu0_hyper_scale_loc: float
    
    A_hyper_loc_loc: float
    A_hyper_loc_scale: float
    A_hyper_scale_loc: float

    dk_geno_hyper_shift_loc: float
    dk_geno_hyper_shift_scale: float
    dk_geno_hyper_loc_loc: float
    dk_geno_hyper_loc_scale: float
    dk_geno_hyper_scale_loc: float
    
    wt_theta_loc: jnp.ndarray
    wt_theta_scale: jnp.ndarray
    
    log_alpha_hyper_loc_loc: float
    log_alpha_hyper_loc_scale: float
    log_beta_hyper_loc_loc: float
    log_beta_hyper_loc_scale: float
    
    growth_k_loc: float
    growth_k_scale: float
    growth_m_loc: float
    growth_m_scale: float

@dataclass
class GrowthModelParameters:

    geno_to_idx: dict
    replicate_to_idx: dict
    condition_to_idx: dict
    raw_cond_to_idx: dict
    titr_conc_to_idx: dict