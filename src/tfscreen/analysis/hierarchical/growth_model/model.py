import jax.numpy as jnp
import numpyro as pyro
from numpyro.handlers import mask
import numpyro.distributions as dist
from flax.struct import dataclass

from .components.growth_hyper import define_model as define_growth_hyper
from .components.growth_indep import define_model as define_growth_indep

from .components.ln_cfu0 import define_model as define_ln_cfu0

from .components.dk_geno import define_model as define_dk_geno
from .components.dk_geno_fixed import define_model as define_dk_geno_fixed

from .components.activity import define_model as define_activity
from .components.activity_horseshoe import define_model as define_activity_horseshoe
from .components.activity_fixed import define_model as define_activity_fixed

from .components.theta_cat import define_model as define_theta_cat
from .components.theta_hill import define_model as define_theta_hill

@dataclass
class GrowthModelData:
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

    # lengths for plates
    num_ln_cfu0: int
    num_condition: int
    num_genotype: int
    num_theta: int
    num_replicate: int
    num_not_wt: int
    num_titrant: int
    num_theta_group: int

    # meta data
    wt_index: int
    not_wt_mask: jnp.ndarray
    good_mask: jnp.ndarray


def growth_model(data,
                 growth_priors,
                 ln_cfu0_priors,
                 dk_geno_priors,
                 activity_priors,
                 theta_priors,
                 use_growth_indep=False,
                 use_fixed_dk_geno=False,
                 use_fixed_activity=False,
                 use_horseshoe_activity=True,
                 use_categorical_theta=False):
    """
    Model growth rates of bacteria in culture.
    """

    # Define base growth parameters
    if use_growth_indep:
        k_pre, m_pre, k_sel, m_sel = define_growth_indep("growth",data,growth_priors)
    else:
        k_pre, m_pre, k_sel, m_sel = define_growth_hyper("growth",data,growth_priors)

    # Define ln_cfu0
    ln_cfu0 = define_ln_cfu0("ln_cfu0",data,ln_cfu0_priors)

    # Define dk_geno
    if use_fixed_dk_geno:
        dk_geno = define_dk_geno_fixed("dk_geno",data,dk_geno_priors)    
    else:
        dk_geno = define_dk_geno("dk_geno",data,dk_geno_priors)

    # Define activity
    if use_fixed_activity:
        activity = define_activity_fixed("activity",data,activity_priors)
    else:
        if use_horseshoe_activity:
            activity = define_activity_horseshoe("activity",data,activity_priors)
        else:
            activity = define_activity("activity",data,activity_priors)

    # Define theta
    if use_categorical_theta:
        theta = define_theta_cat("theta",data,theta_priors)
    else:
        theta = define_theta_hill("theta",data,theta_priors)

    # Calculate growth. All variables have same tensor dimensions. 
    g_pre = k_pre + dk_geno + activity*m_pre*theta
    g_sel = k_sel + dk_geno + activity*m_sel*theta
    ln_cfu_pred = ln_cfu0 + g_pre*data.t_pre + g_sel*data.t_sel

    # Observe data
    with mask(mask=data.good_mask):
        pyro.sample("obs",
                    dist.Normal(ln_cfu_pred,data.ln_cfu_std),
                    obs=data.ln_cfu)

