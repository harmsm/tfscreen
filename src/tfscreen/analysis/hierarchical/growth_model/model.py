
import jax.numpy as jnp
import numpyro as pyro
from numpyro.handlers import mask
import numpyro.distributions as dist

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


def jax_model(data,
              growth_priors,
              ln_cfu0_priors,
              dk_geno_priors,
              activity_priors,
              theta_priors,
              use_growth_indep=False,
              use_fixed_dk_geno=False,
              use_fixed_activity=False,
              no_horseshoe_activity=False,
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
        if no_horseshoe_activity:
            activity = define_activity("activity",data,activity_priors)
        else:
            activity = define_activity_horseshoe("activity",data,activity_priors)
            
    # Define theta
    if use_categorical_theta:
        theta = define_theta_cat("theta",data,theta_priors)
    else:
        theta = define_theta_hill("theta",data,theta_priors)

    # Calculate growth. All variables have same tensor dimensions. 
    g_pre = k_pre + dk_geno + activity*m_pre*theta
    g_sel = k_sel + dk_geno + activity*m_sel*theta
    ln_cfu_pred = ln_cfu0 + g_pre*data.t_pre + g_sel*data.t_sel

    # Make sure that all values in ln_cfu_std are greater than zero. (The 
    # data will have ensured is all non-nan).
    safe_std = jnp.clip(data.ln_cfu_std, a_min=1e-9)
    
    # Get size of this batch
    batch_size = data.ln_cfu.shape[-1]

    # Observe data
    with pyro.plate("main_replicate",size=data.num_replicate,dim=-4):
        with pyro.plate("main_time",size=data.num_time,dim=-3):
            with pyro.plate("main_treatment",size=data.num_treatment,dim=-2):
                with pyro.plate("main_genotype",size=data.num_genotype,subsample_size=batch_size,dim=-1):
                    with mask(mask=data.good_mask):
                        pyro.sample("obs",
                                    dist.Normal(ln_cfu_pred,safe_std),
                                    obs=data.ln_cfu)

