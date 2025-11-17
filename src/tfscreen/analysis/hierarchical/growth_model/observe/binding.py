import numpyro as pyro
from numpyro.handlers import mask
import numpyro.distributions as dist


def observe(name,data,binding_pred):

        # Binding observation
    with pyro.plate(f"{name}_binding_titrant_name", size=data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_binding_titrant_conc", size=data.num_titrant_conc, dim=-2):

            # Batching of innermost genotype plate (shared with ln_cfu) 
            with pyro.plate(f"{name}_binding_genotype", size=data.num_genotype,dim=-1):
                
                # static mask is for good observations; obs_mask is for sampled
                # observations in this batch. 
                with mask(mask=data.good_mask):

                    pyro.sample(f"{name}_binding_obs",
                                dist.Normal(binding_pred, data.theta_std),
                                obs=data.theta_obs)