
import numpyro as pyro
from numpyro.handlers import mask
import numpyro.distributions as dist

def observe(name,data,ln_cfu_pred,binding_pred):

    # Get batch size from the subsample
    batch_size = data.growth.ln_cfu.shape[-1] 

    # Growth observation
    with pyro.plate(f"{name}_replicate", size=data.growth.num_replicate, dim=-4):
        with pyro.plate(f"{name}_time", size=data.growth.num_time, dim=-3):
            with pyro.plate(f"{name}_treatment", size=data.growth.num_treatment, dim=-2):

                # Batching of innermost genotype plate (shared with binding) 
                with pyro.plate(f"{name}_genotype", 
                                size=data.growth.num_genotype,    
                                subsample_size=batch_size, 
                                dim=-1):
                    
                    with mask(mask=data.growth.good_mask):
                        pyro.sample(f"{name}_growth_obs",
                                    dist.Normal(ln_cfu_pred, data.growth.ln_cfu_std),
                                    obs=data.growth.ln_cfu)

    # Binding observation
    with pyro.plate(f"{name}_titrant_name", size=data.binding.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_titrant_conc", size=data.binding.num_titrant_conc, dim=-2):

            # Batching of innermost genotype plate (shared with ln_cfu) 
            with pyro.plate(f"{name}_genotype", 
                            size=data.num_genotype,     
                            subsample_size=batch_size,  
                            dim=-1):
                
                # static mask is for good observations; obs_mask is for sampled
                # observations in this batch. 
                with mask(mask=data.binding.good_mask):
                    pyro.sample(f"{name}_binding_obs",
                                dist.Normal(binding_pred, data.binding.theta_std),
                                obs=data.binding.theta_obs,
                                obs_mask=data.binding.obs_mask)