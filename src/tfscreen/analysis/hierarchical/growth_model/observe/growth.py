import numpyro as pyro
from numpyro.handlers import mask
import numpyro.distributions as dist

def observe(name,data,ln_cfu_pred):

    # Get batch size from the subsample
    batch_size = data.ln_cfu.shape[-1] 

    # Growth observation
    with pyro.plate(f"{name}_replicate", size=data.num_replicate, dim=-4):
        with pyro.plate(f"{name}_time", size=data.num_time, dim=-3):
            with pyro.plate(f"{name}_treatment", size=data.num_treatment, dim=-2):

                # Batching of innermost genotype plate (shared with binding) 
                with pyro.plate(f"{name}_genotype", 
                                size=data.num_genotype,    
                                subsample_size=batch_size, 
                                dim=-1):
                    
                    with mask(mask=data.good_mask):
                        pyro.sample(f"{name}_growth_obs",
                                    dist.Normal(ln_cfu_pred, data.ln_cfu_std),
                                    obs=data.ln_cfu)

