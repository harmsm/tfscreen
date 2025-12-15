import jax.numpy as jnp
import numpyro as pyro

import numpyro.distributions as dist

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

def observe(name: str,
            data: GrowthData,
            ln_cfu_pred: jnp.ndarray):
    """
    Defines the observation site for the growth (ln_CFU) data.

    This function links the predicted log-CFU values (`ln_cfu_pred`) to the
    observed data (`data.ln_cfu`) using a Normal likelihood.
    It operates over a 4D tensor defined by replicate, time, treatment,
    and genotype.

    This function is designed to work with mini-batching (subsampling)
    on the innermost (`genotype`) dimension. It also applies a
    mask (`data.good_mask`) to exclude bad-quality or missing
    observations from the log-likelihood calculation.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/plate sites.
    data : GrowthData
        A Pytree (Flax dataclass) containing the observed growth data
        and metadata. This function primarily uses:
        - ``data.ln_cfu`` : (jnp.ndarray) Observed ln(CFU) data,
          potentially a mini-batch.
        - ``data.ln_cfu_std`` : (jnp.ndarray) Observed std dev of ln(CFU).
        - ``data.num_replicate`` : (int)
        - ``data.num_time`` : (int)
        - ``data.num_treatment`` : (int)
        - ``data.num_genotype`` : (int) Total size of the genotype dimension.
        - ``data.good_mask`` : (jnp.ndarray) Boolean mask for valid data.
    ln_cfu_pred : jnp.ndarray
        The deterministically predicted ln(CFU) values from the model,
        with a shape matching `data.ln_cfu`.
    """

    nu = pyro.sample(f"{name}_nu", dist.Gamma(2.0, 0.1))

    # Growth observation
    with pyro.plate(f"{name}_replicate", size=data.num_replicate, dim=-7):
        with pyro.plate(f"{name}_time", size=data.num_time, dim=-6):
            with pyro.plate(f"{name}_condition_pre", size=data.num_condition_pre, dim=-5):
                with pyro.plate(f"{name}_condition_sel", size=data.num_condition_sel, dim=-4):
                    with pyro.plate(f"{name}_titrant_name", size=data.num_titrant_name, dim=-3):
                        with pyro.plate(f"{name}_titrant_conc", size=data.num_titrant_conc, dim=-2):
                            with pyro.plate("shared_genotype_plate",size=data.batch_size,dim=-1):

                                # Scale data for sub-sampling
                                with pyro.handlers.scale(scale=data.scale_vector):

                                    # Apply mask for good observations
                                    with pyro.handlers.mask(mask=data.good_mask):
                                        
                                        # Define the observation site
                                        pyro.sample(f"{name}_growth_obs",
                                                    dist.StudentT(df=nu, loc=ln_cfu_pred, scale=data.ln_cfu_std),
                                                    obs=data.ln_cfu)

def guide(name: str,
          data: GrowthData,
          ln_cfu_pred: jnp.ndarray):
    """
    Guide corresponding to the observation function.

    This function handles the inference for the latent degrees of freedom
    parameter `nu`. It deliberately excludes the `pyro.plate` context
    and the `growth_obs` sample site because those are observed data,
    not latent variables.
    """

    # --- 1. Latent Variable (nu) ---
    
    # The prior is Gamma(2.0, 0.1), which has a mean of 20.0.
    # We use a LogNormal guide to ensure positive support.
    # Initialize loc at log(20) ≈ 3.0 to start the optimization near the prior mean.
    nu_loc = pyro.param(f"{name}_nu_loc", jnp.array(3.0))
    nu_scale = pyro.param(f"{name}_nu_scale", jnp.array(0.1),
                          constraint=dist.constraints.positive)

    nu = pyro.sample(f"{name}_nu", dist.LogNormal(nu_loc, nu_scale))

    return