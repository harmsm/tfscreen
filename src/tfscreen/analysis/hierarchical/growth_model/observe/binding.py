import numpyro as pyro
import numpyro.distributions as dist
from numpyro.handlers import mask
from jax import numpy as jnp

# Assuming data_class is in a relative path
from tfscreen.analysis.hierarchical.growth_model.data_class import BindingData

def observe(name: str, data: BindingData, binding_pred: jnp.ndarray):
    """
    Defines the observation site for the binding data.

    This function links the predicted binding values (`binding_pred`) to the
    observed data (`data.theta_obs`) using a Normal likelihood.
    It operates over a 3D tensor defined by titrant names, titrant
    concentrations, and genotypes.

    A mask (`data.good_mask`) is applied to exclude bad-quality or
    missing observations from the log-likelihood calculation.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/plate sites.
    data : BindingData
        A Pytree (Flax dataclass) containing the observed binding data
        and metadata. This function primarily uses:
        - ``data.num_titrant_name`` : (int)
        - ``data.num_titrant_conc`` : (int)
        - ``data.num_genotype`` : (int)
        - ``data.good_mask`` : (jnp.ndarray) Boolean mask for valid data.
        - ``data.theta_std`` : (jnp.ndarray) Observed std dev of theta.
        - ``data.theta_obs`` : (jnp.ndarray) Observed mean of theta.
    binding_pred : jnp.ndarray
        The deterministically predicted mean theta values from the model,
        with a shape matching `data.theta_obs`.
    """

    # Binding observation
    with pyro.plate(f"{name}_binding_titrant_name", size=data.num_titrant_name, dim=-3):
        with pyro.plate(f"{name}_binding_titrant_conc", size=data.num_titrant_conc, dim=-2):
            with pyro.plate(f"{name}_binding_genotype", size=data.num_genotype, dim=-1):
                
                # Apply mask for good observations
                with mask(mask=data.good_mask):
                    
                    # Define the observation site
                    pyro.sample(f"{name}_binding_obs",
                                dist.Normal(binding_pred, data.theta_std),
                                obs=data.theta_obs)