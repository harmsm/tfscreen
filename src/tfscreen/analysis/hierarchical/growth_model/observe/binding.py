import torch
import pyro
import pyro.distributions as dist

from tfscreen.analysis.hierarchical.growth_model.data_class import BindingData

def observe(name: str,
            data: BindingData,
            binding_pred: torch.Tensor):
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
        The prefix for all Pyro sample/plate sites.
    data : BindingData
        A frozen dataclass containing the observed binding data
        and metadata. This function primarily uses:
        - ``data.num_titrant_name`` : (int)
        - ``data.num_titrant_conc`` : (int)
        - ``data.num_genotype`` : (int)
        - ``data.good_mask`` : (torch.Tensor) Boolean mask for valid data.
        - ``data.theta_std`` : (torch.Tensor) Observed std dev of theta.
        - ``data.theta_obs`` : (torch.Tensor) Observed mean of theta.
    binding_pred : torch.Tensor
        The deterministically predicted mean theta values from the model,
        with a shape matching `data.theta_obs`.
    """

    # Binding observation
    with pyro.plate(f"{name}_binding_titrant_name", size=data.num_titrant_name,dim=-3):
        with pyro.plate(f"{name}_binding_titrant_conc", size=data.num_titrant_conc,dim=-2):
            with pyro.plate(f"{name}_binding_genotype_plate",size=data.batch_size,dim=-1):
                
                # Scale data for sub-sampling
                with pyro.poutine.scale(scale=data.scale_vector):

                    # Apply mask for good observations
                    with pyro.poutine.mask(mask=data.good_mask):
                        
                        # Define the observation site
                        pyro.sample(f"{name}_binding_obs",
                                    dist.Normal(binding_pred, data.theta_std),
                                    obs=data.theta_obs)
                        
def guide(name: str,
          data: BindingData,
          binding_pred: torch.Tensor):
    """
    Guide corresponding to the observation function.

    This function does nothing, as there are no latent variables to infer
    in the binding model (only observed data).
    """

    return