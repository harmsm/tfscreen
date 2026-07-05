import numpyro as pyro
import numpyro.distributions as dist

from jax import numpy as jnp

from tfscreen.tfmodel.data_class import PreSplitData, GrowthData


def observe(name: str,
            data: PreSplitData,
            ln_cfu0: jnp.ndarray,
            *,
            growth: GrowthData):
    """
    Defines the observation site for the pre-split (t = -t_pre) data.

    These observations come from a single pooled aliquot taken just before
    the culture is split into titrant-concentration conditions, so the
    prediction for each observation is simply the initial-population
    parameter ``ln_cfu0[replicate, condition_pre, genotype]``.  This makes
    the site a direct side-channel constraint on ``ln_cfu0``.

    The pre-split data has no latent parameters of its own -- it reuses the
    ``ln_cfu0`` latent sampled in the main growth model, and it borrows the
    genotype batch state (``batch_idx``/``batch_size``/``scale_vector``) from
    the companion GrowthData so that its genotype axis stays aligned with the
    growth mini-batch.  The innermost plate is therefore the *shared*
    ``"shared_genotype_plate"`` (not a prefixed one), matching the growth
    observer and the activity/dk_geno components so Numpyro shares the
    genotype subsample across all sites.

    A mask (``data.good_mask``) excludes bad-quality or missing observations
    from the log-likelihood.

    Parameters
    ----------
    name : str
        Prefix for the pre-split plate and sample sites.
    data : PreSplitData
        Pre-split observations, shaped
        ``(num_replicate, num_condition_pre, num_genotype)``.
    ln_cfu0 : jnp.ndarray
        The initial-population latent from the growth model, shaped
        ``(num_rep, 1, num_cp, 1, 1, 1, batch_size)``.  Squeezed here to
        ``(num_rep, num_cp, batch_size)`` before being used as the prediction.
    growth : GrowthData
        The companion growth data, used only for its genotype batch state
        (``batch_idx``, ``batch_size``, ``scale_vector``).
    """

    # ln_cfu0 shape: (num_rep, 1, num_cp, 1, 1, 1, batch_size)
    # Squeeze broadcast dims to get (num_rep, num_cp, batch_size).
    ln_cfu0_3d = ln_cfu0[:, 0, :, 0, 0, 0, :]

    bi = growth.batch_idx
    obs_t0 = data.ln_cfu_t0[:, :, bi]
    std_t0 = data.ln_cfu_t0_std[:, :, bi]
    mask_t0 = data.good_mask[:, :, bi]

    with pyro.plate(f"{name}_replicate", size=data.num_replicate, dim=-3):
        with pyro.plate(f"{name}_condition_pre", size=data.num_condition_pre, dim=-2):
            with pyro.plate("shared_genotype_plate",
                            size=growth.batch_size, dim=-1):

                # Scale data for sub-sampling
                with pyro.handlers.scale(scale=growth.scale_vector):

                    # Apply mask for good observations
                    with pyro.handlers.mask(mask=mask_t0):

                        # Define the observation site
                        pyro.sample(f"{name}_obs",
                                    dist.Normal(ln_cfu0_3d, std_t0),
                                    obs=obs_t0)


def guide(name: str,
          data: PreSplitData,
          ln_cfu0: jnp.ndarray,
          *,
          growth: GrowthData):
    """
    Guide corresponding to the observation function.

    This function does nothing, as the pre-split observations introduce no
    latent variables of their own (they constrain the ``ln_cfu0`` latent
    that is already sampled and guided by the growth model).
    """

    return
