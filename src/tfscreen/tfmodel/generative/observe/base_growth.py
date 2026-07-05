import numpyro as pyro
import numpyro.distributions as dist

from jax import numpy as jnp

from tfscreen.tfmodel.data_class import (
    BaseGrowthData,
    GrowthData,
    BaseGrowthPriors,
)


def observe(name: str,
            data: BaseGrowthData,
            dk_geno: jnp.ndarray,
            *,
            growth: GrowthData,
            priors: BaseGrowthPriors):
    """
    Defines the observation site for the base (reference-condition)
    growth-rate data.

    Unlike growth/binding, this measurement is taken independent of the
    titrant/selection system -- it is a direct read of the reference-condition
    growth rate for a subset of genotypes.  It anchors a shared global scalar
    ``k_ref`` and ties it, via the per-genotype ``dk_geno`` latent (with
    ``dk_geno_wt == 0`` pinned), to genotypes with a directly-measured growth
    rate.  This resolves the k/m identifiability confound described in
    ``BaseGrowthData`` and ``model_orchestrator._read_base_growth_df``.

    This observer *owns* the ``k_ref`` latent -- it is sampled here (from its
    prior) and used only in the ``{name}_obs`` likelihood, exactly the way the
    growth observer owns its ``nu`` degrees-of-freedom latent.  The companion
    ``guide`` provides the matching variational site.

    The genotype batch state (``batch_idx``/``batch_size``/``scale_vector``)
    is borrowed from the companion GrowthData so the genotype axis stays
    aligned with the growth mini-batch, and the innermost plate is the shared
    ``"shared_genotype_plate"``.  A mask (``data.good_mask``) excludes
    genotypes with no measurement from the likelihood.

    Parameters
    ----------
    name : str
        Prefix for the ``k_ref`` and observation sample sites.
    data : BaseGrowthData
        Base growth-rate observations, shaped ``(num_genotype,)``.
    dk_geno : jnp.ndarray
        The per-genotype pleiotropic growth-effect latent from the growth
        model, shaped ``(1, 1, 1, 1, 1, 1, batch_size)``.  Flattened here to
        ``(batch_size,)``.
    growth : GrowthData
        The companion growth data, used only for its genotype batch state
        (``batch_idx``, ``batch_size``, ``scale_vector``).
    priors : BaseGrowthPriors
        Prior (loc, scale) for the ``k_ref`` scalar.
    """

    k_ref = pyro.sample(f"{name}_k_ref",
                        dist.Normal(priors.k_ref_loc, priors.k_ref_scale))

    bi = growth.batch_idx
    rate_obs = data.rate_obs[bi]
    rate_std = data.rate_std[bi]
    mask = data.good_mask[bi]

    # dk_geno shape: (1,1,1,1,1,1,batch_size) -> flatten to (batch_size,).
    dk_geno_flat = dk_geno[0, 0, 0, 0, 0, 0, :]

    with pyro.plate("shared_genotype_plate",
                    size=growth.batch_size, dim=-1):

        # Scale data for sub-sampling
        with pyro.handlers.scale(scale=growth.scale_vector):

            # Apply mask for good observations
            with pyro.handlers.mask(mask=mask):

                # Define the observation site
                pyro.sample(f"{name}_obs",
                            dist.Normal(k_ref + dk_geno_flat, rate_std),
                            obs=rate_obs)


def guide(name: str,
          data: BaseGrowthData,
          dk_geno: jnp.ndarray,
          *,
          growth: GrowthData,
          priors: BaseGrowthPriors):
    """
    Guide corresponding to the observation function.

    Registers the ``k_ref`` variational site.  The location and scale are
    ``pyro.param``s initialized from the prior; the scale is constrained
    ``> 1e-4`` to avoid variational scale collapse (see
    ``feedback_svi_nan_scale_collapse``).  This mirrors the growth observer's
    treatment of its ``nu`` latent.

    ``dk_geno`` and ``growth`` are accepted for signature parallelism with
    ``observe`` but are not used here (no likelihood is evaluated in the guide).
    """

    k_ref_loc = pyro.param(f"{name}_k_ref_loc",
                           jnp.array(priors.k_ref_loc))
    k_ref_scale = pyro.param(f"{name}_k_ref_scale",
                             jnp.array(priors.k_ref_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    pyro.sample(f"{name}_k_ref", dist.Normal(k_ref_loc, k_ref_scale))

    return


def get_hyperparameters():
    """
    Default hyperparameter for the k_ref prior's scale -- weakly-informative,
    centred on the empirical guess (see derive_k_ref_guess), wide enough that
    a handful of base_growth_df measurements dominate it rather than the
    reverse.
    """
    return {"k_ref_scale": 0.02}


def derive_k_ref_guess(base_growth_df):
    """
    Empirically derive k_ref's initial guess (and prior location) from wt's
    measured rate in base_growth_df.

    dk_geno is fixed to exactly 0 for wt by every dk_geno component, so wt's
    own measurement is a direct, uncontaminated read of k_ref
    (rate_obs_wt ~ Normal(k_ref + 0, rate_std_wt)).

    Parameters
    ----------
    base_growth_df : pd.DataFrame
        Output of model_orchestrator._read_base_growth_df; must contain a
        'wt' row (enforced there).

    Returns
    -------
    float
        wt's measured rate.
    """
    wt_row = base_growth_df[base_growth_df["genotype"].astype(str) == "wt"]
    return float(wt_row["rate"].iloc[0])


def get_priors(k_ref_loc):
    """
    Build the BaseGrowthPriors dataclass from an empirically-derived
    ``k_ref_loc`` (see derive_k_ref_guess) plus the default scale from
    get_hyperparameters().
    """
    return BaseGrowthPriors(k_ref_loc=k_ref_loc, **get_hyperparameters())


def get_guesses(name, k_ref_loc):
    """
    Initial optimizer guess for the k_ref latent -- the same value used as
    the prior's loc (see get_priors/derive_k_ref_guess).
    """
    return {f"{name}_k_ref": jnp.array(k_ref_loc)}
