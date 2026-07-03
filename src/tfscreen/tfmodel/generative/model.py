
# Import for typing
from tfscreen.tfmodel.data_class import (
    DataClass,
    PriorsClass,
)

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

def jax_model(data: DataClass,
              priors: PriorsClass,
              **control):
    """
    Defines the joint hierarchical model for bacterial growth and binding.

    Parameters
    ----------
    data : DataClass
        A DataClass pytree containing all observed data and experimental
        metadata for both growth and binding assays.
    priors : PriorsClass
        A PriorsClass pytree containing the prior distributions
        (as Numpyro distribution objects) for all latent parameters.
    control : dict
        dictionary of keyword arguments necessary to specify the model. Expects:
        - theta
        - condition_growth
        - ln_cfu0
        - activity
        - transformation
        - dk_geno
        - growth_transition
        - growth_noise
        - sample_offset
        - calculate_growth
        - theta_binding_noise
        - theta_growth_noise
        - binding_observer
        - growth_observer
        - is_guide
        The dictionary can also optionally have `batch_idx` (which overrides
        whatever is in `batch_size`) or `batch_size`. 
    """
    
    # -------------------------------------------------------------------------
    # Parse shared control inputs

    binding_only = control.get("binding_only", False)
    is_guide = control["is_guide"]
    theta_model, calc_theta, get_moments = control["theta"]
    theta_binding_noise_model = control["theta_binding_noise"]
    binding_observer = control["observe_binding"]

    # -------------------------------------------------------------------------
    # Binding-only mode: theta is inferred directly from observed theta values.
    # No growth model components are sampled.

    if binding_only:
        theta = theta_model("theta", data.binding, priors.theta)
        theta_binding = calc_theta(theta, data.binding)
        pyro.deterministic("theta_binding_pred", theta_binding)
        binding_pred = theta_binding_noise_model("theta_binding_noise",
                                                 theta_binding,
                                                 priors.binding.theta_binding_noise)
        if is_guide:
            binding_observer("final_growth_obs", data.binding, None)
        else:
            pyro.deterministic("binding_pred", binding_pred)
            binding_observer("final_growth_obs", data.binding, binding_pred)
        return

    # -------------------------------------------------------------------------
    # Full joint model: unpack remaining growth-model control entries.

    condition_growth_model = control["condition_growth"]
    ln_cfu0_model = control["ln_cfu0"]
    activity_model = control["activity"]
    dk_geno_model = control["dk_geno"]
    transformation_model, transformation_update, transformation_needs_population = \
        control["transformation"]
    theta_growth_noise_model = control["theta_growth_noise"]
    theta_rescale = control["theta_rescale"]
    growth_transition_model = control["growth_transition"]
    growth_noise_model = control["growth_noise"]
    sample_offset_model = control["sample_offset"]
    calculate_growth = control["calculate_growth"]
    growth_observer = control["observe_growth"]

    # -------------------------------------------------------------------------
    # Calculate theta

    # Calculate shared theta
    theta = theta_model("theta",
                        data.growth,
                        priors.theta)

    # Get population moments as anchors for the transformation model
    anchors = get_moments(theta, data.growth)

    # -------------------------------------------------------------------------
    # Make prediction for the binding experiment

    theta_binding = calc_theta(theta,data.binding)
    pyro.deterministic(f"theta_binding_pred",theta_binding)
    binding_pred = theta_binding_noise_model("theta_binding_noise",
                                             theta_binding,
                                             priors.binding.theta_binding_noise)

    # -------------------------------------------------------------------------
    # Make prediction for the growth experiment

    # Get growth parameters
    growth_params = condition_growth_model("condition_growth",
                                           data.growth,
                                           priors.growth.condition_growth)

    # initial populations
    ln_cfu0 = ln_cfu0_model("ln_cfu0",
                            data.growth,
                            priors.growth.ln_cfu0)

    # # pleiotropic effect of mutation
    dk_geno = dk_geno_model("dk_geno",
                            data.growth,
                            priors.growth.dk_geno)

    # activity
    activity = activity_model("activity",
                              data.growth,
                              priors.growth.activity)

    # theta
    theta_growth = calc_theta(theta,data.growth)
    pyro.deterministic(f"theta_growth_pred",theta_growth)

    # Transformation parameters (lam, mu, sigma)
    trans_params = transformation_model("transformation",
                                        data.growth,
                                        priors.growth.transformation,
                                        anchors=anchors)

    # Correct theta for transformation
    # theta_growth shape: (..., titrant_name, titrant_conc, geno) or scattered
    # Result broadcasts to interaction of (rep, pre) and (titrant)
    # Parameters passed as tuple
    if transformation_needs_population:
        # The congression correction's background CDF must be estimated from
        # the full genotype population, not whatever subset of genotypes is
        # active in this particular forward pass (a training minibatch, or a
        # handful of genotypes requested at prediction time) — see
        # transformation/_congression.py::update_thetas.  When the caller
        # hasn't supplied one explicitly (data.growth.external_theta_population),
        # compute it locally by re-running calc_theta over every genotype.
        # This is only correct when data.growth already spans the full
        # population, which holds during SVI training (genotype minibatching
        # never shrinks data.growth.num_genotype — see tensors/batch.py) but
        # NOT for prediction code paths that subset genotypes; those must
        # supply external_theta_population themselves.
        if data.growth.external_theta_population is not None:
            population_theta_growth = data.growth.external_theta_population
        else:
            # Deliberately leave scatter_theta untouched (rather than forcing
            # it to 0) so population_theta_growth's leading (non-genotype)
            # dimensions match theta_growth's exactly -- update_thetas relies
            # on that alignment when broadcasting the correction back onto
            # theta_growth's shape.
            num_genotype = data.growth.num_genotype
            full_population_idx = jnp.arange(num_genotype)
            full_population_data = data.growth.replace(
                batch_idx=full_population_idx,
                geno_theta_idx=full_population_idx,
            )
            population_theta_growth = calc_theta(theta, full_population_data)

        corr_theta_growth = transformation_update(
            theta_growth,
            params=trans_params,
            mask=data.growth.congression_mask,
            population_theta=population_theta_growth,
        )
    else:
        corr_theta_growth = transformation_update(theta_growth,
                                                  params=trans_params,
                                                  mask=data.growth.congression_mask)

    noisy_theta_growth = theta_growth_noise_model("theta_growth_noise",
                                                  corr_theta_growth,
                                                  priors.growth.theta_growth_noise)

    rescaled_theta = theta_rescale(noisy_theta_growth)

    # -------------------------------------------------------------------------
    # finalize

    # If this is a guide, just make the final observations but do not calculate
    # final tensors. We still need to call growth_transition_model so its latent
    # variables (e.g. memory k1/tau0/k2) get guide sample sites registered.
    if is_guide:

        growth_transition_model("growth_transition",
                                data.growth,
                                priors.growth.growth_transition,
                                g_pre=jnp.zeros_like(rescaled_theta),
                                g_sel=jnp.zeros_like(rescaled_theta),
                                t_pre=data.growth.t_pre,
                                t_sel=data.growth.t_sel,
                                theta=rescaled_theta)

        growth_noise_model("growth_noise",
                           data.growth,
                           priors.growth.growth_noise)

        sample_offset_model("sample_offset",
                            data.growth,
                            priors.growth.sample_offset)

        growth_observer("final_binding_obs", data.growth, None)
        binding_observer("final_growth_obs", data.binding, None)

    # real calculation
    else:

        # Pre-split (t = -t_pre) observations — direct constraint on ln_cfu0.
        # ln_cfu0 shape: (num_rep, 1, num_cp, 1, 1, 1, batch_size)
        # Squeeze broadcast dims to get (num_rep, num_cp, batch_size).
        if getattr(data, "presplit", None) is not None:
            ln_cfu0_3d = ln_cfu0[:, 0, :, 0, 0, 0, :]
            ps = data.presplit
            bi = data.growth.batch_idx
            obs_t0  = ps.ln_cfu_t0[:, :, bi]
            std_t0  = ps.ln_cfu_t0_std[:, :, bi]
            mask_t0 = ps.good_mask[:, :, bi]
            with pyro.plate("presplit_replicate",
                            size=ps.num_replicate, dim=-3):
                with pyro.plate("presplit_condition_pre",
                                size=ps.num_condition_pre, dim=-2):
                    with pyro.plate("shared_genotype_plate",
                                    size=data.growth.batch_size, dim=-1):
                        with pyro.handlers.scale(
                                scale=data.growth.scale_vector):
                            with pyro.handlers.mask(mask=mask_t0):
                                pyro.sample(
                                    "presplit_obs",
                                    dist.Normal(ln_cfu0_3d, std_t0),
                                    obs=obs_t0,
                                )

        # calculate observable (all tensors have correct dimensions)
        g_pre, g_sel = calculate_growth(params=growth_params,
                                        dk_geno=dk_geno,
                                        activity=activity,
                                        theta=rescaled_theta)

        total_growth = growth_transition_model("growth_transition",
                                               data.growth,
                                               priors.growth.growth_transition,
                                               g_pre=g_pre,
                                               g_sel=g_sel,
                                               t_pre=data.growth.t_pre,
                                               t_sel=data.growth.t_sel,
                                               theta=rescaled_theta)

        sigma_k = growth_noise_model("growth_noise",
                                     data.growth,
                                     priors.growth.growth_noise)

        delta_sample = sample_offset_model("sample_offset",
                                           data.growth,
                                           priors.growth.sample_offset)

        ln_cfu_pred = ln_cfu0 + total_growth + delta_sample

        # Register results
        pyro.deterministic(f"binding_pred", binding_pred)
        pyro.deterministic(f"growth_pred", ln_cfu_pred)

        # Calculate likelihood
        growth_observer("final_binding_obs", data.growth, ln_cfu_pred, sigma_k=sigma_k)
        binding_observer("final_growth_obs", data.binding, binding_pred)


