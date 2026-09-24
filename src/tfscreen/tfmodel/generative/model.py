
# Import for typing
from tfscreen.tfmodel.data_class import (
    DataClass,
    PriorsClass,
)

import jax
import jax.numpy as jnp
import numpyro as pyro

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
        - observe_binding
        - observe_growth
        - is_guide
        The dictionary can also optionally have `observe_presplit` and/or
        `observe_base_growth` (the side-channel observers, present only when
        their data was supplied), plus `batch_idx` (which overrides whatever
        is in `batch_size`) or `batch_size`.
    """
    
    # -------------------------------------------------------------------------
    # Parse shared control inputs

    binding_only = control.get("binding_only", False)
    is_guide = control["is_guide"]
    theta_model, calc_theta, _ = control["theta"]
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
                                                 priors.binding.theta_binding_noise,
                                                 data=data.binding)
        if is_guide:
            binding_observer("binding", data.binding, None)
        else:
            pyro.deterministic("binding_pred", binding_pred)
            binding_observer("binding", data.binding, binding_pred)
        return

    # -------------------------------------------------------------------------
    # Full joint model: unpack remaining growth-model control entries.

    condition_growth_model = control["condition_growth"]
    ln_cfu0_model = control["ln_cfu0"]
    activity_model = control["activity"]
    dk_geno_model = control["dk_geno"]
    transformation_model, cell_classes, transformation_needs_population = \
        control["transformation"]
    theta_growth_noise_model = control["theta_growth_noise"]
    theta_rescale = control["theta_rescale"]
    growth_transition_model = control["growth_transition"]
    growth_noise_model = control["growth_noise"]
    sample_offset_model = control["sample_offset"]
    calculate_growth = control["calculate_growth"]
    growth_observer = control["observe_growth"]

    # Optional side-channel observers. Present in `control` only when their
    # data was supplied (see model_orchestrator._initialize_classes). Each is
    # the .observe function in the main model and the .guide function in the
    # guide, exactly like binding_observer/growth_observer.
    presplit_observer = control.get("observe_presplit")
    base_growth_observer = control.get("observe_base_growth")

    # -------------------------------------------------------------------------
    # Calculate theta

    # Calculate shared theta
    theta = theta_model("theta",
                        data.growth,
                        priors.theta)

    # -------------------------------------------------------------------------
    # Make prediction for the binding experiment

    theta_binding = calc_theta(theta,data.binding)
    pyro.deterministic(f"theta_binding_pred",theta_binding)
    binding_pred = theta_binding_noise_model("theta_binding_noise",
                                             theta_binding,
                                             priors.binding.theta_binding_noise,
                                             data=data.binding)

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

    # pleiotropic effect of mutation and TF activity. When the congression
    # mixture needs them, the components also return every genotype's value
    # in library order (None if their latents arrived batch-sized).
    if transformation_needs_population:
        dk_geno, dk_population = dk_geno_model("dk_geno",
                                               data.growth,
                                               priors.growth.dk_geno,
                                               return_population=True)
        activity, activity_population = activity_model("activity",
                                                       data.growth,
                                                       priors.growth.activity,
                                                       return_population=True)
    else:
        dk_geno = dk_geno_model("dk_geno",
                                data.growth,
                                priors.growth.dk_geno)
        activity = activity_model("activity",
                                  data.growth,
                                  priors.growth.activity)

    # theta (the genotype's own, before congression)
    theta_growth = calc_theta(theta,data.growth)
    pyro.deterministic(f"theta_growth_pred",theta_growth)

    # Transformation parameters (lambda for the mixture; none for single)
    trans_params = transformation_model("transformation",
                                        data.growth,
                                        priors.growth.transformation)

    # Noise acts on the genotype's own theta, before the cell classes are
    # built; co-residents use noiseless population values.
    noisy_theta_growth = theta_growth_noise_model("theta_growth_noise",
                                                  theta_growth,
                                                  priors.growth.theta_growth_noise,
                                                  data=data.growth)

    # -------------------------------------------------------------------------
    # finalize

    # If this is a guide, just make the final observations but do not calculate
    # final tensors. We still need to call growth_transition_model so its latent
    # variables (e.g. memory k1/tau0/k2) get guide sample sites registered.
    if is_guide:

        rescaled_theta = theta_rescale(noisy_theta_growth)

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

        growth_observer("growth", data.growth, None,
                        priors=priors.growth.growth_obs)
        binding_observer("binding", data.binding, None)

        # Register side-channel guide sites. presplit.guide is a no-op (it
        # introduces no latents); base_growth.guide registers the k_ref
        # variational site so the guide matches the model.
        if presplit_observer is not None:
            presplit_observer("presplit", data.presplit, ln_cfu0,
                              growth=data.growth)
        if base_growth_observer is not None:
            base_growth_observer("base_growth", data.base_growth, dk_geno,
                                 growth=data.growth,
                                 priors=priors.growth.base_growth)

    # real calculation
    else:

        # Pre-split (t = -t_pre) observations — direct constraint on ln_cfu0.
        if presplit_observer is not None:
            presplit_observer("presplit", data.presplit, ln_cfu0,
                              growth=data.growth)

        # Direct growth-rate measurements — anchors k_ref (and, via dk_geno's
        # wt=0 pin, the shared k/m identifiability slack) to genotypes with
        # a directly-measured reference-condition growth rate.
        if base_growth_observer is not None:
            base_growth_observer("base_growth", data.base_growth, dk_geno,
                                 growth=data.growth,
                                 priors=priors.growth.base_growth)

        # Congression: split each genotype's cells into classes (clean, and
        # congressed cells carrying co-resident plasmids), each with its own
        # cell-level theta, activity and dk_geno and a mixture weight. See
        # transformation/mixture.py.
        population = None
        if transformation_needs_population:
            population = _population(data.growth, theta, calc_theta,
                                     dk_population, activity_population)
        classes = cell_classes((noisy_theta_growth, activity, dk_geno),
                               population, trans_params, data.growth)

        # Grow every class. The class axis leads the growth layout and every
        # growth component is elementwise, so each is called once.
        rescaled_theta = theta_rescale(classes.theta)
        g_pre, g_sel = calculate_growth(params=growth_params,
                                        dk_geno=classes.dk_geno,
                                        activity=classes.activity,
                                        theta=rescaled_theta)

        class_growth = growth_transition_model("growth_transition",
                                               data.growth,
                                               priors.growth.growth_transition,
                                               g_pre=g_pre,
                                               g_sel=g_sel,
                                               t_pre=data.growth.t_pre,
                                               t_sel=data.growth.t_sel,
                                               theta=rescaled_theta)

        # Mix the classes at the observable level: exp(ln_cfu) adds, rates
        # do not.
        total_growth = jax.scipy.special.logsumexp(
            classes.log_weight + class_growth, axis=0)

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
        growth_observer("growth", data.growth, ln_cfu_pred, sigma_k=sigma_k,
                        priors=priors.growth.growth_obs)
        binding_observer("binding", data.binding, binding_pred)


def _population(growth, theta, calc_theta, dk_population, activity_population):
    """
    Library-ordered theta, activity and dk_geno for every genotype.

    Each comes from ``growth.external_*_population`` when supplied (prediction
    code that runs a genotype subset, or whose latents are batch-sized
    substitutions), else is computed here: theta by evaluating the theta
    component over the full library (valid because genotype mini-batching
    never shrinks ``num_genotype``; see tensors/batch.py), activity and
    dk_geno from their components' ``return_population`` values.

    Raises
    ------
    ValueError
        If activity or dk_geno is needed but neither the component nor the
        data supplies it.
    """
    if growth.external_theta_population is not None:
        theta_population = growth.external_theta_population
    else:
        full = jnp.arange(growth.num_genotype)
        theta_population = calc_theta(
            theta, growth.replace(batch_idx=full, geno_theta_idx=full))

    if growth.external_dk_population is not None:
        dk_population = growth.external_dk_population
    if growth.external_activity_population is not None:
        activity_population = growth.external_activity_population

    missing = [name for name, value in (("dk_geno", dk_population),
                                        ("activity", activity_population))
               if value is None]
    if missing:
        raise ValueError(
            f"The congression mixture needs every genotype's {missing}, but "
            f"the latents were not library-sized and no "
            f"external_*_population was supplied. Prediction code that runs "
            f"a genotype subset must pass them (see analysis/prediction.py)."
        )

    return theta_population, activity_population, dk_population
