"""
Starting points for optimization: translating between site values and the
parameters of each guide.

Three kinds of dictionary describe a point in parameter space:

- **site values**, keyed by model sample-site name, in constrained space
  (what ``init_to_value`` takes, and what most configured guesses are keyed
  by);
- **AutoDelta parameters**, keyed ``{site}_auto_loc`` (a MAP point, also in
  constrained space);
- **component-guide parameters**, keyed by the guide's ``pyro.param`` names.
  By convention a site ``s`` guided by a Normal or LogNormal has its location
  in ``s_loc``/``s_locs`` and its scale in ``s_scale``/``s_scales``.

``site_values`` pulls site values out of any mixture of the three, and
``component_guide_init`` turns site values into component-guide parameters.
The guide map behind both is built by tracing the guide and checking that each
candidate parameter really is the location (or scale) of its site's
distribution, so a component that does not follow the convention is reported
rather than mis-initialized.
"""

import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers

_WRAPPERS = (dist.ExpandedDistribution, dist.Independent,
             dist.MaskedDistribution)

AUTO_LOC_SUFFIX = "_auto_loc"


def _unwrap(fn):
    """Strip batch-shape wrappers (expand, to_event, mask) off a distribution."""
    while isinstance(fn, _WRAPPERS):
        fn = fn.base_dist
    return fn


def structure_trace(model, seed=0, substitutions=None, substitute_fn=None,
                    **model_kwargs):
    """
    Trace ``model`` once to read its structure: site shapes, supports and
    distributions.

    Unsubstituted latent sites are drawn from the prior, and with a wide
    prior (the thermo ``PddG`` ddG prior with a large learned ``sigma_s``
    gives ln K draws of +/-60) the draw can predict a non-finite ``ln_cfu``.
    numpyro then refuses to construct the observation distribution
    (``CountNegativeBinomial ... invalid log_mu``), although nothing that
    reads a structure trace evaluates the observation. Distribution argument
    validation is therefore switched off for the trace, so a structure
    lookup never depends on the observation model accepting a prior draw.
    Under ``jit`` the checks never run anyway (the values are tracers).

    Parameters
    ----------
    model : callable
        Numpyro model.
    seed : int, optional
        PRNG seed for the trace.
    substitutions : dict or None, optional
        Constrained site values to substitute.
    substitute_fn : callable or None, optional
        ``numpyro.handlers.substitute`` ``substitute_fn`` (site -> value or
        None), applied after ``substitutions``.
    **model_kwargs
        Passed to the model.

    Returns
    -------
    collections.OrderedDict
        The numpyro trace.
    """

    from numpyro import validation_enabled

    fn = handlers.seed(model, seed)
    if substitutions:
        fn = handlers.substitute(fn, data=substitutions)
    if substitute_fn is not None:
        fn = handlers.substitute(fn, substitute_fn=substitute_fn)
    with validation_enabled(False):
        return handlers.trace(fn).get_trace(**model_kwargs)


def unconstrained_substitute_fn(unconstrained):
    """
    ``substitute_fn`` that sets each named latent site from its unconstrained
    value, mapped through ``biject_to`` of the site's own support.

    Every listed site takes its value without a prior draw, and a
    hierarchical site's support is that of its distribution given the
    substituted parents (what ``numpyro.infer.util.constrain_fn`` does).
    """

    from numpyro.distributions.transforms import biject_to

    def fn(site):
        name = site["name"]
        if (site["type"] != "sample" or site.get("is_observed", False)
                or name not in unconstrained):
            return None
        return biject_to(site["fn"].support)(unconstrained[name])

    return fn


def trace_model_sites(model, priors, data, substitutions=None, seed=0,
                      unconstrained=None):
    """
    Trace ``model`` once and return its latent sample sites.

    The trace is a ``structure_trace``: argument validation is off, so a
    prior draw that makes the observation model's parameters non-finite does
    not fail it.

    Parameters
    ----------
    model : callable
        Numpyro model taking ``priors=`` and ``data=``.
    priors, data
        Passed to the model.  ``data`` should be a full (library-ordered)
        batch so per-genotype sites have library shape.
    substitutions : dict or None, optional
        Constrained site values to substitute while tracing.
    seed : int, optional
        PRNG seed for the trace.
    unconstrained : dict or None, optional
        Unconstrained site values (a MAP point) to substitute, each mapped
        through its own site's support (``unconstrained_substitute_fn``).

    Returns
    -------
    dict
        ``{site name: numpyro trace site}`` for unobserved sample sites.
    """

    sub_fn = (None if unconstrained is None
              else unconstrained_substitute_fn(unconstrained))
    tr = structure_trace(model, seed=seed, substitutions=substitutions,
                         substitute_fn=sub_fn, priors=priors, data=data)
    return {name: site for name, site in tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)}


def _matching_param(site_name, suffixes, params, target, shape):
    """Name of the param in ``params`` that equals ``target`` for this site."""
    for suffix in suffixes:
        name = site_name + suffix
        if name not in params:
            continue
        value = jnp.asarray(params[name])
        if value.shape != shape:
            continue
        try:
            same = bool(jnp.allclose(jnp.broadcast_to(value, target.shape),
                                     target))
        except ValueError:
            same = False
        if same:
            return name
    return None


def component_guide_map(guide, priors, data, seed=0):
    """
    Map each guide sample site to its location and scale parameters.

    Parameters
    ----------
    guide : callable
        The component guide (``orchestrator.jax_model_guide``).
    priors, data
        Passed to the guide; ``data`` should be a full batch.
    seed : int, optional
        PRNG seed for the trace.

    Returns
    -------
    mapping : dict
        ``{site: {"kind": "normal"|"lognormal", "loc": param name,
        "scale": param name or None, "shape": site shape,
        "scale_lower_bound": the scale param's constraint lower bound or
        None}}``.
    unmatched : list of str
        Guide sample sites whose location parameter could not be identified
        (not Normal/LogNormal, or not following the naming convention).
    defaults : dict
        ``{param name: initial value}`` for every guide param.
    """

    tr = handlers.trace(handlers.seed(guide, seed)).get_trace(priors=priors,
                                                              data=data)
    params = {name: site["value"] for name, site in tr.items()
              if site["type"] == "param"}
    lower_bounds = {}
    for name, site in tr.items():
        if site["type"] != "param":
            continue
        bound = getattr(site.get("kwargs", {}).get("constraint"),
                        "lower_bound", None)
        if bound is not None:
            lower_bounds[name] = bound

    mapping = {}
    unmatched = []
    for name, site in tr.items():
        if site["type"] != "sample" or site.get("is_observed", False):
            continue
        fn = _unwrap(site["fn"])
        if isinstance(fn, dist.LogNormal):
            kind = "lognormal"
        elif isinstance(fn, dist.Normal):
            kind = "normal"
        else:
            unmatched.append(name)
            continue

        shape = jnp.shape(site["value"])
        loc = _matching_param(name, ("_loc", "_locs"), params,
                              jnp.asarray(fn.loc), shape)
        if loc is None:
            unmatched.append(name)
            continue
        scale = _matching_param(name, ("_scale", "_scales"), params,
                                jnp.asarray(fn.scale), shape)
        mapping[name] = {"kind": kind, "loc": loc, "scale": scale,
                         "shape": shape,
                         "scale_lower_bound": lower_bounds.get(scale)}

    return mapping, sorted(unmatched), params


def site_values(values, model_sites, guide_map=None):
    """
    Collect constrained site values from a mixed-key dictionary.

    Recognized keys, in increasing priority: component-guide location params
    (via ``guide_map``; a LogNormal location is exponentiated), bare site
    names, and AutoDelta ``{site}_auto_loc`` params.  Values that cannot be
    broadcast to the site's shape are skipped.

    Parameters
    ----------
    values : dict
        Guesses, a MAP result, or both.
    model_sites : dict
        ``trace_model_sites`` output.
    guide_map : dict or None, optional
        ``component_guide_map`` mapping.

    Returns
    -------
    dict
        ``{site: array}``.
    """

    out = {}

    def _put(site, value):
        shape = jnp.shape(model_sites[site]["value"])
        try:
            out[site] = jnp.broadcast_to(jnp.asarray(value, dtype=float),
                                         shape)
        except ValueError:
            pass

    if guide_map:
        for site, entry in guide_map.items():
            if site in model_sites and entry["loc"] in values:
                value = jnp.asarray(values[entry["loc"]], dtype=float)
                if entry["kind"] == "lognormal":
                    value = jnp.exp(value)
                _put(site, value)

    for key, value in values.items():
        if key in model_sites:
            _put(key, value)

    for key, value in values.items():
        if key.endswith(AUTO_LOC_SUFFIX):
            site = key[:-len(AUTO_LOC_SUFFIX)]
            if site in model_sites:
                _put(site, value)

    return out


def component_guide_init(values, guide_map, defaults, init_scale=None):
    """
    Component-guide parameters that start at the given site values.

    Parameters
    ----------
    values : dict
        Constrained site values (``site_values`` output).
    guide_map : dict
        ``component_guide_map`` mapping.
    defaults : dict
        Starting param values to fall back on (the guide's own initial values,
        overlaid with any param-keyed guesses).
    init_scale : float or None, optional
        Upper bound for every mapped scale param: each starts at
        ``min(default, init_scale)``.  None leaves scales at their defaults.
        A scale whose constraint has a lower bound (``greater_than(1e-4)``
        on the hyperparameter scales) starts at least ``init_scale`` above
        that bound: a param starting on its bound is ``-inf`` in
        unconstrained space and never moves.

    Returns
    -------
    params : dict
        ``{param name: value}`` for mapped locations and scales.
    skipped : list of str
        Sites with a value that could not be used (a non-positive value for a
        LogNormal site).
    """

    params = {}
    skipped = []
    for site, entry in guide_map.items():
        if site in values:
            value = jnp.asarray(values[site], dtype=float)
            if entry["kind"] == "lognormal":
                if not bool(jnp.all(value > 0)):
                    skipped.append(site)
                    value = None
                else:
                    value = jnp.log(value)
            if value is not None:
                params[entry["loc"]] = jnp.broadcast_to(value, entry["shape"])

        scale = entry["scale"]
        if init_scale is not None and scale is not None and scale in defaults:
            start = jnp.minimum(jnp.asarray(defaults[scale], dtype=float),
                                init_scale)
            bound = entry.get("scale_lower_bound")
            if bound is not None:
                start = jnp.maximum(start, jnp.asarray(bound) + init_scale)
            params[scale] = start

    return params, sorted(skipped)


def site_prior_sds(model_sites):
    """
    Prior standard deviation of each site, broadcast to the site's shape.

    Parameters
    ----------
    model_sites : dict
        ``trace_model_sites`` output (ideally traced with the current site
        values substituted, so hierarchical priors reflect them).

    Returns
    -------
    dict
        ``{site: array}`` for sites whose prior has a finite, positive
        variance.  Other sites are left out.
    """

    out = {}
    for name, site in model_sites.items():
        try:
            sd = jnp.sqrt(jnp.asarray(site["fn"].variance, dtype=float))
            sd = jnp.broadcast_to(sd, jnp.shape(site["value"]))
        except (NotImplementedError, ValueError, TypeError, AttributeError,
                ArithmeticError):
            # ArithmeticError: e.g. an InverseGamma with Python-float
            # concentration 2 divides by zero in its variance.
            continue
        sd = np.asarray(sd)
        if sd.size == 0 or not np.all(np.isfinite(sd)) or not np.all(sd > 0):
            continue
        out[name] = sd
    return out


def site_unconstrained_prior_sds(model_sites, seed=0, num_draws=64):
    """
    Prior spread of each site in its unconstrained coordinates.

    Guide locations live in unconstrained space (a LogNormal guide's location
    is ``log`` of a positive site, an AutoNormal location is the site mapped
    through ``biject_to(support).inv``), so this is the prior width in the
    same units.  A real-valued site with a finite prior variance uses the
    exact SD; every other site uses a robust Monte Carlo estimate, ``IQR /
    1.349`` of ``num_draws`` prior draws mapped to unconstrained space (the SD
    for a normal, finite for heavy-tailed priors such as a half-Cauchy).

    Parameters
    ----------
    model_sites : dict
        ``trace_model_sites`` output (ideally traced with the current site
        values substituted, so hierarchical priors reflect them).
    seed : int, optional
        PRNG seed for the Monte Carlo draws.
    num_draws : int, optional
        Draws per site for the Monte Carlo estimate (default 64).

    Returns
    -------
    dict
        ``{site: array}`` broadcast to the site's shape, for sites whose
        spread is finite and positive everywhere.  Other sites are left out.
    """

    import jax
    from numpyro.distributions.transforms import biject_to

    exact = site_prior_sds(model_sites)
    key = jax.random.PRNGKey(seed)

    out = {}
    for name, site in model_sites.items():
        fn = site["fn"]
        shape = jnp.shape(site["value"])
        try:
            real = _unwrap(fn).support is dist.constraints.real
        except (NotImplementedError, AttributeError):
            real = False
        if real and name in exact:
            out[name] = exact[name]
            continue
        try:
            key, sub = jax.random.split(key)
            draws = fn.sample(sub, (num_draws,))
            u = biject_to(fn.support).inv(draws)
            q25, q75 = jnp.quantile(u, jnp.array([0.25, 0.75]), axis=0)
            sd = np.asarray(jnp.broadcast_to((q75 - q25) / 1.349, shape))
        except (NotImplementedError, ValueError, TypeError, AttributeError,
                ArithmeticError):
            continue
        if sd.size == 0 or not np.all(np.isfinite(sd)) or not np.all(sd > 0):
            continue
        out[name] = sd
    return out
