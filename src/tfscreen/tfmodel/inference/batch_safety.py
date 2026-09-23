"""
Detect latent sites whose shape depends on the genotype mini-batch.

Genotype mini-batching (``tensors/batch.py::get_batch``) slices the growth
tensors down to ``batch_size`` genotypes but leaves ``num_genotype`` at the
full library size.  A per-genotype latent is only safe under mini-batching if
it is sampled at the full library size (a ``num_genotype``-sized plate) and
then sliced to the batch with ``batch_idx`` -- the ``hill_geno`` pattern.  A
latent sampled in a plate sized ``data.batch_size`` instead has one entry per
batch *position*, not per genotype.  Component guides can hide this by
holding full-size variational params and slicing them, but any numpyro
autoguide (``AutoDelta`` for MAP included) sizes its params from a traced
batch, so every step writes whichever genotypes are in that batch onto the
same positions.  The result is silently wrong for every non-binding genotype.

:func:`find_batch_dependent_latents` finds such sites by tracing the model at
two different batch sizes: a correctly written latent has the same shape in
both traces, while a batch-sized one changes shape.

A library-sized latent can still be read in the wrong *order*: the full-batch
training index is binding-first and reshuffled every step (see
``ModelOrchestrator.get_random_idx``), so a component that returns
library-ordered values without slicing them by ``batch_idx`` (for example by
slicing only when ``batch_size < num_genotype``) pairs each batch position
with the wrong genotype. Shapes are right, so the shape check cannot see it.
:func:`find_batch_order_mismatches` catches it by tracing the model under two
orderings of the same genotypes, with the latents held fixed, and checking
that the batch-positional predictions permute with the index.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, substitute, trace


def _latent_shapes(model_fn, priors, data):
    """
    Return {site_name: shape} for every unobserved sample site.

    Traces under ``jax.eval_shape`` (abstract evaluation: no FLOPs, no
    allocation), so the check is free even on a full-size library, falling
    back to a concrete forward pass if a component cannot be traced
    abstractly.
    """

    box = {}

    def _traced(d, p):
        box["trace"] = trace(seed(model_fn, rng_seed=0)).get_trace(data=d,
                                                                   priors=p)
        return jnp.zeros(())

    try:
        jax.eval_shape(_traced, data, priors)
    except Exception:
        box.clear()
        _traced(data, priors)

    return {
        name: tuple(jnp.shape(site["value"]))
        for name, site in box["trace"].items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def orchestrator_latent_dimension(orchestrator):
    """
    Total number of scalar latents in a ``ModelOrchestrator``'s model.

    This is the dimension of the flattened latent vector an
    ``AutoContinuous`` guide (e.g. ``AutoMultivariateNormal``) works in.
    """

    num_genotype = orchestrator.data.num_genotype
    data = orchestrator.get_batch(orchestrator.data, jnp.arange(num_genotype))
    shapes = _latent_shapes(orchestrator.jax_model, orchestrator.priors, data)
    return sum(math.prod(s) for s in shapes.values())


def find_batch_dependent_latents(model_fn, priors, full_data, get_batch,
                                 idx_a, idx_b):
    """
    Find latent sample sites whose shape changes with the batch size.

    Parameters
    ----------
    model_fn : callable
        Numpyro model taking ``data`` and ``priors`` keyword arguments.
    priors : Any
        Priors pytree passed through to ``model_fn``.
    full_data : Any
        Full (un-batched) data pytree.
    get_batch : callable
        ``get_batch(full_data, idx) -> batch_data``.
    idx_a, idx_b : array-like of int
        Two genotype index sets of *different* lengths.

    Returns
    -------
    dict
        Maps each batch-dependent site name to its ``(shape_a, shape_b)``
        pair.  Empty when every latent is safe to mini-batch.

    Raises
    ------
    ValueError
        If ``idx_a`` and ``idx_b`` have the same length (the comparison
        would be meaningless).
    """

    idx_a = jnp.asarray(idx_a)
    idx_b = jnp.asarray(idx_b)
    if idx_a.shape[0] == idx_b.shape[0]:
        raise ValueError(
            "idx_a and idx_b must have different lengths to expose "
            "batch-size-dependent latent shapes."
        )

    shapes_a = _latent_shapes(model_fn, priors, get_batch(full_data, idx_a))
    shapes_b = _latent_shapes(model_fn, priors, get_batch(full_data, idx_b))

    return {
        name: (shapes_a[name], shapes_b[name])
        for name in shapes_a
        if name in shapes_b and shapes_a[name] != shapes_b[name]
    }


def find_orchestrator_batch_dependent_latents(orchestrator):
    """
    Run :func:`find_batch_dependent_latents` on a ``ModelOrchestrator``.

    Traces the model at the full library (``arange(num_genotype)``) and at
    one fewer genotype.  Both index sets are prefixes of the library, so the
    check does not depend on the configured ``batch_size``.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Any object exposing ``jax_model``, ``priors``, ``data`` (with
        ``num_genotype``) and ``get_batch``.

    Returns
    -------
    dict
        See :func:`find_batch_dependent_latents`.
    """

    num_genotype = orchestrator.data.num_genotype
    if num_genotype < 2:
        return {}

    return find_batch_dependent_latents(
        orchestrator.jax_model,
        orchestrator.priors,
        orchestrator.data,
        orchestrator.get_batch,
        idx_a=jnp.arange(num_genotype),
        idx_b=jnp.arange(num_genotype - 1),
    )


# Deterministic sites whose last axis is the batch's genotype positions.
_ORDER_SITES = ("growth_pred", "theta_growth_pred",
                "binding_pred", "theta_binding_pred")


def find_batch_order_mismatches(model_fn, priors, full_data, get_batch,
                                idx_a, idx_b, sites=_ORDER_SITES,
                                rtol=1e-5, atol=1e-6):
    """
    Find batch-positional predictions that do not follow the batch order.

    Traces the model with the genotype index ``idx_a``, then again with
    ``idx_b`` (a reordering of the same genotypes) while substituting every
    latent sampled in the first trace. With the latents fixed, the value a
    genotype gets must not depend on where it sits in the batch: for each
    site in ``sites`` whose last axis has one entry per batch position, the
    ``idx_b`` trace must equal the ``idx_a`` trace reordered to ``idx_b``.

    Parameters
    ----------
    model_fn : callable
        Numpyro model taking ``data`` and ``priors`` keyword arguments.
    priors : Any
        Priors pytree passed through to ``model_fn``.
    full_data : Any
        Full (un-batched) data pytree.
    get_batch : callable
        ``get_batch(full_data, idx) -> batch_data``.
    idx_a, idx_b : array-like of int
        Two orderings of the same genotype indices.
    sites : tuple of str, optional
        Deterministic sites to compare.
    rtol, atol : float, optional
        Tolerances for the comparison.

    Returns
    -------
    dict
        Maps each mismatching site to the largest absolute difference.
        Empty when every compared prediction follows the batch order.

    Raises
    ------
    ValueError
        If ``idx_a`` and ``idx_b`` are not reorderings of the same indices.

    Notes
    -----
    A latent that is itself batch-positional (``noise/beta``'s
    ``{name}_dist``) is substituted in ``idx_a`` order and will show up here
    as a mismatch; it is the documented exception to batch safety.
    """

    idx_a = np.asarray(idx_a, dtype=int)
    idx_b = np.asarray(idx_b, dtype=int)
    if (idx_a.shape != idx_b.shape
            or not np.array_equal(np.sort(idx_a), np.sort(idx_b))):
        raise ValueError("idx_a and idx_b must be reorderings of the same "
                         "genotype indices.")

    data_a = get_batch(full_data, jnp.asarray(idx_a))
    data_b = get_batch(full_data, jnp.asarray(idx_b))

    tr_a = trace(seed(model_fn, rng_seed=0)).get_trace(data=data_a,
                                                       priors=priors)
    latents = {name: site["value"] for name, site in tr_a.items()
               if site["type"] == "sample"
               and not site.get("is_observed", False)}
    tr_b = trace(substitute(seed(model_fn, rng_seed=0),
                            data=latents)).get_trace(data=data_b,
                                                     priors=priors)

    position_a = {int(g): i for i, g in enumerate(idx_a)}
    order = np.array([position_a[int(g)] for g in idx_b])

    found = {}
    for name in sites:
        if name not in tr_a or name not in tr_b:
            continue
        value_a = np.asarray(tr_a[name]["value"])
        value_b = np.asarray(tr_b[name]["value"])
        if value_a.ndim == 0 or value_a.shape[-1] != len(idx_a):
            continue
        expected = value_a[..., order]
        if expected.shape != value_b.shape:
            found[name] = float("inf")
            continue
        if not np.allclose(expected, value_b, rtol=rtol, atol=atol,
                           equal_nan=True):
            diff = np.abs(np.nan_to_num(expected) - np.nan_to_num(value_b))
            found[name] = float(np.max(diff))
    return found


def find_orchestrator_batch_order_mismatches(orchestrator, seed_value=0):
    """
    Run :func:`find_batch_order_mismatches` on a ``ModelOrchestrator``.

    Uses the full-batch training layout: the pinned binding genotypes first
    (``batch_idx[:num_binding]``), then every other genotype. ``idx_a`` has
    the others in library order, ``idx_b`` in a random order, as
    ``get_random_idx`` produces each step.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Any object exposing ``jax_model``, ``priors``, ``data`` (with
        ``batch_idx``, ``num_binding`` and ``not_binding_idx``) and
        ``get_batch``.
    seed_value : int, optional
        Seed for the reordering.

    Returns
    -------
    dict
        See :func:`find_batch_order_mismatches`.
    """

    data = orchestrator.data
    pinned = np.asarray(data.batch_idx)[:data.num_binding]
    rest = np.asarray(data.not_binding_idx)
    if len(rest) < 2:
        return {}

    shuffled = np.random.default_rng(seed_value).permutation(rest)
    if np.array_equal(shuffled, rest):
        shuffled = rest[::-1]

    return find_batch_order_mismatches(
        orchestrator.jax_model,
        orchestrator.priors,
        orchestrator.data,
        orchestrator.get_batch,
        idx_a=np.concatenate([pinned, rest]),
        idx_b=np.concatenate([pinned, shuffled]),
    )
