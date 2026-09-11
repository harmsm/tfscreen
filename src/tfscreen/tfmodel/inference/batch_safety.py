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
"""

import jax.numpy as jnp
from numpyro.handlers import seed, trace


def _latent_shapes(model_fn, priors, data):
    """Return {site_name: shape} for every unobserved sample site."""

    model_trace = trace(seed(model_fn, rng_seed=0)).get_trace(data=data,
                                                              priors=priors)
    return {
        name: tuple(jnp.shape(site["value"]))
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


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
