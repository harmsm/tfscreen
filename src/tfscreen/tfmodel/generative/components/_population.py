"""
Per-genotype values for the batch, and optionally for the whole library.

Per-genotype components (``dk_geno``, ``activity``) sample their latents at
library size and slice them to the batch with ``batch_idx``. The congression
mixture also needs the *library-ordered* values of every genotype, so it can
look up a genotype's co-resident plasmids by index (see
``planning/congression-physics-plan.md``, step 3). :func:`per_genotype`
computes both from the same draw, without adding sample sites.
"""

import jax.numpy as jnp


def per_genotype(compute, latents, data, return_population=False):
    """
    Evaluate a per-genotype quantity for the batch and, optionally, the library.

    Parameters
    ----------
    compute : callable
        ``compute(*latents, genotype_idx) -> values``: elementwise in the
        latents along their last (genotype) axis; ``genotype_idx`` gives the
        library index of each entry (used, e.g., to pin wildtype).
    latents : sequence of jnp.ndarray
        Per-genotype latents, genotype on the last axis. Each is either
        library-sized (``num_genotype``, sampled in a library plate) or an
        already-sliced batch-sized substitution (the posterior forward pass).
    data : GrowthData
        Uses ``num_genotype``, ``batch_idx``.
    return_population : bool, default False
        Also return the library-ordered values.

    Returns
    -------
    batch_values : jnp.ndarray
        Values for the batch's genotypes, in batch order.
    population : jnp.ndarray or None
        Library-ordered values for every genotype, shape ``(...,
        num_genotype)``. None unless ``return_population`` and every latent
        is library-sized; a batch-sized substitution carries no information
        about the rest of the library, so callers must fall back to an
        external reference.

    Notes
    -----
    With ``return_population=False`` this evaluates exactly the old batch-only
    path (slice, then compute), so it costs nothing extra under
    mini-batching. With ``return_population=True`` it computes on the library
    and then slices; because ``compute`` is elementwise the batch values are
    identical.
    """

    library_sized = all(jnp.shape(x)[-1] == data.num_genotype for x in latents)

    if return_population and library_sized:
        population = compute(*latents, jnp.arange(data.num_genotype))
        return population[..., data.batch_idx], population

    sliced = [x[..., data.batch_idx] if jnp.shape(x)[-1] == data.num_genotype
              else x for x in latents]
    return compute(*sliced, data.batch_idx), None
