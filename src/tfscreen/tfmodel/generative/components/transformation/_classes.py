"""
The cell classes a transformation component hands to ``jax_model``.
"""

from typing import NamedTuple

import jax.numpy as jnp


class CellClasses(NamedTuple):
    """
    A genotype's cells split into classes that each grow at their own rate.

    Every field has a leading class axis of size C (the clean class first);
    the remaining axes follow the growth layout and broadcast against
    ``(replicate, time, condition_pre, condition_sel, titrant_name,
    titrant_conc, batch)``.

    Attributes
    ----------
    theta : jnp.ndarray
        Cell-level operator occupancy, before ``theta_rescale``.
    activity : jnp.ndarray
        Cell-level TF activity.
    dk_geno : jnp.ndarray
        Cell-level pleiotropic growth effect.
    log_weight : jnp.ndarray
        Log fraction of the genotype's cells in each class; sums (in linear
        space) to 1 over the class axis.
    """

    theta: jnp.ndarray
    activity: jnp.ndarray
    dk_geno: jnp.ndarray
    log_weight: jnp.ndarray


def single_class(theta, activity, dk_geno) -> CellClasses:
    """One class holding every cell (no congression)."""
    return CellClasses(theta=theta[None],
                       activity=jnp.asarray(activity)[None],
                       dk_geno=jnp.asarray(dk_geno)[None],
                       log_weight=jnp.zeros((1,) + (1,) * jnp.ndim(theta)))
