"""
Shared pinning helpers for hierarchical growth-model components.

A "pinned" hyperparameter is held fixed at a constant value rather than
being learned. This is mathematically equivalent to placing a delta-prior
on it; in practice it is used to clamp linking-function hyperpriors at
values pre-fit from a separate ML run, which removes the bilinear
pathology that arises when ``param = hyper_loc + offset * hyper_scale``
is sampled jointly with sparse-data subgroups.

The mechanism is:

* ``ModelPriors.pinned`` (on each component) is a ``dict[str, float]``
  keyed by *suffix* (e.g. ``"k_hyper_loc"``). Stored as ``pytree_node=False``
  so the dict is treated as static by JAX tracing.
* In ``define_model``, ``_hyper`` registers a ``pyro.deterministic`` site
  instead of a ``pyro.sample`` for every pinned suffix.
* In ``guide``, the matching ``pyro.param`` / ``pyro.sample`` pair is
  skipped entirely so SVI does not try to compute gradients through
  dropped variational parameters.
"""

from typing import Mapping, Optional

import jax.numpy as jnp
import numpyro as pyro


def _hyper(name: str,
           suffix: str,
           dist_obj,
           pinned: Mapping[str, float]) -> jnp.ndarray:
    """
    Sample a hyperparameter, or pin it to a constant.

    Use this in ``define_model``. When ``suffix`` is in ``pinned``, the
    returned value is the pinned constant and the model registers a
    ``pyro.deterministic`` site (so the value still appears in the trace
    for downstream consumers). Otherwise, the value is drawn from
    ``dist_obj`` via ``pyro.sample``.
    """
    if suffix in pinned:
        val = jnp.asarray(pinned[suffix], dtype=float)
        pyro.deterministic(f"{name}_{suffix}", val)
        return val
    return pyro.sample(f"{name}_{suffix}", dist_obj)


def _pinned_value(suffix: str,
                  pinned: Mapping[str, float]) -> Optional[jnp.ndarray]:
    """
    Return the pinned constant for ``suffix``, or ``None`` if not pinned.

    Use this in ``guide`` to short-circuit variational parameter
    registration when a hyperparameter is pinned.
    """
    if suffix in pinned:
        return jnp.asarray(pinned[suffix], dtype=float)
    return None
