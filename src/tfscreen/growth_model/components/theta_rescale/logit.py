import jax
import jax.numpy as jnp

_EPS = 1e-6


def rescale(theta: jnp.ndarray) -> jnp.ndarray:
    """Map theta in (0, 1) to the logit scale log(theta / (1 - theta)).

    Clips theta to [_EPS, 1 - _EPS] before applying the logit so that the
    output is always finite.  The same clipping pattern is used throughout
    the rest of the theta pipeline (e.g. _congression.py, hill.py).
    """
    return jax.scipy.special.logit(jnp.clip(theta, _EPS, 1.0 - _EPS))
