"""
Overflow-safe pieces of the regularized horseshoe prior.

The horseshoe components (theta ``hill_mut``, the thermo ``PK``/``PnnC``/
``PddG`` variants and ``thermo/horseshoe.py``, activity ``horseshoe_mut``)
draw local scales ``lambda ~ HalfCauchy`` and use the slab-regularized scale

    lambda_tilde = sqrt(c2 lambda^2 / (c2 + tau^2 lambda^2)).

Both were written with ``lambda ** 2``. The half-Cauchy tail is heavy and,
where the data stop constraining an effect (a theta already saturated), a
mean-field guide can widen a local scale's LogNormal until a draw reaches
~1e20; its square overflows float32, ``lambda_tilde`` becomes inf / inf =
NaN, the half-Cauchy log density becomes -inf, and the whole fit goes NaN
(count-likelihood grid, run_0007, 2026-09-27). The forms here use
``jnp.hypot``, which does not overflow, and give the same values.
"""

import jax.numpy as jnp
import numpyro.distributions as dist


def regularized_scale(lam, tau, c2):
    """
    ``sqrt(c2 lam^2 / (c2 + tau^2 lam^2))`` without squaring ``lam``.

    Written as ``lam / hypot(1, tau lam / sqrt(c2))``: it tends to
    ``sqrt(c2) / tau`` as ``lam`` grows, and equals ``lam`` at ``tau = 0``
    (epistasis switched off) with finite gradients.
    """
    return lam / jnp.hypot(1.0, tau * lam / jnp.sqrt(c2))


class HalfCauchy(dist.HalfCauchy):
    """
    numpyro's HalfCauchy with a log density that does not overflow:
    ``log1p(z^2)`` is computed as ``2 log hypot(1, z)``. numpyro's returns
    -inf (and NaN gradients) once ``(x / scale)^2`` exceeds float32.
    """

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = value / self.scale
        return (jnp.log(2.0 / jnp.pi) - jnp.log(self.scale)
                - 2.0 * jnp.log(jnp.hypot(1.0, z)))
