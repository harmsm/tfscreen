"""
A "simple" (always-pinned) theta component.

This component is intended for the *calibration* / pre-fit pass of the
two-stage hierarchical workflow.  It treats the per-condition fractional
occupancy ``theta`` as a fully known, deterministic quantity supplied by
the caller — typically computed from a thermodynamic model evaluated at
the wildtype TF parameters.

There are no sampled latent variables in this component:
- ``define_model`` registers zero ``pyro.sample`` sites and only emits a
  ``pyro.deterministic(f"{name}_theta", ...)`` site for visibility.
- ``guide`` is the same pure no-op (no ``pyro.param`` and no
  ``pyro.sample`` sites), keeping model/guide sample sets symmetric.

Because ``theta`` is identical across all genotypes, the broadcast tensor
returned in ``ThetaParam.theta`` has shape
``(num_titrant_name, num_titrant_conc, num_genotype)`` but is constant
along the genotype axis.

The population moments ``(mu, sigma)`` are derived from the supplied
``theta_values``:
- ``mu``  = ``logit(theta_values)``  (with eps clipping)
- ``sigma`` = a small floor (``priors.sigma_floor``)

These match the contract used by downstream transformations (e.g.
``transformation.logit_norm``) that need a population mean / scale even
when theta is "known".
"""

import jax
import jax.numpy as jnp
import numpyro as pyro
from flax.struct import dataclass, field
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass


_THETA_EPS = 1e-6


def _logit(theta: jnp.ndarray, eps: float = _THETA_EPS) -> jnp.ndarray:
    """Clipped logit transform that is safe at theta in {0, 1}."""
    theta_clipped = jnp.clip(theta, eps, 1.0 - eps)
    return jnp.log(theta_clipped) - jnp.log1p(-theta_clipped)


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding the (always-pinned) inputs for the simple theta model.

    Attributes
    ----------
    theta_values : jnp.ndarray
        Pre-computed fractional occupancy values, shape
        ``(num_titrant_name, num_titrant_conc)``.  These are treated as
        exact (delta-prior) and used unchanged for every genotype.
    sigma_floor : float
        A small floor for the population-sigma in logit-space, returned
        as ``ThetaParam.sigma`` so downstream transformations have a
        well-defined positive scale even though theta is "known".
    """

    theta_values: jnp.ndarray
    sigma_floor: float = field(pytree_node=False, default=0.01)


@dataclass(frozen=True)
class ThetaParam:
    """
    JAX Pytree holding the deterministic theta tensors.

    Attributes
    ----------
    theta : jnp.ndarray
        Fractional occupancy, shape
        ``(num_titrant_name, num_titrant_conc, num_genotype)``, constant
        along the genotype axis.
    mu : jnp.ndarray
        Population mean of ``logit(theta)``, shape
        ``(num_titrant_name, num_titrant_conc, 1)``.
    sigma : jnp.ndarray
        Population scale of ``logit(theta)``, shape
        ``(num_titrant_name, num_titrant_conc, 1)``; constant
        ``priors.sigma_floor``.
    concentrations : jnp.ndarray
        The concentrations associated with the categories in theta
        (passed through from ``data.titrant_conc``).
    """

    theta: jnp.ndarray
    mu: jnp.ndarray
    sigma: jnp.ndarray
    concentrations: jnp.ndarray


def _build_theta_param(name: str,
                       data: DataClass,
                       priors: ModelPriors,
                       register_deterministic: bool) -> ThetaParam:
    """
    Shared constructor for ``define_model`` and ``guide``.

    Both functions return identical, deterministic results.  The only
    difference is that ``define_model`` emits a ``pyro.deterministic``
    site for inspection, while ``guide`` does not (sites in a guide must
    not be observed).
    """

    theta_values = priors.theta_values  # (num_titrant_name, num_titrant_conc)

    # Broadcast across genotype axis -> (Name, Conc, Genotype).
    theta = jnp.broadcast_to(
        theta_values[..., None],
        (data.num_titrant_name, data.num_titrant_conc, data.num_genotype),
    )

    # Population moments in logit-space, shape (Name, Conc, 1)
    mu = _logit(theta_values)[..., None]
    sigma = jnp.full_like(mu, priors.sigma_floor)

    if register_deterministic:
        pyro.deterministic(f"{name}_theta", theta)

    return ThetaParam(
        theta=theta,
        mu=mu,
        sigma=sigma,
        concentrations=data.titrant_conc,
    )


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Defines the simple (always-pinned) theta component.

    No sample sites are registered.  A single ``pyro.deterministic`` site
    named ``f"{name}_theta"`` is emitted so the broadcast theta tensor
    appears in the trace.

    Parameters
    ----------
    name : str
        Prefix for the deterministic site.
    data : DataClass
        Data object exposing ``num_titrant_name``, ``num_titrant_conc``,
        ``num_genotype`` and ``titrant_conc``.
    priors : ModelPriors
        Holds the pinned ``theta_values`` and ``sigma_floor``.

    Returns
    -------
    ThetaParam
        Pytree with the deterministic theta tensor and population
        moments.
    """
    return _build_theta_param(name, data, priors,
                              register_deterministic=True)


def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """
    Guide for the simple theta component.

    Pure no-op: no ``pyro.param`` and no ``pyro.sample`` sites.  Returns
    the same ``ThetaParam`` object as ``define_model``, allowing callers
    to invoke the guide independently for diagnostics or prediction.
    """
    return _build_theta_param(name, data, priors,
                              register_deterministic=False)


def run_model(theta_param: ThetaParam, data: DataClass) -> jnp.ndarray:
    """
    Project the (Name, Conc, Genotype) theta tensor to the experiment's
    grid of concentrations and (optionally) scatter to the full
    observation tensor.

    Mirrors ``categorical.run_model`` so this component is a drop-in
    replacement.

    Parameters
    ----------
    theta_param : ThetaParam
        Output of ``define_model`` / ``guide``.
    data : DataClass
        Must expose ``geno_theta_idx``, ``titrant_conc``, and
        ``scatter_theta``.

    Returns
    -------
    jnp.ndarray
        - ``data.scatter_theta == 0`` →
          shape ``(num_titrant_name, num_titrant_conc, num_genotype_subset)``.
        - ``data.scatter_theta == 1`` →
          shape ``(1, 1, 1, 1, num_titrant_name, num_titrant_conc, num_genotype_subset)``.
    """

    # Subset on the genotype axis using the run-time genotype map.
    theta_base = theta_param.theta[..., data.geno_theta_idx]

    # Map this experiment's concentrations onto the columns of theta_base.
    conc_idx = jnp.searchsorted(theta_param.concentrations, data.titrant_conc)
    conc_idx = jnp.clip(conc_idx, 0, theta_param.concentrations.shape[0] - 1)

    theta_calc = theta_base[:, conc_idx, :]

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data: DataClass) -> tuple:
    """
    Returns the (mu, sigma) population moments in logit-space.

    For the simple component these are derived from the pinned
    ``theta_values`` (mu = logit(theta_values), sigma = floor).
    """
    return theta_param.mu, theta_param.sigma


def get_hyperparameters() -> Dict[str, Any]:
    """
    Default hyperparameters for the simple theta component.

    The returned ``theta_values`` is a tiny placeholder; in production
    use the YAML / glue layer overrides this with a real
    ``(num_titrant_name, num_titrant_conc)`` tensor computed from the
    calibration model.

    Returns
    -------
    dict[str, Any]
        Mapping with placeholder ``theta_values`` and a default
        ``sigma_floor``.
    """
    parameters = {}
    parameters["theta_values"] = jnp.full((1, 1), 0.5, dtype=float)
    parameters["sigma_floor"] = 0.01
    return parameters


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    No latent parameters → no guesses.

    Returns
    -------
    dict[str, Any]
        Empty dictionary.
    """
    return {}


def get_priors() -> ModelPriors:
    """
    Utility to construct a populated ``ModelPriors``.

    Note: the default ``theta_values`` returned here is a placeholder.
    Pass an explicit ``theta_values`` keyword to construct a real one.
    """
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    return []
