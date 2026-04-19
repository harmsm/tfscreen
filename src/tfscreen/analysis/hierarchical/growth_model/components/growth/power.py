import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field
from typing import Tuple, Mapping

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)
from tfscreen.analysis.hierarchical.growth_model.components._pinning import (
    _hyper,
    _pinned_value,
)


# Hyperparameter suffixes that may be pinned via ModelPriors.pinned.
_PINNABLE_SUFFIXES = (
    "k_hyper_loc", "k_hyper_scale",
    "m_hyper_loc", "m_hyper_scale",
    "n_hyper_loc", "n_hyper_scale",
)


@dataclass(frozen=True)
class PowerParams:
    """
    Holds power-law growth parameters for pre-selection and selection phases.
    Growth = k + m * (theta**n)
    """
    k_pre: jnp.ndarray
    m_pre: jnp.ndarray
    n_pre: jnp.ndarray
    k_sel: jnp.ndarray
    m_sel: jnp.ndarray
    n_sel: jnp.ndarray

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    growth_k_hyper_loc_loc: float
    growth_k_hyper_loc_scale: float
    growth_k_hyper_scale: float

    growth_m_hyper_loc_loc: float
    growth_m_hyper_loc_scale: float
    growth_m_hyper_scale: float

    growth_n_hyper_loc_loc: float
    growth_n_hyper_loc_scale: float
    growth_n_hyper_scale: float

    pinned: Mapping[str, float] = field(
        pytree_node=False, default_factory=dict
    )


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> PowerParams:
    """
    Growth parameters k_xx, m_xx, n_xx versus condition. These go into
    the model as k + m*(theta**n). Returns full k_pre, m_pre, n_pre,
    k_sel, m_sel, and n_sel tensors.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites (e.g., "power").
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model. The
        ``pinned`` field selects hyperparameters to hold fixed at constant
        values rather than sampling them.

    Returns
    -------
    params : PowerParams
        A dataclass containing k, m, and n for pre and sel.
    """

    pinned = priors.pinned

    def sample_param(param_name, loc_loc, loc_scale, hyper_scale, is_positive=False):
        hyper_loc = _hyper(
            name,
            f"{param_name}_hyper_loc",
            dist.Normal(loc_loc, loc_scale),
            pinned,
        )
        hyper_scale_val = _hyper(
            name,
            f"{param_name}_hyper_scale",
            dist.HalfNormal(hyper_scale),
            pinned,
        )

        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep):
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(0.0, 1.0))

        param_per_condition = hyper_loc + offset * hyper_scale_val
        if is_positive:
            param_per_condition = jnp.exp(param_per_condition)

        pyro.deterministic(f"{name}_{param_name}", param_per_condition)
        return param_per_condition

    k_per_condition = sample_param("k", priors.growth_k_hyper_loc_loc, priors.growth_k_hyper_loc_scale, priors.growth_k_hyper_scale)
    m_per_condition = sample_param("m", priors.growth_m_hyper_loc_loc, priors.growth_m_hyper_loc_scale, priors.growth_m_hyper_scale)
    n_per_condition = sample_param("n", priors.growth_n_hyper_loc_loc, priors.growth_n_hyper_loc_scale, priors.growth_n_hyper_scale, is_positive=True)

    # Expand to full-sized tensors
    k_pre = k_per_condition[data.map_condition_pre]
    m_pre = m_per_condition[data.map_condition_pre]
    n_pre = n_per_condition[data.map_condition_pre]

    k_sel = k_per_condition[data.map_condition_sel]
    m_sel = m_per_condition[data.map_condition_sel]
    n_sel = n_per_condition[data.map_condition_sel]

    return PowerParams(k_pre=k_pre, m_pre=m_pre, n_pre=n_pre,
                       k_sel=k_sel, m_sel=m_sel, n_sel=n_sel)

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> PowerParams:
    """
    Guide corresponding to the pooled power growth model.
    """

    pinned = priors.pinned

    def sample_guide_param(param_name, loc_loc, loc_scale, is_positive=False):
        loc_suffix = f"{param_name}_hyper_loc"
        scale_suffix = f"{param_name}_hyper_scale"

        pinned_loc = _pinned_value(loc_suffix, pinned)
        if pinned_loc is not None:
            hyper_loc = pinned_loc
        else:
            p_loc_loc = pyro.param(f"{name}_{loc_suffix}_loc", jnp.array(loc_loc))
            p_loc_scale = pyro.param(
                f"{name}_{loc_suffix}_scale", jnp.array(loc_scale),
                constraint=dist.constraints.greater_than(1e-4),
            )
            hyper_loc = pyro.sample(
                f"{name}_{loc_suffix}", dist.Normal(p_loc_loc, p_loc_scale)
            )

        pinned_scale = _pinned_value(scale_suffix, pinned)
        if pinned_scale is not None:
            hyper_scale = pinned_scale
        else:
            p_scale_loc = pyro.param(f"{name}_{scale_suffix}_loc", jnp.array(-1.0))
            p_scale_scale = pyro.param(
                f"{name}_{scale_suffix}_scale", jnp.array(0.1),
                constraint=dist.constraints.greater_than(1e-4),
            )
            hyper_scale = pyro.sample(
                f"{name}_{scale_suffix}",
                dist.LogNormal(p_scale_loc, p_scale_scale),
            )

        offset_locs = pyro.param(f"{name}_{param_name}_offset_locs", jnp.zeros(data.num_condition_rep, dtype=float))
        offset_scales = pyro.param(f"{name}_{param_name}_offset_scales", jnp.ones(data.num_condition_rep, dtype=float),
                                   constraint=dist.constraints.positive)

        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep) as idx:
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(offset_locs[idx], offset_scales[idx]))

        param_per_condition = hyper_loc + offset * hyper_scale
        if is_positive:
            param_per_condition = jnp.exp(param_per_condition)
        return param_per_condition

    k_per_condition = sample_guide_param("k", priors.growth_k_hyper_loc_loc, priors.growth_k_hyper_loc_scale)
    m_per_condition = sample_guide_param("m", priors.growth_m_hyper_loc_loc, priors.growth_m_hyper_loc_scale)
    n_per_condition = sample_guide_param("n", priors.growth_n_hyper_loc_loc, priors.growth_n_hyper_loc_scale, is_positive=True)

    # Expand to full-sized tensors
    k_pre = k_per_condition[data.map_condition_pre]
    m_pre = m_per_condition[data.map_condition_pre]
    n_pre = n_per_condition[data.map_condition_pre]

    k_sel = k_per_condition[data.map_condition_sel]
    m_sel = m_per_condition[data.map_condition_sel]
    n_sel = n_per_condition[data.map_condition_sel]

    return PowerParams(k_pre=k_pre, m_pre=m_pre, n_pre=n_pre,
                       k_sel=k_sel, m_sel=m_sel, n_sel=n_sel)

def calculate_growth(params: PowerParams,
                     dk_geno: jnp.ndarray,
                     activity: jnp.ndarray,
                     theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the growth rates for pre-selection and selection phases.
    Growth = k + dk_geno + activity * m * (theta**n)
    """

    g_pre = params.k_pre + dk_geno + activity * params.m_pre * (theta**params.n_pre)
    g_sel = params.k_sel + dk_geno + activity * params.m_sel * (theta**params.n_sel)

    return g_pre, g_sel


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}
    parameters["growth_k_hyper_loc_loc"] = 0.025
    parameters["growth_k_hyper_loc_scale"] = 0.1
    parameters["growth_k_hyper_scale"] = 0.1

    parameters["growth_m_hyper_loc_loc"] = 0.0
    parameters["growth_m_hyper_loc_scale"] = 0.01
    parameters["growth_m_hyper_scale"] = 0.1

    parameters["growth_n_hyper_loc_loc"] = 0.0 # ln(1.0)
    parameters["growth_n_hyper_loc_scale"] = 0.5
    parameters["growth_n_hyper_scale"] = 0.1

    return parameters

def get_guesses(name, data):
    """
    Get guesses for the model parameters.
    """

    shape = data.num_condition_rep

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = 0.025
    guesses[f"{name}_k_hyper_scale"] = 0.1
    guesses[f"{name}_m_hyper_loc"] = 0.0
    guesses[f"{name}_m_hyper_scale"] = 0.01
    guesses[f"{name}_n_hyper_loc"] = 0.0
    guesses[f"{name}_n_hyper_scale"] = 0.1

    guesses[f"{name}_k_offset"] = jnp.zeros(shape, dtype=float)
    guesses[f"{name}_m_offset"] = jnp.zeros(shape, dtype=float)
    guesses[f"{name}_n_offset"] = jnp.zeros(shape, dtype=float)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
