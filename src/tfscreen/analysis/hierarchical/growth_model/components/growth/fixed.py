import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData,
    ConditionGrowthParams
)
from typing import Tuple

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    growth_k_per_cond_rep: jnp.ndarray 
    growth_m_per_cond_rep: jnp.ndarray 

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> ConditionGrowthParams:
    """
    Growth parameters k_xx and m_xx versus condition, where xx are things like
    pheS+4CP, kanR-kan, etc. These go into the model as k + m*theta. Assigns
    each condition/replicate a normal prior. Returns full k_pre, m_pre, k_sel
    and m_sel tensors.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_condition``
        - ``data.num_replicate``
        - ``data.map_condition_pre``
        - ``data.map_condition_sel``
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model, including:
        - ``priors.growth_k_per_cond_rep``
        - ``priors.growth_m_per_cond_rep``

    Returns
    -------
    params : ConditionGrowthParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.
    """

    # Expand to full-sized tensors
    k_pre = priors.growth_k_per_cond_rep[data.map_condition_pre]
    m_pre = priors.growth_m_per_cond_rep[data.map_condition_pre]
    k_sel = priors.growth_k_per_cond_rep[data.map_condition_sel]
    m_sel = priors.growth_m_per_cond_rep[data.map_condition_sel]

    return ConditionGrowthParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)


def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> ConditionGrowthParams:
    """
    Guide for the fixed growth model.

    This guide simply returns the fixed per-condition/replicate values
    defined in the priors, without registering any learnable parameters.
    """

    # Expand to full-sized tensors
    k_pre = priors.growth_k_per_cond_rep[data.map_condition_pre]
    m_pre = priors.growth_m_per_cond_rep[data.map_condition_pre]
    k_sel = priors.growth_k_per_cond_rep[data.map_condition_sel]
    m_sel = priors.growth_m_per_cond_rep[data.map_condition_sel]

    return ConditionGrowthParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def calculate_growth(params: ConditionGrowthParams,
                     dk_geno: jnp.ndarray,
                     activity: jnp.ndarray,
                     theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the growth rates for pre-selection and selection phases.

    Parameters
    ----------
    params : ConditionGrowthParams
        A dataclass containing k_pre, m_pre, k_sel, m_sel. 
    dk_geno : jnp.ndarray
        Genotype-specific death rate. 
    activity : jnp.ndarray
        Genotype activity. 
    theta : jnp.ndarray
        Occupancy/binding probability. 

    Returns
    -------
    g_pre : jnp.ndarray
        Pre-selection growth rate tensor.
    g_sel : jnp.ndarray
        Selection growth rate tensor.
    """
    
    g_pre = params.k_pre + dk_geno + activity * params.m_pre * theta
    g_sel = params.k_sel + dk_geno + activity * params.m_sel * theta
    
    return g_pre, g_sel



def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}
    parameters["growth_k_per_cond_rep"] = jnp.array([ 0.010696,0.015366,0.021437, 0.028558])
    parameters["growth_m_per_cond_rep"] = jnp.array([-0.009933,0.000808,0.006226,-0.000344])

    return parameters

def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())