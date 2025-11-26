
from . import components

from .components.growth_independent import define_model as define_growth_independent
from .components.growth_hierarchical import define_model as define_growth_hierarchical
from .components.growth_fixed import define_model as define_growth_fixed

from .components.ln_cfu0 import define_model as define_ln_cfu0

from .components.dk_geno_fixed import define_model as define_dk_geno_fixed
from .components.dk_geno_hierarchical import define_model as define_dk_geno_hierarchical

from .components.activity_fixed import define_model as define_activity_fixed
from .components.activity_hierarchical import define_model as define_activity_hierarchical
from .components.activity_horseshoe import define_model as define_activity_horseshoe

from .components.theta_cat import define_model as define_theta_cat
from .components.theta_cat import run_model as calc_theta_cat
from .components.theta_hill import define_model as define_theta_hill
from .components.theta_hill import run_model as calc_theta_hill

from .components.no_noise import define_model as define_no_noise
from .components.beta_noise import define_model as define_beta_noise

from .observe.binding import observe as observe_binding
from .observe.growth import observe as observe_growth

# Import for typing
from .data_class import (
    DataClass,
    PriorsClass,
    ControlClass
)

import jax.numpy as jnp
import numpyro as pyro
from typing import Dict, Any

CONDITION_GROWTH_INDEPENDENT = 0
CONDITION_GROWTH_HIERARCHICAL = 1
CONDITION_GROWTH_FIXED = 2
LN_CFU0_HIERARCHICAL = 0
DK_GENO_FIXED = 0
DK_GENO_HIERARCHICAL = 1
ACTIVITY_FIXED = 0
ACTIVITY_HIERARCHICAL = 1
ACTIVITY_HORSESHOE = 2
THETA_CATEGORICAL = 0
THETA_HILL = 1
THETA_GROWTH_NOISE_NONE = 0
THETA_GROWTH_NOISE_BETA = 1
THETA_BINDING_NOISE_NONE = 0
THETA_BINDING_NOISE_BETA = 1

MODEL_COMPONENT_NAMES = {
    "condition_growth":{
        "independent":(CONDITION_GROWTH_INDEPENDENT,
                       components.growth_independent),
        "hierarchical":(CONDITION_GROWTH_HIERARCHICAL,
                        components.growth_hierarchical),
        "fixed":(CONDITION_GROWTH_FIXED,
                   components.growth_fixed)
    },
    "ln_cfu0":{
        "hierarchical":(LN_CFU0_HIERARCHICAL,
                        components.ln_cfu0),
    },
    "dk_geno":{
        "fixed":(DK_GENO_FIXED,
                 components.dk_geno_fixed),
        "hierarchical":(DK_GENO_HIERARCHICAL,
                        components.dk_geno_hierarchical),
    },
    "activity":{
        "fixed":(ACTIVITY_FIXED,
                 components.activity_fixed),
        "hierarchical":(ACTIVITY_HIERARCHICAL,
                        components.activity_hierarchical),
        "horseshoe":(ACTIVITY_HORSESHOE,
                     components.activity_horseshoe),
    },
    "theta":{
        "categorical":(THETA_CATEGORICAL,components.theta_cat),
        "hill":(THETA_HILL,components.theta_hill)
    },
    "theta_growth_noise":{
        "none":(THETA_GROWTH_NOISE_NONE,components.no_noise),
        "beta":(THETA_GROWTH_NOISE_BETA,components.beta_noise),
    },
    "theta_binding_noise":{
        "none":(THETA_BINDING_NOISE_NONE,components.no_noise),
        "beta":(THETA_BINDING_NOISE_BETA,components.beta_noise),
    }
}

def _define_growth(data: DataClass, 
                   priors: PriorsClass, 
                   control: ControlClass, 
                   theta: Any) -> jnp.ndarray:
    """
    Defines the growth model components and calculates predicted ln(CFU).

    This function constructs the deterministic prediction for the log-transformed
    colony-forming units (ln_cfu_pred) based on the model components
    selected by the 'control' object. It combines base growth rates,
    genotype-specific effects, initial cell counts, and the predicted
    fractional occupancy (theta) for the growth conditions.

    The core growth equation is:
    g_pre = k_pre + dk_geno + activity*m_pre*noisy_growth_theta
    g_sel = k_sel + dk_geno + activity*m_sel*noisy_growth_theta
    ln_cfu_pred = ln_cfu0 + g_pre*data.growth.t_pre + g_sel*data.growth.t_sel

    Parameters
    ----------
    data : DataClass
        The main DataClass object, primarily using data.growth.
    priors : PriorsClass
        The main PriorsClass object, primarily using priors.growth.
    control : ControlClass
        The ControlClass object containing integer flags to select
        which sub-model to use for each component (e.g., condition_growth,
        dk_geno, activity).
    theta : dict[str, jnp.ndarray]
        A pytree (typically a dict) containing the sampled latent
        parameters for the fractional occupancy (theta) model, as defined
        in the main jax_model function.

    Returns
    -------
    jnp.ndarray
        A jnp.ndarray representing the deterministically predicted ln(CFU)
        values for all growth experiments.

    Raises
    ------
    ValueError
        If an invalid integer flag is provided in the control object for any
        component.
    """

    # -------------------------------------------------------------------------
    # Define theta for the growth model
    
    # calculate theta under growth conditions and scatter (returns full-sized
    # tensor)
    if control.theta == THETA_CATEGORICAL:
        calc_theta = calc_theta_cat
    elif control.theta == THETA_HILL:
        calc_theta = calc_theta_hill
    else:
        raise ValueError(
            f"theta selection {control.theta} is invalid"
        )

    growth_theta = calc_theta(theta,data.growth)
    pyro.deterministic(f"theta_growth_pred",growth_theta)

    # -------------------------------------------------------------------------
    # Define growth noise on theta

    if control.theta_growth_noise == THETA_GROWTH_NOISE_NONE:
        noisy_growth_theta = define_no_noise("theta_growth_noise",
                                             growth_theta,
                                             priors.growth.theta_growth_noise)
    elif control.theta_growth_noise == THETA_GROWTH_NOISE_BETA:
        noisy_growth_theta = define_beta_noise("theta_growth_noise",
                                               growth_theta,
                                               priors.growth.theta_growth_noise)
    else:
        raise ValueError (
            f"theta_growth_noise selection {control.theta_growth_noise} is invalid"
        )


    # -------------------------------------------------------------------------
    # Define base growth model

    if control.condition_growth == CONDITION_GROWTH_INDEPENDENT:
        k_pre, m_pre, k_sel, m_sel = define_growth_independent("condition_growth",
                                                               data.growth,
                                                               priors.growth.condition_growth)
    elif control.condition_growth == CONDITION_GROWTH_HIERARCHICAL:
        k_pre, m_pre, k_sel, m_sel = define_growth_hierarchical("condition_growth",
                                                                data.growth,
                                                                priors.growth.condition_growth)
    elif control.condition_growth == CONDITION_GROWTH_FIXED:
        k_pre, m_pre, k_sel, m_sel = define_growth_fixed("condition_growth",
                                                         data.growth,
                                                         priors.growth.condition_growth)
    else:
        raise ValueError (
            f"condition_growth selection {control.condition_growth} is invalid"
        )

    # -------------------------------------------------------------------------
    # Define ln_cfu0 model

    if control.ln_cfu0 == LN_CFU0_HIERARCHICAL:
        ln_cfu0 = define_ln_cfu0("ln_cfu0",
                                 data.growth,
                                 priors.growth.ln_cfu0)
    else:
        raise ValueError (
            f"ln_cfu0 selection {control.ln_cfu0} is invalid"
        )

    # -------------------------------------------------------------------------
    # Define dk_geno model

    if control.dk_geno == DK_GENO_FIXED:
        dk_geno = define_dk_geno_fixed("dk_geno",
                                       data.growth,
                                       priors.growth.dk_geno)    
    elif control.dk_geno == DK_GENO_HIERARCHICAL:
        dk_geno = define_dk_geno_hierarchical("dk_geno",
                                              data.growth,
                                              priors.growth.dk_geno)
    else:
        raise ValueError (
            f"dk_geno selection {control.dk_geno} is invalid"
        )

    # -------------------------------------------------------------------------
    # Define activity model

    if control.activity == ACTIVITY_FIXED:
        activity = define_activity_fixed("activity",
                                         data.growth,
                                         priors.growth.activity)
    elif control.activity == ACTIVITY_HIERARCHICAL:
        activity = define_activity_hierarchical("activity",
                                                data.growth,
                                                priors.growth.activity)
    elif control.activity == ACTIVITY_HORSESHOE:
        activity = define_activity_horseshoe("activity",
                                             data.growth,
                                             priors.growth.activity)
    else:
        raise ValueError (
            f"activity selection {control.activity} is invalid"
        )
        

    # Calculate growth. All variables have same tensor dimensions. 
    g_pre = k_pre + dk_geno + activity*m_pre*noisy_growth_theta
    g_sel = k_sel + dk_geno + activity*m_sel*noisy_growth_theta
    ln_cfu_pred = ln_cfu0 + g_pre*data.growth.t_pre + g_sel*data.growth.t_sel

    return ln_cfu_pred

def _define_binding(data: DataClass, 
                    priors: PriorsClass, 
                    control: ControlClass, 
                    theta: Any) -> jnp.ndarray:
    """
    Calculates the predicted fractional occupancy (theta) for binding data.

    This function takes the shared latent theta parameters and computes the
    specific fractional occupancy values predicted for the binding
    experiment's conditions (e.g., titrant concentrations). It then
    applies a selected noise model (e.g., Beta noise) to these
    deterministic predictions.

    Parameters
    ----------
    data : DataClass
        The main DataClass object, primarily using data.binding.
    priors : PriorsClass
        The main PriorsClass object, primarily using priors.binding.
    control : ControlClass
        The ControlClass object containing integer flags to select
        the theta model (categorical vs. hill) and the noise model.
    theta : dict[str, jnp.ndarray]
        A pytree (typically a dict) containing the sampled latent
        parameters for the fractional occupancy (theta) model, as defined

    Returns
    -------
    jnp.ndarray
        A jnp.ndarray representing the deterministically predicted (and noised)
        fractional occupancy values for all binding experiments.

    Raises
    ------
    ValueError
        If an invalid integer flag is provided in the
        control object for the theta or noise components.
    """

    if control.theta == THETA_CATEGORICAL:
        calc_theta = calc_theta_cat
    elif control.theta == THETA_HILL:
        calc_theta = calc_theta_hill
    else:
        raise ValueError(
            f"theta selection {control.theta} is invalid"
        )

    theta_binding = calc_theta(theta,data.binding)
    pyro.deterministic(f"theta_binding_pred",theta_binding)

    # -------------------------------------------------------------------------
    # Define binding noise on theta model

    if control.theta_binding_noise == THETA_BINDING_NOISE_NONE:
        noisy_binding_theta = define_no_noise("theta_binding_noise",
                                              theta_binding,
                                              priors.binding.theta_binding_noise)
    elif control.theta_binding_noise == THETA_BINDING_NOISE_BETA:
        noisy_binding_theta = define_beta_noise("theta_binding_noise",
                                                theta_binding,
                                                priors.binding.theta_binding_noise)
    else:
        raise ValueError (
            f"theta_binding_noise selection {control.theta_binding_noise} is invalid"
        )
    
    return noisy_binding_theta


def jax_model(data: DataClass, priors: PriorsClass, control: ControlClass):
    """
    Defines the joint hierarchical model for bacterial growth and binding.

    This is the main Numpyro model function. It defines the shared latent
    parameters for fractional occupancy (theta), then calls helper functions
    to construct the deterministic predictions for both the growth
    (_define_growth) and binding (_define_binding) assays. Finally, it
    registers the likelihood of the observed data against these predictions
    using the 'observe_growth' and 'observe_binding' functions.

    The model structure is dynamically controlled by the 'control' object,
    which selects the specific implementation for each model component
    (e.g., hierarchical vs. fixed, Hill vs. categorical).

    Parameters
    ----------
    data : DataClass
        A DataClass pytree containing all observed data and experimental
        metadata for both growth and binding assays.
    priors : PriorsClass
        A PriorsClass pytree containing the prior distributions
        (as Numpyro distribution objects) for all latent parameters.
    control : ControlClass
        A ControlClass pytree containing integer flags that
        determine which sub-model to use for each component of the
        hierarchical model.

    Raises
    ------
    ValueError
        If an invalid integer flag is provided in the
        control object for the main theta component.
    """

    # -------------------------------------------------------------------------
    # Start by calculating fractional occupancy of the transcription factor
    # (theta). This is used for both the growth and binding calculations. 

    # Define theta
    if control.theta == THETA_CATEGORICAL:
        theta = define_theta_cat("theta",
                                 data.growth,
                                 priors.theta)
    elif control.theta == THETA_HILL:
        # passing data.growth enforces it as the source of truth for the 
        # titrant_name and genotype seen across both datasets. 
        theta = define_theta_hill("theta",
                                  data.growth, 
                                  priors.theta)
    else:
        raise ValueError (
            f"theta selection {control.theta} is invalid"
        )

    # predict ln_cfu and binding
    ln_cfu_pred = _define_growth(data,priors,control,theta)
    binding_pred = _define_binding(data,priors,control,theta)

    pyro.deterministic(f"growth_pred",ln_cfu_pred)
    pyro.deterministic(f"binding_pred",binding_pred)

    # make final observations
    observe_growth("final_obs",data.growth,ln_cfu_pred)
    observe_binding("final_obs",data.binding,binding_pred)
    
