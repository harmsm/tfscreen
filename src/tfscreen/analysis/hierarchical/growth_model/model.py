
from . import components

from .components.growth_independent import define_model as define_growth_independent
from .components.growth_hierarchical import define_model as define_growth_hierarchical

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

from .components.beta_noise import define_model as define_no_noise
from .components.beta_noise import define_model as define_beta_noise

from .observe import observe


MODEL_COMPONENT_NAMES = {
    "condition_growth":{
        "independent":(0,components.growth_independent),
        "hierarchical":(1,components.growth_hierarchical)
    },
    "ln_cfu0":{
        "hierarchical":(0,components.ln_cfu0),
    },
    "dk_geno":{
        "fixed":(0,components.dk_geno_fixed),
        "hierarchical":(1,components.dk_geno_hierarchical),
    },
    "activity":{
        "fixed":(0,components.activity_fixed),
        "hierarchical":(1,components.activity_hierarchical),
        "horseshoe":(2,components.activity_horseshoe),
    },
    "theta":{
        "categorical":(0,components.theta_cat),
        "hill":(1,components.theta_hill)
    },
    "theta_growth_noise":{
        "none":(0,components.no_noise),
        "beta":(1,components.beta_noise),
    },
    "theta_binding_noise":{
        "none":(0,components.no_noise),
        "beta":(1,components.beta_noise),
    }
}

def _define_growth(data,priors,control,theta,calc_theta):

    # -------------------------------------------------------------------------
    # Define theta for the growth model
    
    # calculate theta under growth conditions and scatter (returns full-sized
    # tensor)
    growth_theta = calc_theta(theta,data.growth)

    # -------------------------------------------------------------------------
    # Define growth noise on theta

    if control.theta_growth_noise == 0:
        noisy_growth_theta = define_no_noise("theta_growth_noise",
                                             growth_theta,
                                             priors.growth.theta_growth_noise)
    elif control.theta_growth_noise == 1:
        noisy_growth_theta = define_beta_noise("theta_growth_noise",
                                               growth_theta,
                                               priors.growth.theta_growth_noise)
    else:
        raise ValueError (
            f"theta_growth_noise selection {control.theta_growth_noise} is invalid"
        )


    # -------------------------------------------------------------------------
    # Define base growth model

    if control.condition_growth == 0:
        k_pre, m_pre, k_sel, m_sel = define_growth_independent("condition_growth",
                                                               data.growth,
                                                               priors.growth.condition_growth)
    if control.condition_growth == 1:
        k_pre, m_pre, k_sel, m_sel = define_growth_hierarchical("condition_growth",
                                                                data.growth,
                                                                priors.growth.condition_growth)
    else:
        raise ValueError (
            f"condition_growth selection {control.condition_growth} is invalid"
        )

    # -------------------------------------------------------------------------
    # Define ln_cfu0 model

    if control.ln_cfu0 == 0:
        ln_cfu0 = define_ln_cfu0("ln_cfu0",
                                 data.growth,
                                 priors.growth.ln_cfu0)
    else:
        raise ValueError (
            f"ln_cfu0 selection {control.ln_cfu0} is invalid"
        )

    # -------------------------------------------------------------------------
    # Define dk_geno model

    if control.dk_geno == 0:
        dk_geno = define_dk_geno_fixed("dk_geno",
                                       data.growth,
                                       priors.growth.dk_geno)    
    if control.dk_geno == 1:
        dk_geno = define_dk_geno_hierarchical("dk_geno",
                                              data.growth,
                                              priors.growth.dk_geno)
    else:
        raise ValueError (
            f"dk_geno selection {control.dk_geno} is invalid"
        )

    # -------------------------------------------------------------------------
    # Define activity model

    if control.activity == 0:
        activity = define_activity_fixed("activity",
                                         data.growth,
                                         priors.growth.activity)
    elif control.activity == 1:
        activity = define_activity_hierarchical("activity",
                                                data.growth,
                                                control.growth.activity)
    elif control.activity == 2:
        activity = define_activity_horseshoe("activity",
                                             data.growth,
                                             control.growth.activity)
    else:
        raise ValueError (
            f"activity_model selection {control.activity} is invalid"
        )
        

    # Calculate growth. All variables have same tensor dimensions. 
    g_pre = k_pre + dk_geno + activity*m_pre*noisy_growth_theta
    g_sel = k_sel + dk_geno + activity*m_sel*noisy_growth_theta
    ln_cfu_pred = ln_cfu0 + g_pre*data.growth.t_pre + g_sel*data.growth.t_sel

    return ln_cfu_pred

def _define_binding(data,priors,control,theta,calc_theta):

    theta_binding = calc_theta(theta,data.binding)

    # -------------------------------------------------------------------------
    # Define binding noise on theta model

    if control.theta_binding_noise == 0:
        noisy_binding_theta = define_no_noise("theta-binding-noise",
                                              theta_binding,
                                              priors.binding.theta_binding_noise)
    elif control.theta_binding_noise == 1:
        noisy_binding_theta = define_beta_noise("theta-binding-noise",
                                                theta_binding,
                                                priors.binding.theta_binding_noise)
    else:
        raise ValueError (
            f"theta_binding_noise selection {control.theta_binding_noise} is invalid"
        )
    
    return noisy_binding_theta


def jax_model(data,priors,control):
    """
    Model growth rates of bacteria in culture.
    """

    # -------------------------------------------------------------------------
    # Start by calculating fractional occupancy of the transcription factor
    # (theta). This is used for both the growth and binding calculations. 

    # Define theta
    if control.theta == 0:
        theta = define_theta_cat("theta",
                                 data,
                                 priors.theta)
        calc_theta = calc_theta_cat
    elif control.theta == 1:
        # passing data.growth enforces it as the source of truth for the 
        # titrant_name and genotype seen across both datasets. 
        theta = define_theta_hill("theta",
                                  data.growth, 
                                  priors.theta)
        calc_theta = calc_theta_hill
    else:
        raise ValueError (
            f"theta selection {control.theta} is invalid"
        )

    # predict ln_cfu and binding
    ln_cfu_pred = _define_growth(data,priors,control,theta,calc_theta)
    binding_pred = _define_binding(data,priors,control,theta,calc_theta)

    # make final observation
    observe("final-obs",data,ln_cfu_pred,binding_pred)

  
    

