"""
A library of mathematical models for extracting growth rates from cfu vs. time
data. The core data structure defined here is MODEL_LIBRARY, which keys model
names to the function used in the model and the arguments that must be passed
to the fitter. 
"""

from .ols import ols
from .wls import wls
from .kf import kf
from .ukf import ukf
from .ukf_lin import ukf_lin
from .gls import gls
from .glm import glm
from .gee import gee
from .nls import nls


MODEL_LIBRARY = {
    "ols":{
        "fcn":ols,
        "args":["t_sel",
                "ln_cfu"],
    },
    "wls":{
        "fcn":wls,
        "args":["t_sel",
                "ln_cfu",
                "ln_cfu_var"],
    },
    "gee":{
        "fcn":gee,
        "args":["t_sel",
                "cfu"],
    },
    "gls":{
        "fcn":gls,
        "args":["t_sel",
                "ln_cfu"],
    },
    "glm":{
        "fcn":glm,
        "args":["t_sel",
                "cfu"],
    },
    "kf":{
        "fcn":kf,
        "args":["t_sel",
                "ln_cfu",
                "ln_cfu_var",
                "growth_rate_wls"],
    },
    "ukf":{
        "fcn":ukf,
        "args":["t_sel",
                "cfu",
                "cfu_var",
                "growth_rate_wls",
                "growth_rate_err_wls"],
    },
    "ukf_lin":{
        "fcn":ukf_lin,
        "args":["t_sel",
                "ln_cfu",
                "ln_cfu_var",
                "growth_rate_wls",
                "growth_rate_err_wls"],
    },
    "nls":{
        "fcn":nls,
        "args":["t_sel",
                "cfu",
                "cfu_var"],
    }
}