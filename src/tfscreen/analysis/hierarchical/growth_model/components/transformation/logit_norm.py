from functools import partial
from . import congression

def get_hyperparameters():
    """
    Gets default values for the model hyperparameters, forced to logit_norm mode.
    """
    parameters = congression.get_hyperparameters()
    parameters["mode"] = "logit_norm"
    return parameters

def get_priors():
    """
    Gets model priors initialized to logit_norm mode.
    """
    return congression.ModelPriors(**get_hyperparameters())

# These are mode-agnostic or depend on priors.mode
get_guesses = congression.get_guesses
define_model = congression.define_model
guide = congression.guide

# Bake the theta_dist for the update function
update_thetas = partial(congression.update_thetas, theta_dist="logit_norm")
