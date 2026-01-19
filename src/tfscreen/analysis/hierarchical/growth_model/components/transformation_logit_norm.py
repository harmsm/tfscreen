from functools import partial
from . import transformation_congression

def get_hyperparameters():
    """
    Gets default values for the model hyperparameters, forced to logit_norm mode.
    """
    parameters = transformation_congression.get_hyperparameters()
    parameters["mode"] = "logit_norm"
    return parameters

def get_priors():
    """
    Gets model priors initialized to logit_norm mode.
    """
    return transformation_congression.ModelPriors(**get_hyperparameters())

# These are mode-agnostic or depend on priors.mode
get_guesses = transformation_congression.get_guesses
define_model = transformation_congression.define_model
guide = transformation_congression.guide

# Bake the theta_dist for the update function
update_thetas = partial(transformation_congression.update_thetas, theta_dist="logit_norm")
