from functools import partial
from . import transformation_congression

def get_hyperparameters():
    """
    Gets default values for the model hyperparameters, forced to empirical mode.
    """
    parameters = transformation_congression.get_hyperparameters()
    parameters["mode"] = "empirical"
    return parameters

def get_priors():
    """
    Gets model priors initialized to empirical mode.
    """
    return transformation_congression.ModelPriors(**get_hyperparameters())

# These are mode-agnostic or depend on priors.mode
get_guesses = transformation_congression.get_guesses
define_model = transformation_congression.define_model
guide = transformation_congression.guide

# Bake the theta_dist for the update function
update_thetas = partial(transformation_congression.update_thetas, theta_dist="empirical")
