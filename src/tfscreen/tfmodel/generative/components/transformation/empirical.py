from functools import partial
import pandas as pd
from . import _congression as congression

def get_hyperparameters():
    """
    Gets default values for the model hyperparameters, forced to empirical mode.
    """
    parameters = congression.get_hyperparameters()
    parameters["mode"] = "empirical"
    return parameters

def get_priors():
    """
    Gets model priors initialized to empirical mode.
    """
    return congression.ModelPriors(**get_hyperparameters())

# These are mode-agnostic or depend on priors.mode
get_guesses = congression.get_guesses
define_model = congression.define_model
guide = congression.guide

# Bake the theta_dist for the update function
update_thetas = partial(congression.update_thetas, theta_dist="empirical")


def get_extract_specs(ctx):
    lam_df = pd.DataFrame({"parameter": ["lam"], "map_all": [0]})
    return [dict(
        input_df=lam_df,
        params_to_get=["lam"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="transformation_",
    )]
