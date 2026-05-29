from functools import partial
import numpy as np
import pandas as pd
from . import _congression as congression

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


def get_extract_specs(ctx):
    lam_df = pd.DataFrame({"parameter": ["lam"], "map_all": [0]})
    specs = [dict(
        input_df=lam_df,
        params_to_get=["lam"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="transformation_",
    )]

    trans_df = (ctx.growth_tm.df[["titrant_name", "titrant_conc",
                                   "titrant_name_idx", "titrant_conc_idx"]]
                .drop_duplicates().copy())
    conc_dim = np.where(np.array(ctx.growth_tm.tensor_dim_names) == "titrant_conc")[0][0]
    num_titrant_conc = len(ctx.growth_tm.tensor_dim_labels[conc_dim])
    trans_df["map_trans"] = (trans_df["titrant_name_idx"] * num_titrant_conc
                             + trans_df["titrant_conc_idx"])
    specs.append(dict(
        input_df=trans_df,
        params_to_get=["mu", "sigma"],
        map_column="map_trans",
        get_columns=["titrant_name", "titrant_conc"],
        in_run_prefix="transformation_",
    ))

    return specs
