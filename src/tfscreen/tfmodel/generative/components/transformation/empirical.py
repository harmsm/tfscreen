from functools import partial
import pandas as pd
from . import _congression as congression

def get_hyperparameters(lam_mean=None, lam_std=None):
    """
    Gets default values for the model hyperparameters, forced to empirical mode.

    Parameters
    ----------
    lam_mean, lam_std : float, optional
        Experimentally measured mean/std of lambda (linear space). See
        ``_congression.get_hyperparameters``.
    """
    parameters = congression.get_hyperparameters(lam_mean=lam_mean, lam_std=lam_std)
    parameters["mode"] = "empirical"
    return parameters

def get_priors(lam_mean=None, lam_std=None):
    """
    Gets model priors initialized to empirical mode.

    Parameters
    ----------
    lam_mean, lam_std : float, optional
        Experimentally measured mean/std of lambda (linear space). See
        ``_congression.get_hyperparameters``.
    """
    return congression.ModelPriors(**get_hyperparameters(lam_mean=lam_mean, lam_std=lam_std))

# These are mode-agnostic or depend on priors.mode
get_guesses = congression.get_guesses
define_model = congression.define_model
guide = congression.guide

# Bake the theta_dist for the update function
update_thetas = partial(congression.update_thetas, theta_dist="empirical")

# The empirical background CDF is estimated from raw per-genotype theta
# samples (see _congression._empirical_cdf), so it is only well-calibrated
# when built from the full genotype population rather than a training
# minibatch or a handful of requested genotypes.  jax_model uses this flag to
# decide whether to compute (or expect an externally supplied)
# population-wide theta reference before calling update_thetas.
NEEDS_FULL_POPULATION_THETA = True


def get_extract_specs(ctx):
    lam_df = pd.DataFrame({"parameter": ["lam"], "map_all": [0]})
    return [dict(
        input_df=lam_df,
        params_to_get=["lam"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="transformation_",
    )]
