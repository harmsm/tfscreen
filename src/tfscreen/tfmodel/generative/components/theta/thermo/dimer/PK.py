"""
PK: the general dimeric-TF model with every mutation effect learned from data.

Registry name ``thermo.dimer.PK``. Each mutation shifts every term of the
variant by Normal(0, σ_term), σ_term ~ HalfNormal(``theta_sigma_d_{term}
_scale``) (non-centered); effector terms add a per-titrant deviation
when there is more than one titrant. No structural information is used.
Physics: ``species.py``; sites and variant flags: ``model.py``.
"""

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import species as sp
from tfscreen.tfmodel.generative.components.theta.thermo.dimer.model import (  # noqa: F401
    ThetaParam,
    build_calc_df,
    common_hyperparameters,
    compute_theta_samples,
    define,
    flag_defaults,
    get_extract_specs,
    get_population_moments,
    make_priors_class,
    predict_unmeasured,
    run_model,
    sigma_hyperparameters,
    wt_guesses,
)


def get_hyperparameters():
    p = common_hyperparameters()
    p.update(sigma_hyperparameters(sp.ALL_TERMS))
    p.update(flag_defaults())
    return p


_STATIC = flag_defaults()
ModelPriors = make_priors_class(
    __name__, [k for k in get_hyperparameters() if k not in _STATIC], _STATIC)


def get_priors():
    return ModelPriors(**get_hyperparameters())


def define_model(name, data, priors):
    """The PK model; see ``model.define``."""
    return define(name, data, priors)


def guide(name, data, priors):
    """Mean-field guide matching :func:`define_model`."""
    return define(name, data, priors, guide=True)


def get_guesses(name, data):
    return wt_guesses(name, data)
