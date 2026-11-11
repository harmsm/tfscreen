"""
PK: the general dimeric-TF model with every mutation effect learned from data.

Registry name ``thermo.dimer.PK``. Each mutation shifts every term of the
variant by Normal(0, σ_term), σ_term ~ HalfNormal(``theta_sigma_d_{term}
_scale``) (non-centered), or held by ``theta_sigma_d_{term}_fixed`` > 0; effector terms add a per-titrant deviation
when there is more than one titrant. No structural information is used.
Physics: ``species.py``; sites and variant flags: ``model.py``.
"""

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import species as sp
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import model as dm
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
    sigma_fixed_defaults,
    sigma_hyperparameters,
    wt_guesses,
)


# Reads the operator and TF-level axes (tfmodel/contexts.py).
SUPPORTS_CONTEXTS = True

def get_hyperparameters():
    p = common_hyperparameters()
    p.update(sigma_hyperparameters(sp.ALL_TERMS))
    p.update(_STATIC)
    return p


_STATIC = {**flag_defaults(), **sigma_fixed_defaults(sp.ALL_TERMS),
           **dm.epi_fixed_defaults(),
           **dm.context_static_defaults()}
ModelPriors = make_priors_class(
    __name__, [k for k in get_hyperparameters() if k not in _STATIC], _STATIC)


def get_priors(data=None, contexts=None):
    """
    The component's priors; with ``data``, the operator and TF-level prior
    arrays for its labels (``model.context_hyperparameters``), from
    ``contexts`` (a read contexts file) when given.
    """
    p = get_hyperparameters()
    p.update(dm.context_hyperparameters(data, contexts, p["theta_tf_total_M"]))
    return ModelPriors(**p)


def define_model(name, data, priors):
    """The PK model; see ``model.define``."""
    return define(name, data, priors)


def guide(name, data, priors):
    """Mean-field guide matching :func:`define_model`."""
    return define(name, data, priors, guide=True)


def get_guesses(name, data):
    return wt_guesses(name, data)
