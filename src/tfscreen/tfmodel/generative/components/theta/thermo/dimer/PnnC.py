"""
PnnC: the general dimeric-TF model with a structure-MLP ΔΔG prior.

Registry name ``thermo.dimer.PnnC``. As ``PddG``, except the per-state
prior means come from per-structure MLPs on LigandMPNN features
(``thermo/nn.py``; ``thermo_data`` is the HDF5 ensemble with structures
H, HO, L, LO, HE2, LE2, extras ignored), learned as empirical-Bayes
model-side params. When the ensemble carries contact distances, each
pair's horseshoe scale is exp(−d/d0) with d its closest contact over the
six structures. The block stays at the six dimer states with or
without monomers in the variant, so the monomer terms keep the PK prior
(PddG reads them from the table's monomer states).
"""

import jax.numpy as jnp

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import model as dm
from tfscreen.tfmodel.generative.components.theta.thermo.dimer.PddG import (  # noqa: F401
    DLNK_PROJ,
    STRUCTURE_NAMES,
    _FREE_TERMS,
    _structural,
    ddG_sigma_state_fixed_defaults,
    struct_perm,
)
from tfscreen.tfmodel.generative.components.theta.thermo.dimer.model import (  # noqa: F401
    ThetaParam,
    build_calc_df,
    compute_theta_samples,
    get_population_moments,
    predict_unmeasured,
    run_model,
)
from tfscreen.tfmodel.generative.components.theta.thermo.nn import (
    _DEFAULT_HIDDEN_SIZE,
    compute_nn_predictions,
)
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    ddG_extract_specs,
    dlnK_guesses,
)


def _static_defaults():
    s = dm.flag_defaults()
    s.update(dm.sigma_fixed_defaults(_FREE_TERMS))
    s["theta_ddG_sigma_fixed"] = 0.0
    s.update(ddG_sigma_state_fixed_defaults())
    s.update(dm.context_static_defaults())
    s["theta_nn_hidden_size"] = int(_DEFAULT_HIDDEN_SIZE)
    return s


# Reads the operator and TF-level axes (tfmodel/contexts.py).
SUPPORTS_CONTEXTS = True

def get_hyperparameters():
    p = dm.common_hyperparameters()
    p.update(dm.sigma_hyperparameters(_FREE_TERMS))
    p["theta_ddG_sigma_loc"] = DEFAULT_DDG_SIGMA_LOC
    p["theta_ddG_sigma_scale"] = DEFAULT_DDG_SIGMA_SCALE
    p.update(_static_defaults())
    return p


ModelPriors = dm.make_priors_class(
    __name__, [k for k in get_hyperparameters() if k not in _static_defaults()],
    _static_defaults())


def get_priors(data=None, contexts=None):
    """
    The component's priors; with ``data``, the operator and TF-level prior
    arrays for its labels (``model.context_hyperparameters``), from
    ``contexts`` (a read contexts file) when given.
    """
    p = get_hyperparameters()
    p.update(dm.context_hyperparameters(data, contexts, p["theta_tf_total_M"]))
    return ModelPriors(**p)


def _nn_means(name, data, priors):
    perm = struct_perm(data)
    features = jnp.asarray(data.struct_features)[:, jnp.array(perm), :]
    n_chains = tuple(data.struct_n_chains[i] for i in perm)
    return compute_nn_predictions(name, features, list(STRUCTURE_NAMES),
                                  n_chains,
                                  hidden_size=priors.theta_nn_hidden_size)


def _pair_distances(data):
    d = getattr(data, "struct_contact_distances", None)
    if d is None or data.num_pair == 0:
        return None
    return jnp.min(jnp.asarray(d)[:, jnp.array(struct_perm(data))], axis=1)


def _build(name, data, priors, guide):
    means = _nn_means(name, data, priors)
    # The guide's Δln K locations start at 0, the untrained MLP's output.
    init = jnp.zeros((data.num_mutation, len(STRUCTURE_NAMES)))
    return dm.define(name, data, priors, guide=guide,
                     structural=_structural(name, data, priors, means,
                                            guide_init=init,
                                            monomer_states=False),
                     pair_distances=_pair_distances(data))


def define_model(name, data, priors):
    """The PnnC model; see ``model.define``."""
    return _build(name, data, priors, False)


def guide(name, data, priors):
    """Mean-field guide matching :func:`define_model`."""
    return _build(name, data, priors, True)


def get_guesses(name, data):
    g = dm.wt_guesses(name, data)
    g.update(dlnK_guesses(
        name, jnp.zeros((data.num_mutation, len(STRUCTURE_NAMES))), DLNK_PROJ))
    return g


def get_extract_specs(ctx):
    return dm.get_extract_specs(ctx, extra=ddG_extract_specs(ctx, STRUCTURE_NAMES))
