"""
PddG: the general dimeric-TF model with a structural ΔΔG prior.

Registry name ``thermo.dimer.PddG``. The six O2_C12 constants (ln_K_h_l,
ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u) take the structural prior
of ``thermo/prior.py``: per mutation, Δln K ~ MVN(P·μ, P·diag(σ_s²)·Pᵀ),
μ the per-state ΔΔG table (``thermo_data``, a CSV with columns H, HO, L,
LO, HE2, LE2; extra columns ignored) and P = :data:`DLNK_PROJ`, the C12
projection plus the unfolding row (Δln K_u = ΔΔG_H). Every other term of
the variant (Oii, monomer, coupling terms) gets the PK prior.

Notes on the mapping:

- ΔΔG_HO and ΔΔG_LO set the shift of the *first* operator engagement.
  The structures do not split it from the second engagement, whose shift
  has its own PK prior.
- ln_K_h_l is the dimer's conformational constant, so it is the
  structural row with or without monomers.
- The Δln K_u row is always sampled, so P is square and invertible and
  the post-hoc per-state ΔΔG (``theta_ddG``) is exact. In a variant
  without the unfolded dimer that row reaches no θ: its prior then fixes
  the per-state common mode, as the conditional mean does in O2's U0
  models. Score per-state ΔΔG only with the unfolded dimer in the model.
- The structural shift of an effector term is shared across titrants;
  with more than one titrant each gets a deviation (``model.py``).
"""

import jax.numpy as jnp

from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import species as sp
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import model as dm
from tfscreen.tfmodel.generative.components.theta.thermo.dimer.model import (  # noqa: F401
    ThetaParam,
    build_calc_df,
    compute_theta_samples,
    get_population_moments,
    predict_unmeasured,
    run_model,
)
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    ddG_extract_specs,
    ddG_sigma_fixed,
    dlnK_guesses,
    guide_dlnK,
    projection,
    sample_dlnK,
)

STRUCTURE_NAMES = mwc.STRUCTURE_NAMES_C12

# Δln K (M × 6, rows dm.STRUCT_TERMS) = ΔΔG @ DLNK_PROJ.T.
DLNK_PROJ = projection(mwc.PROJ_C12, mwc.PROJ_U_C12)

_FREE_TERMS = tuple(t for t in sp.ALL_TERMS if t not in dm.STRUCT_TERMS)


def get_hyperparameters():
    p = dm.common_hyperparameters()
    p.update(dm.sigma_hyperparameters(_FREE_TERMS))
    p["theta_ddG_sigma_loc"] = DEFAULT_DDG_SIGMA_LOC
    p["theta_ddG_sigma_scale"] = DEFAULT_DDG_SIGMA_SCALE
    p.update(_static_defaults())
    return p


def _static_defaults():
    s = dm.flag_defaults()
    s.update(dm.sigma_fixed_defaults(_FREE_TERMS))
    s["theta_ddG_sigma_fixed"] = 0.0   # > 0 holds σ_s (kT), thermo/prior.py
    s.update(ddG_sigma_state_fixed_defaults())
    return s


def ddG_sigma_state_fixed_defaults():
    """``theta_ddG_sigma_{state}_fixed`` (static; > 0 holds that σ_s, kT)."""
    return {f"theta_ddG_sigma_{st}_fixed": 0.0 for st in STRUCTURE_NAMES}


def held_ddG_sigma(priors):
    """
    The held σ_s: 0.0 (sampled), one value for every state
    (``theta_ddG_sigma_fixed``) or a tuple in STRUCTURE_NAMES order from the
    per-state ``theta_ddG_sigma_{state}_fixed`` fields, which must then all
    be > 0 (a partial hold is not supported) and exclude the scalar.
    """
    per = [float(getattr(priors, f"theta_ddG_sigma_{st}_fixed", 0.0))
           for st in STRUCTURE_NAMES]
    if not any(v > 0 for v in per):
        return ddG_sigma_fixed(priors)
    if not all(v > 0 for v in per):
        raise ValueError(
            "theta_ddG_sigma_{state}_fixed holds every state or none; got "
            f"{dict(zip(STRUCTURE_NAMES, per))}.")
    if ddG_sigma_fixed(priors) > 0:
        raise ValueError("Set theta_ddG_sigma_fixed or the per-state "
                         "theta_ddG_sigma_{state}_fixed, not both.")
    return tuple(per)


ModelPriors = dm.make_priors_class(
    __name__, [k for k in get_hyperparameters() if k not in _static_defaults()],
    _static_defaults())


def get_priors():
    return ModelPriors(**get_hyperparameters())


def struct_perm(data):
    """Index of each STRUCTURE_NAMES entry in ``data.struct_names``."""
    names = tuple(data.struct_names or ())
    missing = [s for s in STRUCTURE_NAMES if s not in names]
    if missing:
        raise ValueError(
            f"thermo.dimer needs structures {STRUCTURE_NAMES}; struct_names "
            f"{names} lacks {missing}. Extra structures are ignored.")
    return [names.index(s) for s in STRUCTURE_NAMES]


def _prior_means(data):
    return jnp.asarray(data.struct_features)[:, jnp.array(struct_perm(data))]


def _structural(name, data, priors, prior_means, guide_init=None):
    """The ``model.define`` hook sampling the structural block."""
    S = len(STRUCTURE_NAMES)
    M = data.num_mutation

    def block(guide):
        if guide:
            init = prior_means if guide_init is None else guide_init
            dlnK = guide_dlnK(name, M, S, DLNK_PROJ, init,
                              sigma_fixed=held_ddG_sigma(priors))
        else:
            dlnK = sample_dlnK(name, list(STRUCTURE_NAMES), M, prior_means,
                               DLNK_PROJ,
                               sigma_loc=priors.theta_ddG_sigma_loc,
                               sigma_scale=priors.theta_ddG_sigma_scale,
                               sigma_fixed=held_ddG_sigma(priors))
        return {t: dlnK[:, i] for i, t in enumerate(dm.STRUCT_TERMS)}
    return block


def define_model(name, data, priors):
    """The PddG model; see ``model.define``."""
    return dm.define(name, data, priors,
                     structural=_structural(name, data, priors,
                                            _prior_means(data)))


def guide(name, data, priors):
    """Mean-field guide matching :func:`define_model`."""
    return dm.define(name, data, priors, guide=True,
                     structural=_structural(name, data, priors,
                                            _prior_means(data)))


def get_guesses(name, data):
    g = dm.wt_guesses(name, data)
    g.update(dlnK_guesses(name, _prior_means(data), DLNK_PROJ))
    return g


def get_extract_specs(ctx):
    return dm.get_extract_specs(ctx, extra=ddG_extract_specs(ctx, STRUCTURE_NAMES))
