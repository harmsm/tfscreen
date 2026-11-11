"""
PddG: the general dimeric-TF model with a structural ΔΔG prior.

Registry name ``thermo.dimer.PddG``. The six O2_C12 constants (ln_K_h_l,
ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u) take the structural prior
of ``thermo/prior.py``: per mutation, Δln K ~ MVN(P·μ, P·diag(σ_s²)·Pᵀ),
μ the per-state ΔΔG table (``thermo_data``, a CSV with columns H, HO, L,
LO, HE2, LE2; extra columns ignored) and P = :data:`DLNK_PROJ`, the C12
projection plus the unfolding row (Δln K_u = ΔΔG_H). With monomers in
the variant (``theta_monomers``) the table also supplies the monomer
states H_monomer, L_monomer, HO_monomer and LO_monomer, and the block
takes four more terms (:data:`MONOMER_TERMS`). Every other term of the
variant (Oii, coupling terms) gets the PK prior.

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
- Monomer states (``MONOMER_PROJ``). A dimer state's ΔΔG is the whole
  dimer's (both chains mutated), a monomer state's one chain's, and both
  unfolded states are the reference (ΔΔG_U = 0), the convention of the
  NSF design analysis (2026-11-04: folding = H_monomer, dimerization =
  H − 2 H_monomer). So Δln K_d_h = 2 ΔΔG_Hm − ΔΔG_H (2 H ⇌ H2),
  Δln K_m_h_l = ΔΔG_Hm − ΔΔG_Lm, Δln K_m_h_o = ΔΔG_Hm − ΔΔG_HOm,
  Δln K_m_l_o = ΔΔG_Lm − ΔΔG_LOm. The ten rows are square and
  invertible. With the unfolded monomer as well, Δln K_m_u = ΔΔG_Hm
  = (Δln K_d_h + Δln K_u)/2 is set from the block rather than sampled:
  there is no eleventh state, and both unfolded states sitting at 0 is
  the table's own assumption.
- The effector terms are shared by the monomer and the dimer
  (``species.py``), so they keep the dimer rows; the table's effector
  monomer states are not used.
- PnnC keeps the six-state block (its MLPs predict dimer states only), so
  there the monomer terms keep the PK prior.
"""

import jax.numpy as jnp
import numpy as np

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

# Monomer states the table may carry; read when present
# (io.optional_struct_names) and used when the variant has monomers.
MONOMER_STRUCTURE_NAMES = ("H_monomer", "L_monomer", "HO_monomer",
                           "LO_monomer")
OPTIONAL_STRUCTURE_NAMES = MONOMER_STRUCTURE_NAMES
MONOMER_TERMS = ("ln_K_d_h", "ln_K_m_h_l", "ln_K_m_h_o", "ln_K_m_l_o")

# Rows MONOMER_TERMS, columns STRUCTURE_NAMES + MONOMER_STRUCTURE_NAMES.
MONOMER_PROJ = np.array([
    # H   HO   L   LO  HE2 LE2  Hm   Lm  HOm  LOm
    [-1., 0., 0., 0., 0., 0., 2., 0., 0., 0.],    # Δln K_d_h
    [0., 0., 0., 0., 0., 0., 1., -1., 0., 0.],    # Δln K_m_h_l
    [0., 0., 0., 0., 0., 0., 1., 0., -1., 0.],    # Δln K_m_h_o
    [0., 0., 0., 0., 0., 0., 0., 1., 0., -1.],    # Δln K_m_l_o
])
DLNK_PROJ_MONOMER = np.vstack(
    [np.hstack([DLNK_PROJ, np.zeros((len(DLNK_PROJ), 4))]), MONOMER_PROJ])

_FREE_TERMS = tuple(t for t in sp.ALL_TERMS if t not in dm.STRUCT_TERMS)


def _variant(priors):
    """The variant of a ModelPriors, a dict of priors fields, or None."""
    if isinstance(priors, sp.Variant):
        return priors
    if isinstance(priors, dict):
        defaults = sp.Variant()
        return sp.Variant(**{
            attr: bool(priors.get(f, priors.get(f"theta.{f}",
                                                getattr(defaults, attr))))
            for f, attr in dm.FLAG_FIELDS.items()})
    return dm.variant_from_priors(priors)


def structure_names_for(priors=None):
    """
    The states of the structural block, in its order: STRUCTURE_NAMES, then
    MONOMER_STRUCTURE_NAMES when the variant has monomers. ``priors`` is a
    ModelPriors, a dict of priors fields (e.g. a simulation's
    ``theta_priors``), a Variant or None (the default variant).
    """
    v = _variant(priors)
    return tuple(STRUCTURE_NAMES) + (MONOMER_STRUCTURE_NAMES if v.monomers
                                     else ())


def dlnk_proj_for(priors=None):
    """The block's projection (terms × states) for ``priors``' variant."""
    return DLNK_PROJ_MONOMER if _variant(priors).monomers else DLNK_PROJ


def struct_terms_for(priors=None):
    """The block's sampled terms, in the projection's row order."""
    v = _variant(priors)
    return dm.STRUCT_TERMS + (MONOMER_TERMS if v.monomers else ())


# Reads the operator and TF-level axes (tfmodel/contexts.py).
SUPPORTS_CONTEXTS = True

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
    s.update(dm.epi_fixed_defaults())
    s["theta_ddG_sigma_fixed"] = 0.0   # > 0 holds σ_s (kT), thermo/prior.py
    s.update(ddG_sigma_state_fixed_defaults())
    s.update(dm.context_static_defaults())
    return s


def ddG_sigma_state_fixed_defaults():
    """``theta_ddG_sigma_{state}_fixed`` (static; > 0 holds that σ_s, kT)."""
    return {f"theta_ddG_sigma_{st}_fixed": 0.0
            for st in tuple(STRUCTURE_NAMES) + MONOMER_STRUCTURE_NAMES}


def held_ddG_sigma(priors, names=None):
    """
    The held σ_s: 0.0 (sampled), one value for every state
    (``theta_ddG_sigma_fixed``) or a tuple in ``names`` order (default the
    variant's :func:`structure_names_for`) from the per-state
    ``theta_ddG_sigma_{state}_fixed`` fields, which must then all be > 0
    (a partial hold is not supported) and exclude the scalar.
    """
    names = structure_names_for(priors) if names is None else tuple(names)
    per = [float(getattr(priors, f"theta_ddG_sigma_{st}_fixed", 0.0))
           for st in names]
    if not any(v > 0 for v in per):
        return ddG_sigma_fixed(priors)
    if not all(v > 0 for v in per):
        raise ValueError(
            "theta_ddG_sigma_{state}_fixed holds every state or none; got "
            f"{dict(zip(names, per))}.")
    if ddG_sigma_fixed(priors) > 0:
        raise ValueError("Set theta_ddG_sigma_fixed or the per-state "
                         "theta_ddG_sigma_{state}_fixed, not both.")
    return tuple(per)


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


def struct_perm(data, names=STRUCTURE_NAMES):
    """Index of each ``names`` entry in ``data.struct_names``."""
    have = tuple(data.struct_names or ())
    missing = [s for s in names if s not in have]
    if missing:
        extra = ""
        if set(missing) & set(MONOMER_STRUCTURE_NAMES):
            extra = (" A variant with monomers (theta_monomers) needs the "
                     "monomer states in thermo_data.")
        raise ValueError(
            f"thermo.dimer needs structures {tuple(names)}; struct_names "
            f"{have} lacks {missing}. Extra structures are ignored.{extra}")
    return [have.index(s) for s in names]


def _prior_means(data, priors=None):
    names = structure_names_for(priors)
    return jnp.asarray(data.struct_features)[:, jnp.array(struct_perm(data, names))]


def _structural(name, data, priors, prior_means, guide_init=None,
                monomer_states=True):
    """
    The ``model.define`` hook sampling the structural block.

    ``monomer_states=False`` keeps the six-state block whatever the variant
    (PnnC); ``prior_means`` must then be (M, 6).
    """
    if monomer_states:
        names = structure_names_for(priors)
        proj = dlnk_proj_for(priors)
        terms = struct_terms_for(priors)
    else:
        names, proj, terms = tuple(STRUCTURE_NAMES), DLNK_PROJ, dm.STRUCT_TERMS
    variant = _variant(priors)
    S = len(names)
    M = data.num_mutation
    sigma_fixed = held_ddG_sigma(priors, names)

    def block(guide):
        if guide:
            init = prior_means if guide_init is None else guide_init
            dlnK = guide_dlnK(name, M, S, proj, init, sigma_fixed=sigma_fixed)
        else:
            dlnK = sample_dlnK(name, list(names), M, prior_means, proj,
                               sigma_loc=priors.theta_ddG_sigma_loc,
                               sigma_scale=priors.theta_ddG_sigma_scale,
                               sigma_fixed=sigma_fixed)
        out = {t: dlnK[:, i] for i, t in enumerate(terms)}
        if "ln_K_d_h" in out and variant.unfolded_monomer:
            # ΔΔG_U = 0 for the monomer and the dimer alike (module notes).
            out["ln_K_m_u"] = 0.5 * (out["ln_K_d_h"] + out["ln_K_u"])
        return out
    return block


def define_model(name, data, priors):
    """The PddG model; see ``model.define``."""
    return dm.define(name, data, priors,
                     structural=_structural(name, data, priors,
                                            _prior_means(data, priors)))


def guide(name, data, priors):
    """Mean-field guide matching :func:`define_model`."""
    return dm.define(name, data, priors, guide=True,
                     structural=_structural(name, data, priors,
                                            _prior_means(data, priors)))


def get_guesses(name, data):
    # Made before the priors (so the variant) are known: the six-state
    # block. With monomers its Δln K and σ_s guesses do not fit the sites
    # and are skipped, so those sites start at their prior medians.
    g = dm.wt_guesses(name, data)
    g.update(dlnK_guesses(name, _prior_means(data), DLNK_PROJ))
    return g


def get_extract_specs(ctx):
    names = structure_names_for(getattr(ctx, "theta_priors", None))
    return dm.get_extract_specs(ctx, extra=ddG_extract_specs(ctx, names))
