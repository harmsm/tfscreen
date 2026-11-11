"""
Shared parts of the thermo.dimer theta components (PK, PddG, PnnC).

The physics is ``species.py``. This module turns a component's priors into
per-genotype terms and back: the variant flags, the priors class, the wt,
per-mutation and per-pair sites (one code path for model and guide), the
assembly, ``run_model`` and the extraction helpers.

Variant flags
-------------
Six static (``pytree_node=False``) bool fields of every ModelPriors pick
the species and terms (``species.Variant``): ``theta_monomers``,
``theta_unfolded_dimer`` (default on), ``theta_unfolded_monomer``,
``theta_second_engagement`` (default on), ``theta_effector_coupling`` and
``theta_operator_coupling``. Set them with ``tfs-configure-model
--set_priors theta_monomers=1``, in the priors CSV, or in a simulate
config's ``theta_priors``.

Sites (``name`` is the component's prefix, "theta")
-----------------------------------------------------
wt, one per active term, ``{name}_{term}_wt`` ~ Normal(loc, scale); per
titrant inside ``{name}_wt_plate`` for the effector terms. The operator
terms are put on the wt *effective* constant, ``{name}_ln_K_h_o_eff_wt``
= ln[K_o (1 + K_oii)], which is what the data see; the first engagement
``{name}_ln_K_h_o_wt`` (deterministic) is that minus softplus(ln K_oii).
So the prior on the wt operator does not depend on whether the second
engagement is in the model.

Per mutation, a shared shift on every active term. PK (and the
non-structural terms of PddG/PnnC): ``{name}_sigma_d`` (K,) ~
HalfNormal(``theta_sigma_d_{term}_scale``) and ``{name}_d_offset``
(K, M) ~ Normal(0, 1), shift = offset · σ. A static
``theta_sigma_d_{term}_fixed`` > 0 holds that term's σ at the value (kT):
the term then has no σ site (``{name}_sigma_d`` covers the free terms
only) and its shift is offset · held σ. Joint MAP does not estimate these
scales (planning/studies/dimer-utility, stage A: σ runs up while the
offsets shrink, or falls to 0 for a term the data do not see), so a MAP
fit should hold them and set them by EM (``dimer_em.py`` in that study). PddG/PnnC put the structural
prior (``thermo/prior.py``) on the six C12 constants (ln_K_h_l, ln_K_h_o,
ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u). With more than one titrant, each
mutation's effect on an effector term is its shared shift plus a
deviation per titrant, ``{name}_d_dev`` (K_t, T, M) ~
Normal(0, ``theta_d_dev_scale``): the study's per-effector remedy (step
5), which ties the effectors together instead of fitting each freely.

Per pair (with epistasis), one shift on every active term, shared across
titrants: a regularized horseshoe, ``{name}_epi_tau``, ``{name}_epi_c2``,
``{name}_epi_lambda`` and ``{name}_epi_offset`` (K, P), with the local
scale's prior HalfCauchy(exp(−d/d0)), d the pair's closest contact over
the structures when the data carry contact distances (PnnC), else 1.
Static ``theta_epi_tau_fixed`` and ``theta_epi_c2_fixed`` > 0 hold the
global scale τ and the slab variance c2 at those values: the site becomes
a deterministic and the guide has no parameters for it. Joint MAP does
not estimate them (planning/studies/dimer-design, tau test, 2026-11-10:
τ runs to about 800 and c2 to about 600 even when the truth has no
epistasis, turning the horseshoe into a free 25 kT slab on every pair),
so a MAP fit should hold both.

Operators and TF levels (``tfmodel/contexts.py``)
-------------------------------------------------
The data's θ conditions are (titrant, operator, TF level); the effector
terms are indexed by effector (``data.theta_effector_names``), not by θ
condition, and ``run_model`` maps each condition to its effector, its
operator and its total TF. Mutation effects are shared across operators
and TF levels.

Each operator other than the reference (``theta_operator_reference``) has
a shift, ``{name}_operator_shift`` = loc + scale · ``{name}_operator_shift_z``
(kT), the ΔΔG of operator binding (positive binds more weakly), subtracted
from ln_K_h_o and ln_K_l_o, so it moves the effective operator constant of
both conformations by the same ΔΔG (one literature or measured number per
operator); a scale of 0 holds it at loc. It also has a deviation
on every active operator term (:data:`OPERATOR_DEV_TERMS`),
``{name}_operator_dev`` = ``theta_operator_dev_scale`` ·
``{name}_operator_dev_z``. Each named TF level has
``{name}_ln_tf_total`` = loc + scale · ``{name}_tf_level_z`` (ln M
protomer). Binding measurements use their known protein concentration
(``data.theta_tf_total_M``); a condition with neither uses
``theta_tf_total_M``. The loc/scale arrays (``theta_operator_shift_loc``,
``theta_operator_shift_scale``, ``theta_tf_level_ln_conc_loc``,
``theta_tf_level_ln_conc_scale``) are indexed by the data's operator and TF
level names, come from the contexts file (``tfs-configure-model
--contexts``) and are written to the priors CSV as rows tagged with
``operator`` or ``tf_level``. Without operator or TF-level labels they are
empty and nothing changes.

Deterministic sites, the names extraction and the simulator's truth
tables use: ``{name}_d_{term}`` ((M,), or (T, M) for effector terms),
``{name}_epi_{term}`` (P,), ``{name}_{term}`` per genotype ((G,) or
(T, G)) and ``{name}_ln_K_h_o_eff``/``{name}_ln_K_l_o_eff`` per genotype.
"""

import math
from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from flax.struct import dataclass, field

from tfscreen.genetics.build_mut_geno_matrix import (
    apply_mut_matrix,
    apply_pair_matrix,
)
from tfscreen.tfmodel.generative.components._horseshoe import (
    HalfCauchy,
    regularized_scale,
)
from tfscreen.tfmodel.contexts import named_labels
from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import species as sp

DEFAULT_TF_TOTAL_M = mwc.DEFAULT_TF_TOTAL_M
DEFAULT_CONC_UNIT_SCALE = mwc.DEFAULT_CONC_UNIT_SCALE
_ZERO_CONC_VALUE = 0.0

# Effective operator constants are what the wt prior sits on.
_OPERATOR_TERMS = {"ln_K_h_o": "ln_K_h_oii", "ln_K_l_o": "ln_K_l_oii"}

# wt prior (loc, scale) per wt site, natural-log units. With every
# optional term off these are the O2_C12 U1 PK defaults (thermo/mwc.py).
# The monomer and Oii values follow the study's lac-like truth
# (planning/studies/thermo-identifiability, step 3): a tight dimer
# (1e10 M⁻¹), monomer half-site binding 1e6 (H) and 1e3 (L) M⁻¹, ln K_oii 4.
WT_PRIORS = {
    "ln_K_h_l": (mwc.DEFAULT_LN_K_H_L, 2.0),
    "ln_K_h_o_eff": (mwc.DEFAULT_LN_K_H_O, 2.0),
    "ln_K_l_o_eff": (mwc.DEFAULT_LN_K_L_O, 3.0),
    "ln_K_u": (-12.0, 3.0),
    "ln_K_h_oii": (4.0, 3.0),
    "ln_K_l_oii": (4.0, 3.0),
    "ln_K_d_h": (math.log(1e10), 3.0),
    "ln_K_m_h_l": (0.5 * mwc.DEFAULT_LN_K_H_L, 2.0),
    "ln_K_m_h_o": (math.log(1e6), 3.0),
    "ln_K_m_l_o": (math.log(1e3), 3.0),
    "ln_K_m_u": (-12.0, 3.0),
    "ln_K_h_e": (mwc.DEFAULT_LN_K_H_E, 2.0),
    "ln_K_l_e": (mwc.DEFAULT_LN_K_L_E, 2.0),
    "ln_K_h_ee": (0.0, 1.0),
    "ln_K_l_ee": (0.0, 1.0),
    "ln_K_h_eo": (0.0, 1.0),
    "ln_K_l_eo": (0.0, 1.0),
}

FLAG_FIELDS = {
    "theta_monomers": "monomers",
    "theta_unfolded_dimer": "unfolded_dimer",
    "theta_unfolded_monomer": "unfolded_monomer",
    "theta_second_engagement": "second_engagement",
    "theta_effector_coupling": "effector_coupling",
    "theta_operator_coupling": "operator_coupling",
}

# The six constants of the O2_C12 structural prior, in the row order of
# prior.projection(mwc.PROJ_C12, mwc.PROJ_U_C12).
STRUCT_TERMS = ("ln_K_h_l", "ln_K_h_o", "ln_K_h_e", "ln_K_l_o", "ln_K_l_e",
                "ln_K_u")


# An operator's shift moves the effective operator constant of each
# conformation through its first engagement; every active operator term
# also takes a per-operator deviation.
OPERATOR_SHIFT_TERMS = ("ln_K_h_o", "ln_K_l_o")
OPERATOR_DEV_TERMS = ("ln_K_h_o", "ln_K_l_o", "ln_K_h_oii", "ln_K_l_oii",
                      "ln_K_m_h_o", "ln_K_m_l_o")

# Priors for an operator or TF level the contexts file does not give (kT,
# ln units): used only when the model is built without one.
DEFAULT_OPERATOR_SHIFT_SD = 2.0
DEFAULT_TF_LEVEL_LN_SD = 1.0

def wt_site(term):
    """The wt sample-site suffix of a term (effective for the operator)."""
    return f"{term}_eff" if term in _OPERATOR_TERMS else term


def variant_from_priors(priors):
    """The :class:`species.Variant` a ModelPriors (or None) selects."""
    defaults = sp.Variant()
    kwargs = {}
    for f, attr in FLAG_FIELDS.items():
        val = getattr(priors, f, getattr(defaults, attr)) if priors is not None \
            else getattr(defaults, attr)
        kwargs[attr] = bool(val)
    return sp.Variant(**kwargs)


# ---------------------------------------------------------------------------
# Priors class
# ---------------------------------------------------------------------------

def common_hyperparameters():
    """Hyperparameters every thermo.dimer variant has."""
    p = {}
    for site, (loc, scale) in WT_PRIORS.items():
        p[f"theta_{site}_wt_loc"] = float(loc)
        p[f"theta_{site}_wt_scale"] = float(scale)
    p["theta_tf_total_M"] = DEFAULT_TF_TOTAL_M     # M, protomer units
    p["theta_conc_unit_scale"] = DEFAULT_CONC_UNIT_SCALE   # mM -> M
    p["theta_d_dev_scale"] = 0.5     # per-titrant deviation of effector effects
    p["theta_operator_dev_scale"] = 0.5   # per-operator deviation, kT
    p["theta_epi_tau_scale"] = 0.1
    p["theta_epi_slab_scale"] = 2.0
    p["theta_epi_slab_df"] = 4.0
    p["theta_epi_d0"] = 8.0          # Å
    # Operator and TF-level prior arrays; empty until get_priors sees data
    # with labels (context_hyperparameters).
    for f in CONTEXT_ARRAY_FIELDS:
        p[f] = jnp.zeros(0)
    return p


def context_hyperparameters(data=None, contexts=None,
                            tf_total_M=DEFAULT_TF_TOTAL_M):
    """
    The operator and TF-level prior arrays for the data's labels.

    Parameters
    ----------
    data : GrowthData or BindingData, optional
        Carries ``theta_operator_names`` and ``theta_tf_level_names``. None
        (or data without labels): empty arrays, no operator or TF axis.
    contexts : dict, optional
        From ``tfmodel.contexts.read_contexts``. Without one, the first
        operator is the reference, every other gets Normal(0,
        :data:`DEFAULT_OPERATOR_SHIFT_SD`), and the TF levels sit at
        ``tf_total_M`` (held for a single level, ln SD
        :data:`DEFAULT_TF_LEVEL_LN_SD` otherwise).

    Returns
    -------
    dict
        The four float arrays and the static ``theta_operator_reference``.
    """
    ops = named_labels(getattr(data, "theta_operator_names", None))
    levels = named_labels(getattr(data, "theta_tf_level_names", None))
    given_ops = (contexts or {}).get("operators") or {}
    given_tf = (contexts or {}).get("tf_levels") or {}

    reference = ""
    shift_loc, shift_scale = [], []
    if ops:
        refs = [o for o in ops if given_ops.get(o, {}).get("reference")]
        reference = refs[0] if refs else ops[0]
        for o in ops:
            if o == reference:
                shift_loc.append(0.0)
                shift_scale.append(0.0)
            elif o in given_ops:
                shift_loc.append(float(given_ops[o]["shift_kT"]))
                shift_scale.append(float(given_ops[o]["shift_sd_kT"]))
            else:
                shift_loc.append(0.0)
                shift_scale.append(DEFAULT_OPERATOR_SHIFT_SD)

    tf_loc, tf_scale = [], []
    for lvl in levels:
        if lvl in given_tf:
            tf_loc.append(math.log(given_tf[lvl]["conc_M"]))
            tf_scale.append(float(given_tf[lvl]["ln_sd"]))
        else:
            tf_loc.append(math.log(tf_total_M))
            tf_scale.append(0.0 if len(levels) == 1 else DEFAULT_TF_LEVEL_LN_SD)

    return {"theta_operator_shift_loc": jnp.asarray(shift_loc, dtype=float),
            "theta_operator_shift_scale": jnp.asarray(shift_scale, dtype=float),
            "theta_tf_level_ln_conc_loc": jnp.asarray(tf_loc, dtype=float),
            "theta_tf_level_ln_conc_scale": jnp.asarray(tf_scale, dtype=float),
            "theta_operator_reference": reference}


CONTEXT_ARRAY_FIELDS = ("theta_operator_shift_loc",
                        "theta_operator_shift_scale",
                        "theta_tf_level_ln_conc_loc",
                        "theta_tf_level_ln_conc_scale")


def context_static_defaults():
    """The static field naming the reference operator."""
    return {"theta_operator_reference": ""}


def sigma_hyperparameters(terms):
    """``theta_sigma_d_{term}_scale`` (HalfNormal scale of σ) per term."""
    return {f"theta_sigma_d_{t}_scale": 1.0 for t in terms}


def sigma_fixed_defaults(terms):
    """``theta_sigma_d_{term}_fixed`` (static; > 0 holds σ, kT) per term."""
    return {f"theta_sigma_d_{t}_fixed": 0.0 for t in terms}


def epi_fixed_defaults():
    """``theta_epi_tau_fixed``/``theta_epi_c2_fixed`` (static; > 0 holds)."""
    return {"theta_epi_tau_fixed": 0.0, "theta_epi_c2_fixed": 0.0}


def held_epi(priors, which):
    """The held pair-horseshoe τ or c2 (``which``), or 0.0 when sampled."""
    return float(getattr(priors, f"theta_epi_{which}_fixed", 0.0))


def held_sigma_d(priors, term):
    """The held σ of a term's shifts (kT), or 0.0 when σ is sampled."""
    return float(getattr(priors, f"theta_sigma_d_{term}_fixed", 0.0))


def flag_defaults():
    """The variant flags and their defaults."""
    v = sp.Variant()
    return {f: bool(getattr(v, attr)) for f, attr in FLAG_FIELDS.items()}


def make_priors_class(module_name, float_fields, static_fields):
    """
    A flax ModelPriors dataclass with the given fields.

    ``float_fields`` are pytree leaves; ``static_fields`` maps name to a
    default and becomes ``pytree_node=False`` fields (hashable under jit).
    The class is named ``ModelPriors`` in ``module_name`` so it pickles.
    """
    ann = {}
    ns = {"__module__": module_name, "__qualname__": "ModelPriors",
          "__doc__": "Hyperparameters of a thermo.dimer theta component "
                     "(thermo/dimer/model.py)."}
    for f in float_fields:
        ann[f] = float
    for f, default in static_fields.items():
        ann[f] = type(default)
        ns[f] = field(pytree_node=False, default=default)
    ns["__annotations__"] = ann
    cls = type("ModelPriors", (), ns)
    return dataclass(frozen=True)(cls)


# ---------------------------------------------------------------------------
# ThetaParam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """
    Per-genotype terms of a thermo.dimer model.

    One field per term (None when the variant lacks it): (G,) for the
    genotype terms, (T, G) for the effector terms.
    """

    ln_K_h_l: jnp.ndarray = None
    ln_K_h_o: jnp.ndarray = None
    ln_K_l_o: jnp.ndarray = None
    ln_K_u: jnp.ndarray = None
    ln_K_h_oii: jnp.ndarray = None
    ln_K_l_oii: jnp.ndarray = None
    ln_K_d_h: jnp.ndarray = None
    ln_K_m_h_l: jnp.ndarray = None
    ln_K_m_h_o: jnp.ndarray = None
    ln_K_m_l_o: jnp.ndarray = None
    ln_K_m_u: jnp.ndarray = None
    ln_K_h_e: jnp.ndarray = None
    ln_K_l_e: jnp.ndarray = None
    ln_K_h_ee: jnp.ndarray = None
    ln_K_l_ee: jnp.ndarray = None
    ln_K_h_eo: jnp.ndarray = None
    ln_K_l_eo: jnp.ndarray = None
    # Effective operator constants K_o (1 + K_oii), (G,); for the truth
    # tables and parameters_df, not used by run_model.
    ln_K_h_o_eff: jnp.ndarray = None
    ln_K_l_o_eff: jnp.ndarray = None
    tf_total: float = DEFAULT_TF_TOTAL_M    # M, protomer units
    # Operator offsets (O, K) on the terms ``operator_terms`` and ln total
    # TF per named level (L,); None without operator or TF-level labels.
    operator_offset: jnp.ndarray = None
    ln_tf_total: jnp.ndarray = None
    operator_terms: tuple = field(pytree_node=False, default=())
    conc_unit_scale: float = field(pytree_node=False,
                                   default=DEFAULT_CONC_UNIT_SCALE)
    variant: sp.Variant = field(pytree_node=False, default=sp.Variant())

    def terms(self):
        """Term name → array for the variant's terms."""
        return {t: getattr(self, t) for t in sp.active_terms(self.variant)}


def _broadcast_terms(terms, variant, genotype_idx=None):
    """
    Terms shaped for a (T, C, G) evaluation: genotype terms (1, 1, G),
    effector terms (T, 1, G); optionally indexed to ``genotype_idx``.
    """
    out = {}
    for t, v in terms.items():
        if genotype_idx is not None:
            v = v[..., genotype_idx]
        # effector terms, and operator terms with offsets, are (T, G)
        out[t] = v[:, None, :] if v.ndim == 2 else v[None, None, :]
    return out


def compute_theta(terms, titrant_conc, tf_total, variant, genotype_idx=None):
    """θ (T, C, G') from per-genotype terms; ``titrant_conc`` (C,) in M."""
    b = _broadcast_terms(terms, variant, genotype_idx)
    return sp.theta(b, jnp.asarray(titrant_conc)[None, :, None], tf_total,
                    variant)


def run_model(theta_param, data):
    """
    θ at the concentrations and genotypes of ``data``.

    Returns
    -------
    (T, C, num_obs), or (1, 1, 1, 1, T, C, num_obs) when
    ``data.scatter_theta`` is 1.
    """
    geno_idx = data.batch_idx[data.geno_theta_idx]
    terms, tf_total = condition_terms(theta_param, data)
    theta_calc = compute_theta(terms,
                               data.titrant_conc * theta_param.conc_unit_scale,
                               tf_total, theta_param.variant,
                               genotype_idx=geno_idx)
    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]
    return theta_calc


def condition_terms(theta_param, data):
    """
    The terms and total TF at each θ condition of ``data``.

    Effector terms (E, G) are taken at each condition's effector; with
    operator offsets, each operator term becomes (T, G) with its
    condition's offset added; total TF is a scalar, or (T, 1, 1) when the
    data carry operator or TF-level labels. Data without the θ-condition
    fields (a simulation's SimData) are read as one condition per titrant.
    """
    terms = dict(theta_param.terms())
    tt = set(sp.titrant_terms(theta_param.variant))
    eff = getattr(data, "theta_effector_idx", None)
    if eff is not None:
        idx = np.asarray(eff, dtype=int)
        for t in tt:
            terms[t] = terms[t][idx]
    tf_total = theta_param.tf_total
    if getattr(data, "theta_operator_idx", None) is None:
        return terms, tf_total

    if theta_param.operator_offset is not None and \
            theta_param.operator_offset.shape[0] > 0:
        op_idx = np.asarray(data.theta_operator_idx, dtype=int)
        off = theta_param.operator_offset[op_idx]            # (T, K)
        for k, t in enumerate(theta_param.operator_terms):
            terms[t] = terms[t][None, :] + off[:, k][:, None]

    lvl = np.asarray(data.theta_tf_level_idx, dtype=int)
    known = np.asarray(data.theta_tf_total_M, dtype=float)
    has_levels = (theta_param.ln_tf_total is not None
                  and theta_param.ln_tf_total.shape[0] > 0
                  and (lvl >= 0).any())
    if not has_levels and not np.isfinite(known).any():
        return terms, tf_total
    tf = jnp.full((len(lvl),), tf_total)
    if has_levels:
        level_tf = jnp.exp(theta_param.ln_tf_total)[np.maximum(lvl, 0)]
        tf = jnp.where(jnp.asarray(lvl >= 0), level_tf, tf)
    if np.isfinite(known).any():
        tf = jnp.where(jnp.asarray(np.isfinite(known)),
                       jnp.asarray(np.nan_to_num(known, nan=1.0)), tf)
    return terms, tf[:, None, None]


def get_population_moments(theta_param, data):
    """
    Not computed: nothing consumes population moments of θ since the
    ``logit_norm`` transformation was removed, and computing them costs a
    full-library θ evaluation per step.
    """
    return None, None


# ---------------------------------------------------------------------------
# Sampling helpers (model and guide share these)
# ---------------------------------------------------------------------------

def _normal(site, loc, scale, guide, shape, plural, loc_init=None):
    """A Normal site, or in the guide its mean-field Normal."""
    if not guide:
        return pyro.sample(site, dist.Normal(loc, scale))
    s = "s" if plural else ""
    init = loc if loc_init is None else loc_init
    q_loc = pyro.param(f"{site}_loc{s}", jnp.broadcast_to(
        jnp.asarray(init, dtype=jnp.float32), shape))
    q_scale = pyro.param(f"{site}_scale{s}", jnp.ones(shape),
                         constraint=constraints.positive)
    return q_loc, q_scale


def _sample_wt(name, priors, variant, T, guide):
    """wt terms: dict term → () or (T,); operator terms first engagement."""
    wt = {}
    gterms = sp.genotype_terms(variant)
    tterms = sp.titrant_terms(variant)
    for t in gterms:
        site = f"{name}_{wt_site(t)}_wt"
        loc = getattr(priors, f"theta_{wt_site(t)}_wt_loc")
        scale = getattr(priors, f"theta_{wt_site(t)}_wt_scale")
        if guide:
            q_loc, q_scale = _normal(site, loc, scale, True, (), False)
            wt[t] = pyro.sample(site, dist.Normal(q_loc, q_scale))
        else:
            wt[t] = _normal(site, loc, scale, False, (), False)
    q = {}
    if guide:
        for t in tterms:
            site = f"{name}_{t}_wt"
            q[t] = _normal(site, getattr(priors, f"theta_{t}_wt_loc"), None,
                           True, (T,), True)
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        for t in tterms:
            site = f"{name}_{t}_wt"
            if guide:
                wt[t] = pyro.sample(site, dist.Normal(*q[t]))
            else:
                wt[t] = pyro.sample(site, dist.Normal(
                    getattr(priors, f"theta_{t}_wt_loc"),
                    getattr(priors, f"theta_{t}_wt_scale")))
    # The wt sites of the operator terms are effective constants.
    for t, oii in _OPERATOR_TERMS.items():
        if variant.second_engagement:
            wt[t] = wt[t] - jnp.logaddexp(0.0, wt[oii])
        if not guide:
            pyro.deterministic(f"{name}_{t}_wt", wt[t])
    return wt


def _sample_free_shifts(name, priors, terms, num_mut, guide):
    """Hierarchical per-mutation shifts for ``terms``: dict term → (M,)."""
    K = len(terms)
    if K == 0:
        return {}
    if num_mut == 0:
        return {t: jnp.zeros((0,)) for t in terms}
    held = [held_sigma_d(priors, t) for t in terms]
    free = [i for i, h in enumerate(held) if not h > 0]
    if guide:
        if free:
            s_loc = pyro.param(f"{name}_sigma_d_locs",
                               jnp.full(len(free), -1.0))
            s_scale = pyro.param(f"{name}_sigma_d_scales",
                                 jnp.full(len(free), 0.1),
                                 constraint=constraints.positive)
        o_loc = pyro.param(f"{name}_d_offset_locs", jnp.zeros((K, num_mut)))
        o_scale = pyro.param(f"{name}_d_offset_scales", jnp.ones((K, num_mut)),
                             constraint=constraints.positive)
    sigma = jnp.asarray(held, dtype=jnp.float32)
    if free:
        scales = jnp.array([getattr(priors, f"theta_sigma_d_{terms[i]}_scale")
                            for i in free])
        with pyro.plate(f"{name}_sigma_d_plate", len(free), dim=-1):
            s_free = pyro.sample(f"{name}_sigma_d",
                                 dist.LogNormal(s_loc, s_scale) if guide
                                 else dist.HalfNormal(scales))
        sigma = sigma.at[jnp.array(free)].set(s_free)
    with pyro.plate(f"{name}_d_term_plate", K, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            offset = pyro.sample(f"{name}_d_offset",
                                 dist.Normal(o_loc, o_scale) if guide
                                 else dist.Normal(0.0, 1.0))
    d = offset * sigma[:, None]
    return {t: d[i] for i, t in enumerate(terms)}


def _sample_deviations(name, priors, terms, T, num_mut, guide):
    """Per-titrant deviations of effector effects: dict term → (T, M)."""
    K = len(terms)
    if K == 0 or T < 2 or num_mut == 0:
        return {t: jnp.zeros((T, num_mut)) for t in terms}
    dev_scale = priors.theta_d_dev_scale
    if guide:
        q_loc = pyro.param(f"{name}_d_dev_locs", jnp.zeros((K, T, num_mut)))
        q_scale = pyro.param(f"{name}_d_dev_scales",
                             jnp.full((K, T, num_mut), dev_scale),
                             constraint=constraints.positive)
    with pyro.plate(f"{name}_dev_term_plate", K, dim=-3):
        with pyro.plate(f"{name}_dev_titrant_plate", T, dim=-2):
            with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
                dev = pyro.sample(f"{name}_d_dev",
                                  dist.Normal(q_loc, q_scale) if guide
                                  else dist.Normal(0.0, dev_scale))
    return {t: dev[i] for i, t in enumerate(terms)}


def _sample_contexts(name, priors, variant, operator_names, guide):
    """
    Operator offsets (O, K) and ln total TF per level (L,), or None.

    Non-centered, so a prior scale of 0 holds the value at its loc without
    a change in the model's structure.
    """
    dev_terms = tuple(t for t in OPERATOR_DEV_TERMS
                      if t in sp.active_terms(variant))
    shift_loc = jnp.asarray(getattr(priors, "theta_operator_shift_loc",
                                    jnp.zeros(0)))
    O = int(shift_loc.shape[0])
    offset = None
    if O > 0:
        names = tuple(operator_names or ())
        ref = str(getattr(priors, "theta_operator_reference", ""))
        if len(names) != O:
            raise ValueError(
                f"The theta priors give {O} operator(s) but the data have "
                f"{list(names)}. Regenerate the priors with "
                f"tfs-configure-model.")
        if ref not in names:
            raise ValueError(f"theta_operator_reference {ref!r} is not one of "
                             f"the data's operators {list(names)}.")
        free = np.array([i for i, n in enumerate(names) if n != ref], int)
        K = len(dev_terms)
        shift = jnp.zeros(O)
        dev = jnp.zeros((O, K))
        if free.size:
            with pyro.plate(f"{name}_operator_plate", free.size, dim=-1):
                z = (pyro.sample(f"{name}_operator_shift_z",
                                 dist.Normal(*_normal(
                                     f"{name}_operator_shift_z", 0.0, None,
                                     True, (free.size,), False)))
                     if guide else
                     pyro.sample(f"{name}_operator_shift_z",
                                 dist.Normal(0.0, 1.0)))
            scale = jnp.asarray(priors.theta_operator_shift_scale)
            shift = shift.at[free].set(shift_loc[free] + scale[free] * z)
            if K:
                if guide:
                    q = _normal(f"{name}_operator_dev_z", 0.0, None, True,
                                (K, free.size), True)
                with pyro.plate(f"{name}_operator_dev_term_plate", K, dim=-2):
                    with pyro.plate(f"{name}_operator_plate", free.size,
                                    dim=-1):
                        zd = pyro.sample(f"{name}_operator_dev_z",
                                         dist.Normal(*q) if guide
                                         else dist.Normal(0.0, 1.0))
                dev = dev.at[free].set(priors.theta_operator_dev_scale * zd.T)
        is_shift = jnp.asarray([t in OPERATOR_SHIFT_TERMS for t in dev_terms],
                               dtype=float)
        # a shift is a ΔΔG of operator binding: positive binds more weakly
        offset = -shift[:, None] * is_shift[None, :] + dev
        if not guide:
            pyro.deterministic(f"{name}_operator_shift", shift)
            pyro.deterministic(f"{name}_operator_dev", dev)

    tf_loc = jnp.asarray(getattr(priors, "theta_tf_level_ln_conc_loc",
                                 jnp.zeros(0)))
    L = int(tf_loc.shape[0])
    ln_tf = None
    if L > 0:
        with pyro.plate(f"{name}_tf_level_plate", L, dim=-1):
            z = (pyro.sample(f"{name}_tf_level_z", dist.Normal(*_normal(
                     f"{name}_tf_level_z", 0.0, None, True, (L,), True)))
                 if guide else
                 pyro.sample(f"{name}_tf_level_z", dist.Normal(0.0, 1.0)))
        ln_tf = tf_loc + jnp.asarray(priors.theta_tf_level_ln_conc_scale) * z
        if not guide:
            pyro.deterministic(f"{name}_ln_tf_total", ln_tf)
    return offset, ln_tf, (dev_terms if offset is not None else ())


def _sample_pair_shifts(name, priors, terms, num_pair, distances, guide):
    """Horseshoe pair shifts on every term: dict term → (P,)."""
    K = len(terms)
    tau_held = held_epi(priors, "tau")
    c2_held = held_epi(priors, "c2")
    if guide:
        if tau_held <= 0:
            tau_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-2.0))
            tau_scale = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.5),
                                   constraint=constraints.positive)
        if c2_held <= 0:
            c2_loc = pyro.param(f"{name}_epi_c2_loc", jnp.array(1.4))
            c2_scale = pyro.param(f"{name}_epi_c2_scale", jnp.array(0.5),
                                  constraint=constraints.positive)
        lam_loc = pyro.param(f"{name}_epi_lambda_locs",
                             jnp.zeros((K, num_pair)))
        lam_scale = pyro.param(f"{name}_epi_lambda_scales",
                               jnp.ones((K, num_pair)),
                               constraint=constraints.positive)
        off_loc = pyro.param(f"{name}_epi_offset_locs",
                             jnp.zeros((K, num_pair)))
        off_scale = pyro.param(f"{name}_epi_offset_scales",
                               jnp.ones((K, num_pair)),
                               constraint=constraints.positive)
        tau = (jnp.asarray(tau_held) if tau_held > 0 else
               pyro.sample(f"{name}_epi_tau",
                           dist.LogNormal(tau_loc, tau_scale)))
        c2 = (jnp.asarray(c2_held) if c2_held > 0 else
              pyro.sample(f"{name}_epi_c2", dist.LogNormal(c2_loc, c2_scale)))
    else:
        if tau_held > 0:
            tau = pyro.deterministic(f"{name}_epi_tau", jnp.asarray(tau_held))
        else:
            tau = pyro.sample(f"{name}_epi_tau",
                              HalfCauchy(priors.theta_epi_tau_scale))
        if c2_held > 0:
            c2 = pyro.deterministic(f"{name}_epi_c2", jnp.asarray(c2_held))
        else:
            df = priors.theta_epi_slab_df
            c2 = pyro.sample(f"{name}_epi_c2", dist.InverseGamma(
                df / 2.0, df * priors.theta_epi_slab_scale ** 2 / 2.0))
        if distances is None:
            prior_lam = jnp.ones((num_pair,))
        else:
            prior_lam = jnp.maximum(
                jnp.exp(-jnp.asarray(distances) / priors.theta_epi_d0), 1e-12)
    with pyro.plate(f"{name}_epi_term_plate", K, dim=-2):
        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            lam = pyro.sample(f"{name}_epi_lambda",
                              dist.LogNormal(lam_loc, lam_scale) if guide
                              else HalfCauchy(prior_lam[None, :]))
            offset = pyro.sample(f"{name}_epi_offset",
                                 dist.Normal(off_loc, off_scale) if guide
                                 else dist.Normal(0.0, 1.0))
    epi = offset * tau * regularized_scale(lam, tau, c2)   # (K, P)
    return {t: epi[i] for i, t in enumerate(terms)}


def define(name, data, priors, guide=False, structural=None,
           pair_distances=None):
    """
    The model (``guide=False``) or mean-field guide of a thermo.dimer model.

    Parameters
    ----------
    name : str
        Site prefix.
    data : GrowthData or BindingData
        Needs the mutation (and, with pairs, pair) scatter indices.
    priors : ModelPriors
    guide : bool
    structural : callable or None
        ``structural(guide) -> dict`` of per-mutation shared shifts (M,) for
        some terms (PddG/PnnC's :data:`STRUCT_TERMS`); those terms get no
        hierarchical prior here. None (PK): every term is hierarchical.
    pair_distances : array (P,) or None
        Closest contact distance of each pair (Å), for the horseshoe scale.

    Returns
    -------
    ThetaParam
    """
    variant = variant_from_priors(priors)
    T = num_effectors(data)
    M = data.num_mutation
    has_pairs = data.num_pair > 0
    gterms = sp.genotype_terms(variant)
    tterms = sp.titrant_terms(variant)
    terms = gterms + tterms

    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)

    wt = _sample_wt(name, priors, variant, T, guide)

    shared = dict(structural(guide)) if structural is not None else {}
    free = [t for t in terms if t not in shared]
    shared.update(_sample_free_shifts(name, priors, free, M, guide))
    dev = _sample_deviations(name, priors, tterms, T, M, guide)

    d = {t: shared[t] for t in gterms}
    for t in tterms:
        d[t] = shared[t][None, :] + dev[t]                    # (T, M)

    epi = {}
    if has_pairs:
        epi = _sample_pair_shifts(name, priors, terms, data.num_pair,
                                  pair_distances, guide)
        pair_scatter = partial(
            apply_pair_matrix,
            pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
            pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
            num_genotype=data.num_genotype)

    values = {}
    for t in terms:
        if t in tterms:
            v = wt[t][:, None] + mut_scatter(d[t])            # (T, G)
            if has_pairs:
                v = v + pair_scatter(epi[t])[None, :]
        else:
            v = wt[t] + mut_scatter(d[t])                     # (G,)
            if has_pairs:
                v = v + pair_scatter(epi[t])
        values[t] = v

    eff = {}
    for t, oii in _OPERATOR_TERMS.items():
        eff[f"{t}_eff"] = values[t]
        if variant.second_engagement:
            eff[f"{t}_eff"] = values[t] + jnp.logaddexp(0.0, values[oii])

    if not guide:
        for t in terms:
            pyro.deterministic(f"{name}_d_{t}", d[t])
            if has_pairs:
                pyro.deterministic(f"{name}_epi_{t}", epi[t])
            pyro.deterministic(f"{name}_{t}", values[t])
        for k, v in eff.items():
            pyro.deterministic(f"{name}_{k}", v)

    operator_offset, ln_tf, op_terms = _sample_contexts(
        name, priors, variant, getattr(data, "theta_operator_names", None),
        guide)

    return ThetaParam(**values, **eff, tf_total=priors.theta_tf_total_M,
                      operator_offset=operator_offset, ln_tf_total=ln_tf,
                      operator_terms=op_terms,
                      conc_unit_scale=priors.theta_conc_unit_scale,
                      variant=variant)


def num_effectors(data):
    """Effectors the effector terms are indexed by (titrants without names)."""
    names = getattr(data, "theta_effector_names", None)
    return len(names) if names is not None else data.num_titrant_name


def wt_guesses(name, data):
    """
    Starting values for the wt sites every variant has.

    Guesses are made before the priors CSV is read (so before the variant
    is known), and a guess for a site the model lacks would not be used;
    the variant-specific sites start at their prior medians instead.
    """
    T = num_effectors(data)
    g = {}
    for t in ("ln_K_h_l", "ln_K_h_o", "ln_K_l_o"):
        g[f"{name}_{wt_site(t)}_wt"] = jnp.array(WT_PRIORS[wt_site(t)][0])
    for t in ("ln_K_h_e", "ln_K_l_e"):
        g[f"{name}_{t}_wt"] = jnp.full(T, WT_PRIORS[t][0])
    return g


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def get_extract_specs(ctx, extra=()):
    """``tfs-extract-params`` tables for a thermo.dimer model."""
    variant = variant_from_priors(getattr(ctx, "theta_priors", None))
    gterms = list(sp.genotype_terms(variant))
    tterms = list(sp.titrant_terms(variant))
    tm = ctx.growth_tm
    num_genotype = len(tm.tensor_dim_labels[tm.tensor_dim_names.index("genotype")])
    titrant_names = list(tm.tensor_dim_labels[tm.tensor_dim_names.index("titrant_name")])
    mut_labels = list(ctx.mut_labels)
    num_mut = len(mut_labels)

    geno_df = tm.df[["genotype", "genotype_idx"]].drop_duplicates().copy()
    geno_df["map_geno"] = geno_df["genotype_idx"]
    specs = [dict(input_df=geno_df,
                  params_to_get=gterms + ["ln_K_h_o_eff", "ln_K_l_o_eff"],
                  map_column="map_geno", get_columns=["genotype"],
                  in_run_prefix="theta_")]

    # Effector terms are (E, G), indexed by effector, not by θ condition.
    conditions = getattr(ctx, "theta_conditions", None)
    effector_names = (list(conditions.effector_names) if conditions is not None
                      else titrant_names)
    te_df = pd.concat([geno_df[["genotype", "genotype_idx"]]
                       .assign(titrant_name=en, effector_idx=e)
                       for e, en in enumerate(effector_names)],
                      ignore_index=True)
    te_df["map_theta_KE"] = te_df["effector_idx"] * num_genotype \
        + te_df["genotype_idx"]
    specs.append(dict(input_df=te_df, params_to_get=tterms,
                      map_column="map_theta_KE",
                      get_columns=["genotype", "titrant_name"],
                      in_run_prefix="theta_"))

    priors = getattr(ctx, "theta_priors", None)
    ops = (named_labels(conditions.operator_names)
           if conditions is not None else ())
    if ops and np.asarray(getattr(priors, "theta_operator_shift_loc",
                                  ())).size == len(ops):
        specs.append(dict(input_df=pd.DataFrame({"operator": ops,
                                                 "map_op": range(len(ops))}),
                          params_to_get=["operator_shift"],
                          map_column="map_op", get_columns=["operator"],
                          in_run_prefix="theta_"))
        dev_terms = [t for t in OPERATOR_DEV_TERMS
                     if t in sp.active_terms(variant)]
        rows = [{"operator": o, "term": t, "map_op_dev": i * len(dev_terms) + k}
                for i, o in enumerate(ops) for k, t in enumerate(dev_terms)]
        specs.append(dict(input_df=pd.DataFrame(rows),
                          params_to_get=["operator_dev"],
                          map_column="map_op_dev",
                          get_columns=["operator", "term"],
                          in_run_prefix="theta_"))
    levels = (named_labels(conditions.tf_level_names)
              if conditions is not None else ())
    if levels and np.asarray(getattr(priors, "theta_tf_level_ln_conc_loc",
                                     ())).size == len(levels):
        specs.append(dict(input_df=pd.DataFrame({"tf_level": levels,
                                                 "map_tf": range(len(levels))}),
                          params_to_get=["ln_tf_total"],
                          map_column="map_tf", get_columns=["tf_level"],
                          in_run_prefix="theta_"))

    if num_mut > 0:
        mut_df = pd.DataFrame({"mutation": mut_labels, "map_mut": range(num_mut)})
        specs.append(dict(input_df=mut_df,
                          params_to_get=[f"d_{t}" for t in gterms],
                          map_column="map_mut", get_columns=["mutation"],
                          in_run_prefix="theta_"))
        rows = [{"titrant_name": tn, "mutation": m,
                 "map_theta_d_KE": ti * num_mut + mi}
                for ti, tn in enumerate(effector_names)
                for mi, m in enumerate(mut_labels)]
        specs.append(dict(input_df=pd.DataFrame(rows),
                          params_to_get=[f"d_{t}" for t in tterms],
                          map_column="map_theta_d_KE",
                          get_columns=["titrant_name", "mutation"],
                          in_run_prefix="theta_"))

    if ctx.pair_labels:
        pair_df = pd.DataFrame({"pair": list(ctx.pair_labels),
                                "map_pair": range(len(ctx.pair_labels))})
        specs.append(dict(input_df=pair_df,
                          params_to_get=[f"epi_{t}" for t in gterms + tterms],
                          map_column="map_pair", get_columns=["pair"],
                          in_run_prefix="theta_"))
    specs.extend(extra)
    return specs


CONTEXT_COLUMNS = ("operator", "tf_level")
_INTERNAL = ["genotype_idx", "titrant_name_idx", "_effector_idx",
             "_operator_idx", "_tf_level_idx", "_tf_total_M"]


def resolve_context_rows(df, conditions):
    """
    Each row's effector, operator and TF-level indices, and known total TF.

    Parameters
    ----------
    df : pandas.DataFrame
        ``titrant_name`` and optionally ``operator``, ``tf_level`` and
        ``tf_total_uM`` (µM protomer, a known concentration, as for
        binding). A column may be left out when the data have one label on
        that axis.
    conditions : tfmodel.contexts.ThetaConditions

    Returns
    -------
    pandas.DataFrame
        ``df`` with ``_effector_idx``, ``_operator_idx`` (into the data's
        operator names; -1 for none), ``_tf_level_idx`` (-1 for none) and
        ``_tf_total_M`` (NaN unless ``tf_total_uM`` was given).
    """
    df = df.copy()
    effectors = list(conditions.effector_names)
    bad = sorted(set(df["titrant_name"].astype(str)) - set(effectors))
    if bad:
        raise ValueError(f"titrant_name values {bad} are not among the model's "
                         f"titrants {effectors}.")
    df["_effector_idx"] = [effectors.index(t)
                           for t in df["titrant_name"].astype(str)]

    def _axis(column, names, what):
        names = list(names)
        if column in df.columns:
            labels = df[column]
            bad = sorted(set(labels[labels.notna()].astype(str)) - set(names))
            if bad:
                raise ValueError(f"{what} {bad} are not among the model's "
                                 f"{names}.")
            return np.array([names.index(str(v)) if pd.notna(v) else -1
                             for v in labels], dtype=int)
        if len(names) > 1:
            return None
        return np.full(len(df), 0 if names else -1, dtype=int)

    ops = list(named_labels(conditions.operator_names))
    op_idx = _axis("operator", ops, "operators")
    if op_idx is None:
        raise ValueError(f"The model has operators {ops}; give each row an "
                         f"'operator'.")
    df["_operator_idx"] = op_idx

    known = (pd.to_numeric(df["tf_total_uM"], errors="coerce").to_numpy(float)
             if "tf_total_uM" in df.columns else np.full(len(df), np.nan))
    df["_tf_total_M"] = known * 1e-6
    levels = list(named_labels(conditions.tf_level_names))
    lvl_idx = _axis("tf_level", levels, "TF levels")
    if lvl_idx is None:
        if not np.isfinite(known).all():
            raise ValueError(f"The model has TF levels {levels}; give each "
                             f"row a 'tf_level' or a 'tf_total_uM'.")
        lvl_idx = np.full(len(df), -1, dtype=int)
    df["_tf_level_idx"] = np.where(np.isfinite(known), -1, lvl_idx)
    return df


def build_calc_df(model, manual_titrant_df):
    """
    Concentration grid for θ extraction.

    Without ``manual_titrant_df``, every (genotype, θ condition,
    concentration) of the training data, with ``titrant_name`` and the
    ``operator``/``tf_level`` columns the data name. A manual grid gives
    ``titrant_name`` and ``titrant_conc``, optionally ``genotype``,
    ``operator``, ``tf_level`` or ``tf_total_uM`` (see
    :func:`resolve_context_rows`).
    """
    import tfscreen.util.dataframe

    conditions = getattr(model, "theta_conditions", None)
    if conditions is None:
        from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U1_a.thermo import (
            build_calc_df as _o2_build_calc_df,
        )
        calc_df, cols, extra = _o2_build_calc_df(model, manual_titrant_df)
        extra["variant"] = variant_from_priors(model.priors.theta)
        return calc_df, cols, extra

    tm_df = model.training_tm.df
    context_cols = [c for c, names in (("operator", conditions.operator_names),
                                       ("tf_level", conditions.tf_level_names))
                    if named_labels(names) and c in tm_df.columns]
    geno_map = (tm_df[["genotype", "genotype_idx"]].drop_duplicates()
                .set_index("genotype")["genotype_idx"].to_dict())
    if manual_titrant_df is None:
        calc_df = (tm_df[["genotype", "titrant_name", *context_cols,
                          "titrant_conc", "genotype_idx"]]
                   .drop_duplicates().reset_index(drop=True))
    else:
        tfscreen.util.dataframe.check_columns(
            manual_titrant_df, required_columns=["titrant_name", "titrant_conc"])
        if "genotype" not in manual_titrant_df.columns:
            calc_df = pd.concat([manual_titrant_df.assign(genotype=g)
                                 for g in tm_df["genotype"].unique()],
                                ignore_index=True)
        else:
            calc_df = manual_titrant_df.copy().reset_index(drop=True)
        calc_df["genotype_idx"] = calc_df["genotype"].map(geno_map)
        if calc_df["genotype_idx"].isna().any():
            bad = calc_df.loc[calc_df["genotype_idx"].isna(),
                              "genotype"].unique()
            raise ValueError(f"Genotypes {list(bad)} in manual_titrant_df are "
                             f"not in the model data.")
        calc_df["genotype_idx"] = calc_df["genotype_idx"].astype(int)
    calc_df = resolve_context_rows(calc_df, conditions)

    extra = {"tf_total": float(model.priors.theta.theta_tf_total_M),
             "conc_unit_scale": float(model.priors.theta.theta_conc_unit_scale),
             "variant": variant_from_priors(model.priors.theta)}
    internal = [c for c in _INTERNAL if c in calc_df.columns]
    return calc_df, internal, extra


def _loader(param_posteriors):
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    def load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return np.asarray(v)
    return load


def _apply_row_contexts(terms, rows, load, tf_total, variant):
    """
    Add each row's operator offset to the operator terms and return the
    per-row total TF: (S, N) arrays, from the posterior's
    ``theta_operator_shift``/``theta_operator_dev``/``theta_ln_tf_total``.
    """
    op = rows["_operator_idx"].to_numpy(int) if "_operator_idx" in rows \
        else np.full(len(rows), -1)
    if (op >= 0).any():
        shift = load("theta_operator_shift")                   # (S, O)
        dev = load("theta_operator_dev")                       # (S, O, K)
        dev_terms = [t for t in OPERATOR_DEV_TERMS
                     if t in sp.active_terms(variant)]
        use = op >= 0
        o = np.maximum(op, 0)
        for k, t in enumerate(dev_terms):
            off = dev[:, o, k] - (shift[:, o] if t in OPERATOR_SHIFT_TERMS
                                  else 0.0)
            terms[t] = terms[t] + np.where(use[None, :], off, 0.0)

    tf = np.full(len(rows), float(tf_total))
    lvl = rows["_tf_level_idx"].to_numpy(int) if "_tf_level_idx" in rows \
        else np.full(len(rows), -1)
    known = rows["_tf_total_M"].to_numpy(float) if "_tf_total_M" in rows \
        else np.full(len(rows), np.nan)
    if (lvl >= 0).any():
        ln_tf = load("theta_ln_tf_total")                      # (S, L)
        tf = np.where((lvl >= 0)[None, :],
                      np.exp(ln_tf[:, np.maximum(lvl, 0)]), tf[None, :])
    if np.isfinite(known).any():
        tf = np.where(np.isfinite(known), np.nan_to_num(known, nan=1.0), tf)
    return terms, tf


def compute_theta_samples(calc_df, param_posteriors, *, tf_total,
                          conc_unit_scale=1.0, variant=None):
    """
    Posterior θ samples (S, N) at the rows of ``calc_df``.

    Reads the per-genotype term sites ``theta_{term}``, each row's effector
    (``_effector_idx``, else ``titrant_name_idx``) and, with operators and
    TF levels, the posterior's operator offsets and total TF.
    """
    variant = sp.Variant() if variant is None else variant
    load = _loader(param_posteriors)
    g = calc_df["genotype_idx"].values.astype(int)
    ti = (calc_df["_effector_idx"] if "_effector_idx" in calc_df
          else calc_df["titrant_name_idx"]).values.astype(int)
    tt = set(sp.titrant_terms(variant))
    terms = {}
    for t in sp.active_terms(variant):
        v = load(f"theta_{t}")
        terms[t] = v[:, ti, g] if t in tt else v[:, g]
    terms, tf = _apply_row_contexts(terms, calc_df, load, tf_total, variant)
    conc = calc_df["titrant_conc"].values.astype(float) * conc_unit_scale
    return sp.theta_np(terms, conc[None, :], tf, variant)


def predict_unmeasured(target_genotypes, titrant_names, manual_titrant_df,
                       mut_labels, pair_labels, param_posteriors, q_to_get, *,
                       tf_total, conc_unit_scale=1.0, theta_priors=None,
                       theta_conditions=None):
    """
    θ quantiles for genotypes assembled from wt, per-mutation and per-pair
    posteriors (genotypes need not be in the training data).

    With ``theta_conditions`` the grid's rows may name an operator, a TF
    level or a known ``tf_total_uM`` (:func:`resolve_context_rows`), and
    the output keeps those columns.
    """
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )

    variant = variant_from_priors(theta_priors)
    target_genotypes = list(target_genotypes)
    mut_mat, pair_mat, is_valid = _build_genotype_indicators(
        target_genotypes, mut_labels, pair_labels)
    if theta_conditions is None:
        calc_df, geno_idx, titrant_idx = _build_prediction_grid(
            target_genotypes, titrant_names, manual_titrant_df)
        out_cols = ["genotype", "titrant_name", "titrant_conc"]
    else:
        grid = manual_titrant_df.drop(columns=["genotype"], errors="ignore")
        calc_df = pd.concat([grid.assign(genotype=gt)
                             for gt in target_genotypes], ignore_index=True)
        calc_df = resolve_context_rows(calc_df, theta_conditions)
        geno_idx = np.repeat(np.arange(len(target_genotypes)), len(grid))
        titrant_idx = calc_df["_effector_idx"].to_numpy(int)
        out_cols = ["genotype", "titrant_name",
                    *[c for c in (*CONTEXT_COLUMNS, "tf_total_uM")
                      if c in calc_df.columns], "titrant_conc"]
    load = _loader(param_posteriors)
    tt = set(sp.titrant_terms(variant))
    has_pairs = len(pair_labels) > 0

    rows = {}
    for t in sp.active_terms(variant):
        wt = load(f"theta_{t}_wt")
        d = load(f"theta_d_{t}")
        if t in tt:                                   # wt (S, E), d (S, E, M)
            v = wt[:, :, None] + np.einsum("stm,nm->stn", d, mut_mat)
            if has_pairs:
                v = v + np.einsum("sp,np->sn", load(f"theta_epi_{t}"),
                                  pair_mat)[:, None, :]
            rows[t] = v[:, titrant_idx, geno_idx]
        else:                                         # wt (S,), d (S, M)
            v = wt[:, None] + np.einsum("sm,nm->sn", d, mut_mat)
            if has_pairs:
                v = v + np.einsum("sp,np->sn", load(f"theta_epi_{t}"), pair_mat)
            rows[t] = v[:, geno_idx]

    rows, tf = _apply_row_contexts(rows, calc_df, load, tf_total, variant)
    conc = calc_df["titrant_conc"].values.astype(float) * conc_unit_scale
    theta_samples = sp.theta_np(rows, conc[None, :], tf, variant)
    theta_samples[:, ~is_valid[geno_idx]] = np.nan

    result_df = calc_df[out_cols].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(theta_samples, q_val, axis=0)
    return result_df
